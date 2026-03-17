#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL 交易模型训练与回测主脚本
版本: v0.9
日期: 20260311

v0.9 变更 (基于 v0.81):
  - 输入特征增强: 引入模型历史预测误差统计量 (Bias & Uncertainty)
  - 新增: predicted_close (预测目标价)，归一化后作为第 9 个市场特征
  - N_MARKET_FEATURES: 6 -> 9
  - 核心逻辑 (Top-100 + Cash Anchor + LastKnownPrice) 保持不变。

数据粒度说明:
  - 数据粒度由 prepare_rl_data_v0.3 的 --resample 参数决定。
  - 若生成 npz 时传入 --resample W-FRI，则每行=1周，temporal_window=10 即 10 周。
  - 若不传 --resample，则每行=1 交易日，temporal_window=10 即 10 个交易日。
  - 训练脚本直接使用 npz，不做重采样。

使用方式:
    python scripts/train_rl_v0.9_20260311.py --config configs/rl_config_v0.9.yaml
    数据路径: 在脚本中设置 DATASET_DIR
"""

import json
import sys
import time
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy

from src.utils.backtest import (
    compute_metrics,
    estimate_periods_per_year,
    print_backtest_report,
    plot_backtest,
)

# ─── 数据目录 (手动修改此处切换数据源) ────────────────────────────────────
# 注意: 这里需要指向 v0.3 生成的数据集目录
DATASET_DIR = "/data/projectRL_20260218/data/processed_data/rl_dataset_v0.3_202603111056" 

# ─── 常量 ───────────────────────────────────────────────────────────────────

N_MARKET_FEATURES = 9  # v0.9: 8 -> 9 (新增 predicted_close)
PREDICTED_CLOSE_SCALE = 1000.0  # 预测目标价归一化除数，避免绝对价格量级过大
N_GLOBAL_FEATURES = 2
MAX_HOLDINGS_IN_EXCEL = 30
MAX_CLEARED_IN_EXCEL = 20
MIN_WEIGHT_THRESHOLD = 0.0
TOP_K_STOCKS = 100


# ═══════════════════════════════════════════════════════════════════════════
# 第一部分: 数据加载 (v0.9 增强)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradingDatasetV3:
    dates: np.ndarray
    stock_ids: list
    stock_codes: list
    stock_names: list
    actual_close: np.ndarray
    predicted_return_120d: np.ndarray
    valid_mask: np.ndarray
    return_1w: np.ndarray
    return_1m: np.ndarray
    return_3m: np.ndarray
    volatility_1m: np.ndarray
    price_vs_ma60: np.ndarray
    step_returns: np.ndarray
    error_mean: np.ndarray  # v0.9 新增
    error_std: np.ndarray   # v0.9 新增
    predicted_close: np.ndarray  # v0.9 新增，预测目标价 (原始值)

    @property
    def n_dates(self):
        return len(self.dates)

    @property
    def n_stocks(self):
        return len(self.stock_ids)


def load_precomputed_dataset(npz_path: str, metadata: dict) -> TradingDatasetV3:
    """从 .npz 文件加载预计算好的数据集 (v0.3 格式)"""
    data = np.load(npz_path)
    
    # 兼容旧版数据: 如果没有 error_mean/std，则填充 0
    if "error_mean" in data:
        error_mean = data["error_mean"]
        error_std = data["error_std"]
    else:
        print("警告: 数据集中未找到 error_mean/std，将使用全 0 填充 (退化为 v0.81)")
        shape = data["actual_close"].shape
        error_mean = np.zeros(shape, dtype=np.float32)
        error_std = np.zeros(shape, dtype=np.float32)

    # 兼容旧版数据: 如果没有 predicted_close，则填充 0 (模型将退化为无预测目标价)
    if "predicted_close" in data:
        predicted_close = data["predicted_close"]
    else:
        print("警告: 数据集中未找到 predicted_close，将使用全 0 填充")
        predicted_close = np.zeros(data["actual_close"].shape, dtype=np.float32)

    return TradingDatasetV3(
        dates=data["dates"],
        stock_ids=metadata["stock_ids"],
        stock_codes=metadata["stock_codes"],
        stock_names=metadata["stock_names"],
        actual_close=data["actual_close"],
        predicted_return_120d=data["predicted_return_120d"],
        valid_mask=data["valid_mask"],
        return_1w=data["return_1w"],
        return_1m=data["return_1m"],
        return_3m=data["return_3m"],
        volatility_1m=data["volatility_1m"],
        price_vs_ma60=data["price_vs_ma60"],
        step_returns=np.clip(data["step_returns"], -0.5, 0.5),
        error_mean=error_mean,
        error_std=error_std,
        predicted_close=predicted_close,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 第二部分: 交易环境 (v0.9 增强)
# ═══════════════════════════════════════════════════════════════════════════

def _softmax_with_anchor(scores, anchor_score=0.0):
    max_score = max(np.max(scores), anchor_score)
    e_scores = np.exp(scores - max_score)
    e_anchor = np.exp(anchor_score - max_score)
    
    total = np.sum(e_scores) + e_anchor
    weights = e_scores / total
    cash_weight = e_anchor / total
    return weights, cash_weight


class MultiStockTradingEnvV5(gym.Env):
    """
    v0.9 多股票组合交易环境
    变更: 观测空间包含 error_mean、error_std、predicted_close (归一化)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset: TradingDatasetV3, initial_cash=10_000_000.0,
                 transaction_cost=0.001, max_weight_per_stock=0.10,
                 softmax_temperature=1.0, reward_type="log_return",
                 turnover_penalty=0.5, temporal_window=10):
        super().__init__()
        self.dataset = dataset
        self.n_stocks = dataset.n_stocks
        self.n_dates = dataset.n_dates
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_weight_per_stock = max_weight_per_stock
        self.temperature = softmax_temperature
        self.reward_type = reward_type
        self.turnover_penalty = turnover_penalty
        self.W = temporal_window
        self.lookback = max(temporal_window, 4)

        # 预测目标价归一化: 除以 PREDICTED_CLOSE_SCALE，避免绝对价格 (3~500) 量级过大
        predicted_close_norm = np.nan_to_num(dataset.predicted_close, 0.0) / PREDICTED_CLOSE_SCALE

        # v0.9: 堆叠 9 个特征
        self.market_features = np.stack([
            dataset.predicted_return_120d,
            dataset.return_1w,
            dataset.return_1m,
            dataset.return_3m,
            dataset.volatility_1m,
            dataset.price_vs_ma60,
            dataset.error_mean,  # Bias
            dataset.error_std,   # Uncertainty
            predicted_close_norm,  # 预测目标价 (已归一化)
        ], axis=-1).astype(np.float32)

        n_per_stock = self.W * N_MARKET_FEATURES + 1
        obs_dim = self.n_stocks * n_per_stock + N_GLOBAL_FEATURES
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.n_stocks,), dtype=np.float32
        )

        self.weights = np.zeros(self.n_stocks, dtype=np.float64)
        self.cash_ratio = 1.0
        self.portfolio_value = initial_cash
        self.current_step = self.lookback

    def _get_obs(self):
        t = self.current_step
        W = self.W

        start = max(0, t - W + 1)
        window = self.market_features[start:t + 1]

        actual_len = window.shape[0]
        if actual_len < W:
            pad = np.zeros((W - actual_len, self.n_stocks, N_MARKET_FEATURES), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)

        temporal_flat = window.transpose(1, 0, 2).reshape(self.n_stocks, -1)

        stock_obs = np.concatenate(
            [temporal_flat, self.weights[:, np.newaxis].astype(np.float32)], axis=1
        )

        global_features = np.array([
            self.cash_ratio,
            (self.portfolio_value / self.initial_cash) - 1.0,
        ], dtype=np.float32)

        obs = np.concatenate([stock_obs.flatten(), global_features])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        obs = np.clip(obs, -10.0, 10.0)
        return obs

    def _action_to_weights(self, action):
        scores = action.astype(np.float64).copy()
        valid = self.dataset.valid_mask[self.current_step]
        scores[~valid] = -np.inf
        
        if valid.sum() == 0:
            return np.zeros(self.n_stocks, dtype=np.float64)

        n_valid = valid.sum()
        k = min(TOP_K_STOCKS, n_valid)
        
        if k < self.n_stocks:
            top_k_indices = np.argpartition(scores, -k)[-k:]
            mask = np.ones_like(scores, dtype=bool)
            mask[top_k_indices] = False
            scores[mask] = -np.inf

        stock_weights, cash_weight_from_softmax = _softmax_with_anchor(
            scores / self.temperature, 
            anchor_score=0.0
        )
        
        over = stock_weights > self.max_weight_per_stock
        if over.any():
            stock_weights[over] = self.max_weight_per_stock
            
        return stock_weights

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback
        self.weights = np.zeros(self.n_stocks, dtype=np.float64)
        self.cash_ratio = 1.0
        self.portfolio_value = self.initial_cash
        return self._get_obs(), {}

    def step(self, action):
        new_weights = self._action_to_weights(action)
        new_cash_ratio = 1.0 - new_weights.sum()
        
        if new_cash_ratio < 0:
            new_cash_ratio = 0.0
            new_weights = new_weights / new_weights.sum()

        turnover = np.abs(new_weights - self.weights).sum()
        turnover += abs(new_cash_ratio - self.cash_ratio)
        trade_cost = turnover * self.transaction_cost

        self.weights = new_weights
        self.cash_ratio = new_cash_ratio
        self.current_step += 1
        terminated = self.current_step >= self.n_dates - 1
        truncated = False

        t = min(self.current_step, self.n_dates - 1)
        stock_returns = self.dataset.step_returns[t]
        portfolio_return = np.dot(self.weights, stock_returns) - trade_cost
        self.portfolio_value *= (1.0 + portfolio_return)

        if self.reward_type == "log_return":
            reward = np.log1p(np.clip(portfolio_return, -0.99, None))
        else:
            reward = portfolio_return
        reward -= self.turnover_penalty * trade_cost

        drifted = self.weights * (1.0 + stock_returns)
        total_after = drifted.sum() + self.cash_ratio
        if total_after > 1e-10:
            self.weights = drifted / total_after
            self.cash_ratio = self.cash_ratio / total_after
        else:
            self.weights = np.zeros(self.n_stocks)
            self.cash_ratio = 1.0

        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "turnover": turnover,
            "trade_cost": trade_cost,
            "date": self.dataset.dates[t],
            "cash_ratio": self.cash_ratio,
            "n_active_stocks": int((self.weights > 1e-6).sum()),
        }
        return self._get_obs(), float(reward), terminated, truncated, info


# ═══════════════════════════════════════════════════════════════════════════
# 第三部分: Temporal Attention 策略网络 (保持不变，参数适配)
# ═══════════════════════════════════════════════════════════════════════════

class TemporalScoringMlpExtractor(nn.Module):
    def __init__(self, feature_dim, n_stocks, temporal_window,
                 n_market_features=9, hidden_dim=64, n_heads=2):
        super().__init__()
        self.n_stocks = n_stocks
        self.W = temporal_window
        self.n_market = n_market_features
        self.n_per_stock = temporal_window * n_market_features + 1
        embed_dim = hidden_dim // 2

        self.input_proj = nn.Linear(n_market_features, embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, temporal_window, embed_dim) * 0.02
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True, dropout=0.0
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

        self.score_head = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(embed_dim + N_GLOBAL_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

        self.latent_dim_pi = n_stocks
        self.latent_dim_vf = embed_dim

    def _parse_obs(self, features):
        B = features.shape[0]
        n = self.n_stocks * self.n_per_stock
        stock_flat = features[:, :n].reshape(B, self.n_stocks, self.n_per_stock)

        temporal = stock_flat[:, :, :self.W * self.n_market].reshape(
            B, self.n_stocks, self.W, self.n_market
        )
        current_weight = stock_flat[:, :, -1:]
        global_obs = features[:, n:]
        return temporal, current_weight, global_obs

    def _temporal_encode(self, temporal):
        B, N, W, F = temporal.shape
        x = temporal.reshape(B * N, W, F)

        h = self.input_proj(x) + self.pos_encoding[:, :W, :]

        attn_out, _ = self.temporal_attn(h, h, h)
        h = self.attn_norm(h + attn_out)
        h = self.ffn_norm(h + self.ffn(h))

        embeddings = h[:, -1, :]
        return embeddings.reshape(B, N, -1)

    def forward(self, features):
        temporal, current_weight, global_obs = self._parse_obs(features)
        embeddings = self._temporal_encode(temporal)

        score_in = torch.cat([embeddings, current_weight], dim=-1)
        scores = self.score_head(score_in).squeeze(-1)

        pooled = embeddings.mean(dim=1)
        value_input = torch.cat([pooled, global_obs], dim=1)
        vf = self.value_mlp(value_input)

        return scores, vf

    def forward_actor(self, features):
        temporal, current_weight, _ = self._parse_obs(features)
        embeddings = self._temporal_encode(temporal)
        score_in = torch.cat([embeddings, current_weight], dim=-1)
        scores = self.score_head(score_in).squeeze(-1)
        return scores

    def forward_critic(self, features):
        temporal, _, global_obs = self._parse_obs(features)
        embeddings = self._temporal_encode(temporal)
        pooled = embeddings.mean(dim=1)
        value_input = torch.cat([pooled, global_obs], dim=1)
        return self.value_mlp(value_input)


class TemporalScoringPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 n_stocks=100, temporal_window=10, n_market_features=9,
                 hidden_dim=64, n_heads=2, **kwargs):
        self._n_stocks = n_stocks
        self._temporal_window = temporal_window
        self._n_market = n_market_features
        self._hidden_dim = hidden_dim
        self._n_heads = n_heads
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = TemporalScoringMlpExtractor(
            self.features_dim, self._n_stocks, self._temporal_window,
            self._n_market, self._hidden_dim, self._n_heads,
        )

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )


# ═══════════════════════════════════════════════════════════════════════════
# 第四部分: 训练回调 (保持不变)
# ═══════════════════════════════════════════════════════════════════════════

class TrainingLogCallback(BaseCallback):
    def __init__(self, total_timesteps, n_steps=512, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps
        self._start_time = None
        self._last_n_lines = 0

    def _on_training_start(self):
        self._start_time = time.time()

    def _fmt_time(self, seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s" if h > 0 else f"{m}m{s:02d}s"

    def _on_rollout_end(self):
        elapsed = time.time() - self._start_time if self._start_time else 0
        pct = self.num_timesteps / self.total_timesteps * 100
        eta = self._fmt_time(elapsed / max(pct, 0.01) * (100 - pct)) if pct > 0 else "---"

        ep_rew = float("nan")
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            ep_rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer])

        logger = self.model.logger.name_to_value if hasattr(self.model.logger, "name_to_value") else {}
        loss = logger.get("train/loss", float("nan"))
        ev = logger.get("train/explained_variance", float("nan"))

        filled = int(30 * pct / 100)
        bar = "█" * filled + "░" * (30 - filled)

        lines = [
            f"  [{bar}] {pct:5.1f}%  steps={self.num_timesteps}/{self.total_timesteps}",
            f"  elapsed={self._fmt_time(elapsed)}  eta={eta}  "
            f"ep_rew={ep_rew:.4f}  loss={loss:.4f}  ev={ev:+.3f}"
            if not np.isnan(loss) else
            f"  elapsed={self._fmt_time(elapsed)}  eta={eta}  ep_rew={ep_rew:.4f}",
        ]

        buf = ""
        if self._last_n_lines > 0:
            buf += f"\033[{self._last_n_lines}A\r"
        for line in lines:
            buf += f"\033[2K{line}\n"
        sys.stdout.write(buf)
        sys.stdout.flush()
        self._last_n_lines = len(lines)

    def _on_training_end(self):
        sys.stdout.write("\n")

    def _on_step(self):
        return True


class TrainingHistoryCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.history = []
        self._start = None

    def _on_training_start(self):
        self._start = time.time()

    def _on_rollout_end(self):
        logger = self.model.logger.name_to_value if hasattr(self.model.logger, "name_to_value") else {}
        ep_rew = float("nan")
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            ep_rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
        self.history.append({
            "timesteps": self.num_timesteps,
            "elapsed": time.time() - self._start if self._start else 0,
            "ep_rew_mean": ep_rew,
            "loss": logger.get("train/loss", float("nan")),
            "value_loss": logger.get("train/value_loss", float("nan")),
            "policy_gradient_loss": logger.get("train/policy_gradient_loss", float("nan")),
            "explained_variance": logger.get("train/explained_variance", float("nan")),
            "approx_kl": logger.get("train/approx_kl", float("nan")),
        })

    def _on_step(self):
        return True

    def _on_training_end(self):
        if self.history:
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.history).to_excel(self.save_path, index=False)
            print(f"  训练历史已保存: {self.save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 第五部分: 回测 (v0.81 逻辑 + v0.9 数据)
# ═══════════════════════════════════════════════════════════════════════════

def compute_buy_and_hold_v2(env):
    ds = env.dataset
    start = env.lookback
    values = [env.initial_cash]
    for t in range(start + 1, ds.n_dates):
        valid = ds.valid_mask[t] & ds.valid_mask[t - 1]
        n_valid = valid.sum()
        ret = np.sum(ds.step_returns[t, valid]) / n_valid if n_valid > 0 else 0.0
        values.append(values[-1] * (1.0 + ret))
    return np.array(values)


def run_detailed_backtest(env, model, normalize_env=None):
    obs, _ = env.reset()
    done = False
    dataset = env.dataset
    initial_cash = env.initial_cash

    stock_codes = np.array(dataset.stock_codes)
    code_to_name = {c: n for c, n in zip(dataset.stock_codes, dataset.stock_names)}
    code_to_idx = {c: i for i, c in enumerate(dataset.stock_codes)}

    holdings = {}
    realized_pnl_cum = 0.0
    last_known_prices = np.full(dataset.n_stocks, np.nan, dtype=np.float32)

    prev_weights_map = {}
    prev_shares_map = {}

    records = []
    all_dates = [dataset.dates[env.current_step]]
    all_values = [env.portfolio_value]
    all_turnovers = []

    t_start = min(env.current_step, env.n_dates - 1)
    init_prices = dataset.actual_close[t_start]
    valid_init = ~np.isnan(init_prices) & (init_prices > 0)
    last_known_prices[valid_init] = init_prices[valid_init]

    while not done:
        if normalize_env:
            obs_norm = normalize_env.normalize_obs(obs)
            action, _ = model.predict(obs_norm, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)

        next_obs, reward, terminated, truncated, info = env.step(action)

        t = min(env.current_step, env.n_dates - 1)
        
        current_raw_prices = dataset.actual_close[t]
        valid_price_mask = ~np.isnan(current_raw_prices) & (current_raw_prices > 0)
        last_known_prices[valid_price_mask] = current_raw_prices[valid_price_mask]
        current_prices = last_known_prices 

        current_total_asset = info["portfolio_value"]
        current_weights = env.weights.copy()
        target_values = current_total_asset * current_weights

        step_realized_pnl = 0.0
        current_weights_map = {}
        current_shares_map = {}

        for i in range(len(stock_codes)):
            if np.isnan(current_prices[i]) or current_prices[i] <= 0:
                continue

            code = stock_codes[i]
            price = current_prices[i]
            w = current_weights[i]
            
            is_significant = (w > 0.0)
            
            target_val = target_values[i]
            target_share = target_val / price if price > 0 else 0.0

            old_pos = holdings.get(code, {'shares': 0.0, 'avg_cost': 0.0})
            old_share = old_pos['shares']
            diff_share = target_share - old_share

            if is_significant:
                current_weights_map[code] = w
                current_shares_map[code] = target_share
                if abs(diff_share) > 1e-6:
                    if diff_share > 0:
                        cost_basis = old_pos['avg_cost'] * old_share + price * diff_share
                        new_avg_cost = cost_basis / target_share if target_share > 1e-10 else price
                        holdings[code] = {'shares': target_share, 'avg_cost': new_avg_cost}
                    else:
                        sell_share = min(abs(diff_share), old_share)
                        pnl = (price - old_pos['avg_cost']) * sell_share
                        step_realized_pnl += pnl
                        holdings[code] = {'shares': target_share, 'avg_cost': old_pos['avg_cost']}
                elif code not in holdings:
                    holdings[code] = {'shares': target_share, 'avg_cost': price}
            else:
                if code in holdings:
                    pnl = (price - old_pos['avg_cost']) * old_share
                    step_realized_pnl += pnl
                    del holdings[code]

        realized_pnl_cum += step_realized_pnl

        unrealized_pnl = 0.0
        valid_holdings = []

        for code, pos in holdings.items():
            idx = code_to_idx.get(code)
            if idx is None: continue
            curr_price = current_prices[idx]
            
            if np.isnan(curr_price) or curr_price <= 0:
                continue

            market_val = pos['shares'] * curr_price
            u_pnl = (curr_price - pos['avg_cost']) * pos['shares']
            unrealized_pnl += u_pnl

            weight = market_val / current_total_asset if current_total_asset > 0 else 0.0
            prev_weight = prev_weights_map.get(code, 0.0)
            delta_weight = weight - prev_weight

            prev_share = prev_shares_map.get(code, 0.0)
            buy_sell_amount = (pos['shares'] - prev_share) * curr_price

            valid_holdings.append({
                'name': code_to_name.get(code, code),
                'weight': weight,
                'delta_weight': delta_weight,
                'buy_sell_amount': buy_sell_amount,
            })

        valid_holdings.sort(key=lambda x: x['weight'], reverse=True)

        cleared_stocks = []
        for code, prev_w in prev_weights_map.items():
            if code not in current_weights_map:
                cleared_stocks.append({
                    'name': code_to_name.get(code, code),
                    'prev_weight': prev_w,
                })
        cleared_stocks.sort(key=lambda x: x['prev_weight'], reverse=True)

        pos_details_flat = {}
        for rank, h in enumerate(valid_holdings[:MAX_HOLDINGS_IN_EXCEL]):
            prefix = f"持仓_{rank + 1}"
            pos_details_flat[f"{prefix}_名称"] = h['name']
            pos_details_flat[f"{prefix}_权重"] = h['weight']
            pos_details_flat[f"{prefix}_权重变化"] = h['delta_weight']
            pos_details_flat[f"{prefix}_买卖金额"] = h['buy_sell_amount']

        for rank, c in enumerate(cleared_stocks[:MAX_CLEARED_IN_EXCEL]):
            prefix = f"清仓_{rank + 1}"
            pos_details_flat[f"{prefix}_名称"] = c['name']
            pos_details_flat[f"{prefix}_原权重"] = c['prev_weight']

        record = {
            "日期": info["date"],
            "总资产": current_total_asset,
            "总盈亏(Asset-Init)": current_total_asset - initial_cash,
            "已实现盈亏(累计)": realized_pnl_cum,
            "浮动盈亏(当前)": unrealized_pnl,
            "交易成本(当日)": info["trade_cost"],
            "换手率": info["turnover"],
            "现金比例": info["cash_ratio"],
            "持仓股票数": len(current_weights_map),
            "清仓股票数": len(cleared_stocks),
        }
        record.update(pos_details_flat)
        records.append(record)

        all_dates.append(info["date"])
        all_values.append(info["portfolio_value"])
        all_turnovers.append(info["turnover"])

        prev_weights_map = current_weights_map.copy()
        prev_shares_map = current_shares_map.copy()
        obs = next_obs
        done = terminated or truncated

    df_details = pd.DataFrame(records)
    return df_details, np.array(all_dates), np.array(all_values), np.array(all_turnovers)


# ═══════════════════════════════════════════════════════════════════════════
# 第六部分: 主流程
# ═══════════════════════════════════════════════════════════════════════════

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_gpu_info():
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"  GPU: {gpu.name}  显存: {gpu.total_memory / 1024**3:.1f} GB  CUDA: {torch.version.cuda}")
    else:
        print("  未检测到 GPU，使用 CPU")


def main():
    parser = argparse.ArgumentParser(description="RL 交易模型训练 v0.9")
    parser.add_argument("--config", type=str,
                        default=str(project_root / "configs" / "rl_config_v0.9.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    run_ts = datetime.now().strftime("%Y%m%d%H%M%S")

    print("=" * 70)
    print(f"  RL 交易模型 v0.9 (Bias & Uncertainty Enhanced) — {run_ts}")
    print("=" * 70)
    print_gpu_info()

    # ─── 1. 加载预计算数据 ───
    print("\n[Step 1] 加载预计算数据...")
    t0 = time.time()
    
    # 优先使用脚本中定义的 DATASET_DIR，如果不存在则回退到 config
    dataset_dir_str = DATASET_DIR
    if not dataset_dir_str or not Path(dataset_dir_str).exists():
        config_dir = config["data"].get("dataset_dir", "")
        if config_dir and Path(config_dir).exists():
            print(f"  注意: 脚本内置路径不存在，使用配置文件路径")
            dataset_dir_str = config_dir
        else:
            raise FileNotFoundError(
                f"请先在训练脚本中设置 DATASET_DIR (当前: {dataset_dir_str})，"
                "或确保 config 中 dataset_dir 指向有效的 v0.3 数据集目录"
            )
    dataset_dir = Path(dataset_dir_str)
    print(f"  数据目录: {dataset_dir}")

    meta_path = dataset_dir / "metadata.json"
    train_npz = dataset_dir / "train_dataset.npz"
    test_npz = dataset_dir / "test_dataset.npz"

    if not meta_path.exists():
        raise FileNotFoundError(f"未找到 metadata.json: {meta_path}")
    if not train_npz.exists():
        raise FileNotFoundError(f"未找到 train_dataset.npz: {train_npz}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    train_dataset = load_precomputed_dataset(str(train_npz), metadata)
    test_dataset = None
    if test_npz.exists():
        test_dataset = load_precomputed_dataset(str(test_npz), metadata)

    load_time = time.time() - t0
    print(f"  加载耗时: {load_time:.2f}s")
    print(f"  股票数: {train_dataset.n_stocks}")
    print(f"  训练集: {train_dataset.n_dates} 期")
    if test_dataset:
        print(f"  测试集: {test_dataset.n_dates} 期")
    
    # 检查新特征是否有效
    if np.all(train_dataset.error_mean == 0) and np.all(train_dataset.error_std == 0):
        print("  警告: error_mean/std 全为 0，请确认是否使用了 v0.3 数据集")
    else:
        print("  特征检查: error_mean/std 已加载 (非全 0)")
    if np.all(train_dataset.predicted_close == 0):
        print("  警告: predicted_close 全为 0，请确认 npz 由 v0.3 脚本生成且含 predicted_close")
    else:
        print("  特征检查: predicted_close 已加载 (非全 0)")

    train_ppy = estimate_periods_per_year(train_dataset.dates)
    test_ppy = estimate_periods_per_year(test_dataset.dates) if test_dataset else train_ppy
    n_stocks = train_dataset.n_stocks

    rl_cfg = config["rl"]
    W = rl_cfg.get("temporal_window", 10)
    print(f"  时间窗口: W={W}, 频率: {train_ppy:.0f}/year")
    print(f"  输入特征数: {N_MARKET_FEATURES} (含 Bias, Uncertainty, predicted_close)")

    # ─── 2. 输出目录 ───
    output_dir = project_root / "experiments" / f"rl_ppo_v0.9_{run_ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(exist_ok=True)

    with open(configs_dir / "rl_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    # ─── 3. 创建环境 ───
    print("\n[Step 2] 创建环境 (v0.9 MultiStockTradingEnvV5)...")
    trading_cfg = config["trading"]
    reward_cfg = config["reward"]

    device = rl_cfg.get("device", "auto")
    if device == "cuda" and not torch.cuda.is_available():
        print("  警告: cuda 不可用，回退到 cpu")
        device = "cpu"

    env_kwargs = dict(
        initial_cash=trading_cfg["initial_cash"],
        transaction_cost=trading_cfg["transaction_cost"],
        max_weight_per_stock=trading_cfg["max_weight_per_stock"],
        softmax_temperature=trading_cfg["softmax_temperature"],
        reward_type=reward_cfg["type"],
        turnover_penalty=reward_cfg["turnover_penalty"],
        temporal_window=W,
    )

    def make_env(dataset, seed=0):
        def _init():
            env = MultiStockTradingEnvV5(dataset=dataset, **env_kwargs)
            env.reset(seed=seed)
            return Monitor(env)
        return _init

    train_vec_env = DummyVecEnv([make_env(train_dataset, seed=42)])

    if rl_cfg.get("normalize_observations", True):
        train_vec_env = VecNormalize(
            train_vec_env, norm_obs=True,
            norm_reward=rl_cfg.get("normalize_rewards", False),
            clip_obs=10.0, clip_reward=10.0,
        )

    obs_dim = train_vec_env.observation_space.shape[0]
    print(f"  观测维度: {obs_dim:,} (= {n_stocks} × ({W}×{N_MARKET_FEATURES}+1) + 2)")

    # ─── 4. 创建 PPO ───
    print("\n[Step 3] 创建 PPO (Temporal Attention Policy)...")
    episode_len = train_dataset.n_dates - max(W, 4) - 1
    n_steps = min(rl_cfg["n_steps"], episode_len)
    batch_size = min(rl_cfg["batch_size"], n_steps)

    policy_kwargs = dict(
        n_stocks=n_stocks,
        temporal_window=W,
        n_market_features=N_MARKET_FEATURES,  # 9
        hidden_dim=rl_cfg.get("hidden_dim", 64),
        n_heads=rl_cfg.get("n_heads", 2),
    )

    model = PPO(
        policy=TemporalScoringPolicy,
        env=train_vec_env,
        learning_rate=rl_cfg["learning_rate"],
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=rl_cfg["n_epochs"],
        gamma=rl_cfg["gamma"],
        gae_lambda=rl_cfg["gae_lambda"],
        clip_range=rl_cfg["clip_range"],
        ent_coef=rl_cfg["ent_coef"],
        vf_coef=rl_cfg["vf_coef"],
        max_grad_norm=rl_cfg["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        verbose=0, seed=42, device=device,
    )

    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"  总可训练参数: {total_params:,}")
    print(f"  Batch: {batch_size}, N Steps: {n_steps}, Episode Length: {episode_len}")

    # ─── 5. 训练 ───
    total_ts = rl_cfg["total_timesteps"]
    print(f"\n[Step 4] 开始训练 ({total_ts:,} timesteps)...")

    log_cb = TrainingLogCallback(total_timesteps=total_ts, n_steps=n_steps)
    hist_cb = TrainingHistoryCallback(save_path=str(results_dir / "training_metrics.xlsx"))
    callbacks = CallbackList([log_cb, hist_cb])

    model.learn(total_timesteps=total_ts, callback=callbacks, progress_bar=False)
    print("  训练完成!")

    # ─── 6. 保存 ───
    model.save(str(model_dir / "ppo_trading"))
    if isinstance(train_vec_env, VecNormalize):
        train_vec_env.save(str(model_dir / "vec_normalize.pkl"))
    print(f"  模型已保存: {model_dir}")

    # ─── 7. 回测 ───
    print("\n[Step 5] 回测 (生成详细交易明细)...")

    def process_backtest(dataset, ppy, prefix):
        if not dataset or dataset.n_dates < 3:
            print(f"  {prefix}: 数据不足, 跳过")
            return None

        env = MultiStockTradingEnvV5(dataset=dataset, **env_kwargs)

        norm_env = None
        if isinstance(train_vec_env, VecNormalize):
            _dummy = DummyVecEnv([lambda: env])
            norm_env = VecNormalize.load(str(model_dir / "vec_normalize.pkl"), _dummy)
            norm_env.training = False
            norm_env.norm_reward = False

        df_details, dates, values, turnovers = run_detailed_backtest(env, model, norm_env)

        detail_path = results_dir / f"{prefix}_trade_details.xlsx"
        df_details.to_excel(str(detail_path), index=False)
        print(f"  {prefix} 交易明细已保存: {detail_path}")

        bh_values = compute_buy_and_hold_v2(env)
        min_len = min(len(values), len(bh_values))

        metrics = compute_metrics(values[:min_len], dates[:min_len], periods_per_year=ppy)
        bh_metrics = compute_metrics(bh_values[:min_len], dates[:min_len], periods_per_year=ppy)

        print_backtest_report(metrics, bh_metrics,
                              label=f"RL Agent ({prefix})", benchmark_label="等权买入持有")
        plot_backtest(
            dates[:min_len], values[:min_len], bh_values[:min_len],
            save_path=str(results_dir / f"{prefix}_backtest.png"),
            title=f"{prefix.capitalize()} Backtest (v0.9 Bias & Uncertainty)",
        )
        return metrics

    process_backtest(train_dataset, train_ppy, "train")
    if test_dataset:
        process_backtest(test_dataset, test_ppy, "test")

    print(f"\n{'=' * 70}")
    print(f"  完成! 结果: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
