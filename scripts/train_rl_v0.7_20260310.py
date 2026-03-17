#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL 交易模型训练与回测主脚本
版本: v0.7
日期: 20260310

v0.7 变更 (基于 v0.6):
  - 数据加载: 直接从预计算的 .npz 文件加载 (秒级), 不再逐文件读取 + 在线计算特征
  - 模型架构: 与 v0.6 相同 (Temporal Attention + W=10 时间窗口)
  - 需先运行 prepare_rl_data_v0.2 生成 train_dataset.npz / test_dataset.npz

使用方式:
    python scripts/train_rl_v0.7_20260310.py --config configs/rl_config_v0.7.yaml
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
DATASET_DIR = "/data/projectRL_20260218/data/processed_data/rl_dataset_202603101235"

# ─── 常量 ───────────────────────────────────────────────────────────────────

N_MARKET_FEATURES = 6
N_GLOBAL_FEATURES = 2


# ═══════════════════════════════════════════════════════════════════════════
# 第一部分: 数据加载 (从预计算 npz 直接加载)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradingDatasetV2:
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

    @property
    def n_dates(self):
        return len(self.dates)

    @property
    def n_stocks(self):
        return len(self.stock_ids)


def load_precomputed_dataset(npz_path: str, metadata: dict) -> TradingDatasetV2:
    """从 .npz 文件加载预计算好的数据集"""
    data = np.load(npz_path)
    return TradingDatasetV2(
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
    )


# ═══════════════════════════════════════════════════════════════════════════
# 第二部分: 交易环境 (与 v0.6 相同 — 时间窗口观测)
# ═══════════════════════════════════════════════════════════════════════════

def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class MultiStockTradingEnvV3(gym.Env):
    """
    v0.7 多股票组合交易环境

    观测: 展平向量, 维度 = n_stocks × (W × 6 + 1) + 2
      per-stock:
        temporal (W×6): 最近 W 步的 [pred_ret_120d, ret_1w, ret_1m, ret_3m, vol, price_vs_ma]
        static   (1):   current_weight
      global (2): cash_ratio, portfolio_return

    动作: (n_stocks,) 原始分数 → softmax → 组合权重
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset: TradingDatasetV2, initial_cash=10_000_000.0,
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

        self.market_features = np.stack([
            dataset.predicted_return_120d,
            dataset.return_1w,
            dataset.return_1m,
            dataset.return_3m,
            dataset.volatility_1m,
            dataset.price_vs_ma60,
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
        scores[~valid] = -1e9
        if valid.sum() == 0:
            return np.zeros(self.n_stocks, dtype=np.float64)
        weights = _softmax(scores / self.temperature)
        for _ in range(10):
            over = weights > self.max_weight_per_stock
            if not over.any():
                break
            weights[over] = self.max_weight_per_stock
            total = weights.sum()
            if total > 0:
                weights = weights / total
        return weights

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
# 第三部分: Temporal Attention 策略网络 (与 v0.6 相同)
# ═══════════════════════════════════════════════════════════════════════════

class TemporalScoringMlpExtractor(nn.Module):
    """
    时序 Attention 打分网络。

    Per-stock Temporal Attention (共享参数):
      input_proj(6 → embed) + pos_encoding
      → MultiHeadSelfAttention over W time steps
      → LayerNorm + FFN + LayerNorm
      → 取最后一步输出 → stock_embedding

    Policy: concat(embedding, current_weight) → score_head → per-stock score
    Value:  mean_pool(embeddings) + global → value_mlp
    """

    def __init__(self, feature_dim, n_stocks, temporal_window,
                 n_market_features=6, hidden_dim=64, n_heads=2):
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
    """v0.7 自定义策略: Temporal Attention + Identity action_net"""

    def __init__(self, observation_space, action_space, lr_schedule,
                 n_stocks=100, temporal_window=10, n_market_features=6,
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
# 第四部分: 训练回调
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
# 第五部分: 回测
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


def run_backtest(env, model, normalize_env=None):
    obs, _ = env.reset()
    done = False
    dates = [env.dataset.dates[env.current_step]]
    values = [env.portfolio_value]
    turnovers = []

    while not done:
        if normalize_env:
            obs_norm = normalize_env.normalize_obs(obs)
            action, _ = model.predict(obs_norm, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        dates.append(info["date"])
        values.append(info["portfolio_value"])
        turnovers.append(info["turnover"])

    return np.array(dates), np.array(values), np.array(turnovers)


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
    parser = argparse.ArgumentParser(description="RL 交易模型训练 v0.7")
    parser.add_argument("--config", type=str,
                        default=str(project_root / "configs" / "rl_config_v0.7.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    run_ts = datetime.now().strftime("%Y%m%d%H%M%S")

    print("=" * 70)
    print(f"  RL 交易模型 v0.7 (Temporal Attention + Fast Load) — {run_ts}")
    print("=" * 70)
    print_gpu_info()

    # ─── 1. 加载预计算数据 ───
    print("\n[Step 1] 加载预计算数据...")
    t0 = time.time()
    data_cfg = config["data"]
    dataset_dir = Path(DATASET_DIR or data_cfg["dataset_dir"])

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
    print(f"  训练集: {train_dataset.n_dates} 期 ({train_dataset.dates[0]} ~ {train_dataset.dates[-1]})")
    if test_dataset:
        print(f"  测试集: {test_dataset.n_dates} 期 ({test_dataset.dates[0]} ~ {test_dataset.dates[-1]})")
    print(f"  拆分日期: {metadata.get('split_date', 'N/A')}")

    if train_dataset.n_dates < 20:
        print("错误: 训练集数据不足")
        sys.exit(1)

    train_ppy = estimate_periods_per_year(train_dataset.dates)
    test_ppy = estimate_periods_per_year(test_dataset.dates) if test_dataset else train_ppy
    n_stocks = train_dataset.n_stocks

    rl_cfg = config["rl"]
    W = rl_cfg.get("temporal_window", 10)
    print(f"  时间窗口: W={W}, 频率: {train_ppy:.0f}/year")

    # ─── 2. 输出目录 ───
    output_dir = project_root / "experiments" / f"rl_ppo_v0.7_{run_ts}"
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
    print("\n[Step 2] 创建环境...")
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
            env = MultiStockTradingEnvV3(dataset=dataset, **env_kwargs)
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
    print(f"  观测维度: {obs_dim:,} (= {n_stocks} × ({W}×6+1) + 2)")

    # ─── 4. 创建 PPO ───
    print("\n[Step 3] 创建 PPO (Temporal Attention Policy)...")
    episode_len = train_dataset.n_dates - max(W, 4) - 1
    n_steps = min(rl_cfg["n_steps"], episode_len)
    batch_size = min(rl_cfg["batch_size"], n_steps)

    policy_kwargs = dict(
        n_stocks=n_stocks,
        temporal_window=W,
        n_market_features=N_MARKET_FEATURES,
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
    attn_params = (
        sum(p.numel() for p in model.policy.mlp_extractor.input_proj.parameters())
        + model.policy.mlp_extractor.pos_encoding.numel()
        + sum(p.numel() for p in model.policy.mlp_extractor.temporal_attn.parameters())
        + sum(p.numel() for p in model.policy.mlp_extractor.attn_norm.parameters())
        + sum(p.numel() for p in model.policy.mlp_extractor.ffn.parameters())
        + sum(p.numel() for p in model.policy.mlp_extractor.ffn_norm.parameters())
    )
    print(f"  总可训练参数: {total_params:,}")
    print(f"  其中 Temporal Attention: {attn_params:,}")
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
    print("\n[Step 5] 回测...")

    def process_backtest(dataset, ppy, prefix):
        if not dataset or dataset.n_dates < 3:
            return None

        env = MultiStockTradingEnvV3(dataset=dataset, **env_kwargs)

        norm_env = None
        if isinstance(train_vec_env, VecNormalize):
            _dummy = DummyVecEnv([lambda: env])
            norm_env = VecNormalize.load(str(model_dir / "vec_normalize.pkl"), _dummy)
            norm_env.training = False
            norm_env.norm_reward = False

        dates, values, turnovers = run_backtest(env, model, norm_env)
        bh_values = compute_buy_and_hold_v2(env)
        min_len = min(len(values), len(bh_values))

        metrics = compute_metrics(values[:min_len], dates[:min_len], periods_per_year=ppy)
        bh_metrics = compute_metrics(bh_values[:min_len], dates[:min_len], periods_per_year=ppy)

        print_backtest_report(metrics, bh_metrics,
                              label=f"RL Agent ({prefix})", benchmark_label="等权买入持有")
        plot_backtest(
            dates[:min_len], values[:min_len], bh_values[:min_len],
            save_path=str(results_dir / f"{prefix}_backtest.png"),
            title=f"{prefix.capitalize()} Backtest (v0.7 Temporal Attention)",
        )
        pd.DataFrame({
            "date": dates[:min_len], "portfolio_value": values[:min_len],
            "benchmark_value": bh_values[:min_len],
            "turnover": np.concatenate([[0], turnovers[:min_len - 1]]),
        }).to_excel(str(results_dir / f"{prefix}_details.xlsx"), index=False)
        return metrics

    process_backtest(train_dataset, train_ppy, "train")
    if test_dataset:
        process_backtest(test_dataset, test_ppy, "test")

    print(f"\n{'=' * 70}")
    print(f"  完成! 结果: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
