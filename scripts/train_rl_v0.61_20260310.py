#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL 交易模型训练与回测主脚本
版本: v0.61
日期: 20260310

v0.61 变更 (基于 v0.6):
  - 回测输出增强: 新增 run_detailed_backtest, 生成包含完整交易明细的 Excel
    (总资产/已实现盈亏/浮动盈亏/每只持仓股票名称·权重·权重变化·仓位变化金额)
  - 数据加载、环境、网络架构、训练流程与 v0.6 完全相同

使用方式:
    python scripts/train_rl_v0.61_20260310.py --config configs/rl_config_v0.6.yaml
"""

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
from datetime import datetime, timedelta

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
MERGED_DATA_DIR = "/data/projectRL_20260218/data/processed_data/emerge_real_pre_close_price_202603101234"

# ─── 常量 ───────────────────────────────────────────────────────────────────

N_MARKET_FEATURES = 6   # 有时间历史的市场特征数
N_GLOBAL_FEATURES = 2   # cash_ratio, portfolio_return
MAX_HOLDINGS_IN_EXCEL = 30  # Excel 中最多展示的持仓股票数
MAX_CLEARED_IN_EXCEL = 20   # Excel 中最多展示的清仓股票数
MIN_WEIGHT_THRESHOLD = 1e-3  # 有效持仓阈值 (0.1%)


# ═══════════════════════════════════════════════════════════════════════════
# 第一部分: 数据加载与特征计算 (与 v0.6 相同)
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
    auto_split_date: str = None

    @property
    def n_dates(self):
        return len(self.dates)

    @property
    def n_stocks(self):
        return len(self.stock_ids)

    def split_by_date(self, split_date: str):
        train_idx = np.where(self.dates <= split_date)[0]
        test_idx = np.where(self.dates > split_date)[0]

        def _slice(idx):
            if len(idx) == 0:
                return None
            return TradingDatasetV2(
                dates=self.dates[idx], stock_ids=self.stock_ids,
                stock_codes=self.stock_codes, stock_names=self.stock_names,
                actual_close=self.actual_close[idx],
                predicted_return_120d=self.predicted_return_120d[idx],
                valid_mask=self.valid_mask[idx],
                return_1w=self.return_1w[idx], return_1m=self.return_1m[idx],
                return_3m=self.return_3m[idx],
                volatility_1m=self.volatility_1m[idx],
                price_vs_ma60=self.price_vs_ma60[idx],
                step_returns=self.step_returns[idx],
                auto_split_date=self.auto_split_date,
            )
        return _slice(train_idx), _slice(test_idx)


def _compute_period_return(close, period, valid_mask):
    ret = np.zeros_like(close)
    if close.shape[0] <= period:
        return ret
    prev = close[:-period]
    curr = close[period:]
    valid = valid_mask[period:] & valid_mask[:-period] & (prev > 0)
    np.divide(curr - prev, prev, out=ret[period:], where=valid)
    ret[period:][~valid] = 0.0
    return ret


def load_merged_data(data_dir: str, min_data_points: int = 100):
    data_path = Path(data_dir)
    stocks_dir = data_path / "stocks"
    meta_path = data_path / "stock_list.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"未找到 stock_list.csv: {meta_path}")

    meta_df = pd.read_csv(meta_path)
    meta_df = meta_df.sort_values("stock_id").reset_index(drop=True)
    meta_df = meta_df[meta_df["n_raw_dates"] >= min_data_points].reset_index(drop=True)
    print(f"  通过过滤的股票: {len(meta_df)} (min_data_points={min_data_points})")

    all_stock_data = {}
    all_dates_set = set()
    for _, row in meta_df.iterrows():
        sid = str(row["stock_id"])
        code = str(row["stock_code"])
        fpath = stocks_dir / f"{sid}_{code}.parquet"
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath, columns=["date", "actual_close", "predicted_return_120d", "split"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        all_stock_data[sid] = df
        all_dates_set.update(df["date"].values)

    dates_dt = np.array(sorted(all_dates_set))
    dates_str = np.array([str(d)[:10] for d in pd.to_datetime(dates_dt)])
    n_dates = len(dates_str)
    n_stocks = len(all_stock_data)
    print(f"  日期范围: {dates_str[0]} ~ {dates_str[-1]}, 共 {n_dates} 天")
    print(f"  股票数量: {n_stocks}")

    actual_close = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
    predicted_return_120d = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)

    stock_ids, stock_codes, stock_names = [], [], []
    date_index = {d: i for i, d in enumerate(dates_dt)}

    for si, (sid, df) in enumerate(all_stock_data.items()):
        meta_row = meta_df[meta_df["stock_id"].astype(str) == sid].iloc[0]
        stock_ids.append(sid)
        stock_codes.append(str(meta_row["stock_code"]))
        stock_names.append(str(meta_row["company_name"]))
        indices = np.array([date_index[d] for d in df["date"].values if d in date_index])
        df_filtered = df[df["date"].isin(date_index.keys())]
        actual_close[indices, si] = df_filtered["actual_close"].values
        pred_vals = df_filtered["predicted_return_120d"].values
        valid_pred = ~np.isnan(pred_vals)
        predicted_return_120d[indices[valid_pred], si] = pred_vals[valid_pred]

    valid_mask = ~np.isnan(actual_close) & (actual_close > 0)

    pred_df = pd.DataFrame(predicted_return_120d)
    predicted_return_120d = pred_df.ffill().values
    predicted_return_120d = np.nan_to_num(predicted_return_120d, 0.0)

    print("  计算特征...")
    return_1w = _compute_period_return(actual_close, 5, valid_mask)
    return_1m = _compute_period_return(actual_close, 20, valid_mask)
    return_3m = _compute_period_return(actual_close, 60, valid_mask)
    step_returns = _compute_period_return(actual_close, 1, valid_mask)

    vol_df = pd.DataFrame(step_returns.copy())
    volatility_1m = np.nan_to_num(vol_df.rolling(20, min_periods=10).std().values, 0.0)

    close_df = pd.DataFrame(actual_close)
    ma60 = close_df.rolling(60, min_periods=30).mean().values
    price_vs_ma60 = np.zeros_like(actual_close)
    valid_ma = (ma60 > 0) & valid_mask
    np.divide(actual_close - ma60, ma60, out=price_vs_ma60, where=valid_ma)
    price_vs_ma60[~valid_ma] = 0.0

    return_1w = np.clip(np.nan_to_num(return_1w, 0.0), -1.0, 1.0)
    return_1m = np.clip(np.nan_to_num(return_1m, 0.0), -1.0, 1.0)
    return_3m = np.clip(np.nan_to_num(return_3m, 0.0), -2.0, 2.0)
    volatility_1m = np.clip(volatility_1m, 0, 1.0)
    price_vs_ma60 = np.clip(np.nan_to_num(price_vs_ma60, 0.0), -1.0, 1.0)
    predicted_return_120d = np.clip(predicted_return_120d, -2.0, 2.0)
    step_returns = np.clip(np.nan_to_num(step_returns, 0.0), -0.5, 0.5)

    first_val_dates = []
    for sid, df in all_stock_data.items():
        if "split" in df.columns:
            val_rows = df[df["split"] == "val"]
            if len(val_rows) > 0:
                first_val_dates.append(pd.Timestamp(val_rows["date"].iloc[0]))
    auto_split_date = None
    if first_val_dates:
        median_val = pd.Series(first_val_dates).median()
        auto_split_date = (median_val - timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  自动检测分割日期: {auto_split_date} (基于 {len(first_val_dates)} 只股票的 val 起始日中位数)")

    dataset = TradingDatasetV2(
        dates=dates_str, stock_ids=stock_ids, stock_codes=stock_codes,
        stock_names=stock_names, actual_close=actual_close,
        predicted_return_120d=predicted_return_120d, valid_mask=valid_mask,
        return_1w=return_1w, return_1m=return_1m, return_3m=return_3m,
        volatility_1m=volatility_1m, price_vs_ma60=price_vs_ma60,
        step_returns=step_returns, auto_split_date=auto_split_date,
    )
    print(f"  数据覆盖率: {valid_mask.sum() / valid_mask.size * 100:.1f}%")
    return dataset


def resample_dataset_v2(dataset: TradingDatasetV2, rule="W-FRI"):
    raw_dates = pd.to_datetime(dataset.dates)
    s = pd.Series(np.arange(len(raw_dates)), index=raw_dates)
    idx = s.resample(rule).last().dropna().astype(int).values
    if len(idx) == 0:
        raise ValueError("降采样后没有剩余数据")
    print(f"  降采样: {len(dataset.dates)} → {len(idx)} ({len(idx)/len(dataset.dates):.1%})")
    return TradingDatasetV2(
        dates=dataset.dates[idx], stock_ids=dataset.stock_ids,
        stock_codes=dataset.stock_codes, stock_names=dataset.stock_names,
        actual_close=dataset.actual_close[idx],
        predicted_return_120d=dataset.predicted_return_120d[idx],
        valid_mask=dataset.valid_mask[idx],
        return_1w=dataset.return_1w[idx], return_1m=dataset.return_1m[idx],
        return_3m=dataset.return_3m[idx],
        volatility_1m=dataset.volatility_1m[idx],
        price_vs_ma60=dataset.price_vs_ma60[idx],
        step_returns=np.clip(_compute_period_return(
            dataset.actual_close[idx], 1, dataset.valid_mask[idx]), -0.5, 0.5),
        auto_split_date=dataset.auto_split_date,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 第二部分: 交易环境 (v0.6 — 时间窗口观测)
# ═══════════════════════════════════════════════════════════════════════════

def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class MultiStockTradingEnvV3(gym.Env):
    """
    v0.6 多股票组合交易环境

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
# 第三部分: Temporal Attention 策略网络
# ═══════════════════════════════════════════════════════════════════════════

class TemporalScoringMlpExtractor(nn.Module):
    """
    替换 SB3 的 MlpExtractor，使用时序 Attention 打分。

    数据流:
      obs → parse → temporal (B, n_stocks, W, 6) + weight (B, n_stocks, 1) + global (B, 2)
          ↓
      Per-stock Temporal Attention (共享参数):
        input_proj(6 → embed) + pos_encoding
        → MultiHeadSelfAttention over W time steps
        → LayerNorm + FFN + LayerNorm
        → 取最后一步输出 → stock_embedding (embed_dim,)
          ↓
      Policy: concat(embedding, current_weight) → score_head → score per stock
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

        # ─── Temporal Attention Block (共享, per-stock) ───
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

        # ─── Score Head ───
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        # ─── Value Path ───
        self.value_mlp = nn.Sequential(
            nn.Linear(embed_dim + N_GLOBAL_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

        # SB3 接口
        self.latent_dim_pi = n_stocks
        self.latent_dim_vf = embed_dim

    def _parse_obs(self, features):
        B = features.shape[0]
        n = self.n_stocks * self.n_per_stock
        stock_flat = features[:, :n].reshape(B, self.n_stocks, self.n_per_stock)

        temporal = stock_flat[:, :, :self.W * self.n_market].reshape(
            B, self.n_stocks, self.W, self.n_market
        )
        current_weight = stock_flat[:, :, -1:]  # (B, n_stocks, 1)
        global_obs = features[:, n:]            # (B, 2)
        return temporal, current_weight, global_obs

    def _temporal_encode(self, temporal):
        """
        temporal: (B, n_stocks, W, 6)
        returns:  (B, n_stocks, embed_dim)
        """
        B, N, W, F = temporal.shape
        x = temporal.reshape(B * N, W, F)               # (B*N, W, 6)

        h = self.input_proj(x) + self.pos_encoding[:, :W, :]  # (B*N, W, embed)

        attn_out, _ = self.temporal_attn(h, h, h)
        h = self.attn_norm(h + attn_out)
        h = self.ffn_norm(h + self.ffn(h))

        embeddings = h[:, -1, :]                         # (B*N, embed)
        return embeddings.reshape(B, N, -1)              # (B, n_stocks, embed)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        temporal, current_weight, _ = self._parse_obs(features)
        embeddings = self._temporal_encode(temporal)       # (B, n_stocks, embed)
        score_in = torch.cat([embeddings, current_weight], dim=-1)
        scores = self.score_head(score_in).squeeze(-1)     # (B, n_stocks)
        return scores

    def forward_critic(self, features):
        temporal, _, global_obs = self._parse_obs(features)
        embeddings = self._temporal_encode(temporal)        # (B, n_stocks, embed)
        pooled = embeddings.mean(dim=1)                     # (B, embed)
        value_input = torch.cat([pooled, global_obs], dim=1)
        return self.value_mlp(value_input)


class TemporalScoringPolicy(ActorCriticPolicy):
    """v0.61 自定义策略: Temporal Attention + Identity action_net"""

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
# 第五部分: 回测 (v0.61 增强 — 详细交易明细)
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
    """
    运行详细回测，维护影子账户计算已实现/浮动盈亏，生成完整交易明细 DataFrame。

    只跟踪权重 >= MIN_WEIGHT_THRESHOLD 的有效持仓，避免 softmax 产生的
    上千个微小权重造成噪声。

    Returns:
        df_details: DataFrame, 每行对应一个交易周期
        dates: ndarray, 日期序列 (含初始日期)
        values: ndarray, 组合净值序列 (含初始值)
        turnovers: ndarray, 换手率序列
    """
    obs, _ = env.reset()
    done = False
    dataset = env.dataset
    initial_cash = env.initial_cash

    stock_codes = np.array(dataset.stock_codes)
    code_to_name = {c: n for c, n in zip(dataset.stock_codes, dataset.stock_names)}
    code_to_idx = {c: i for i, c in enumerate(dataset.stock_codes)}

    # 影子账户: 只跟踪有效持仓 {stock_code: {'shares': float, 'avg_cost': float}}
    holdings = {}
    realized_pnl_cum = 0.0

    # 上期状态 (仅有效持仓)
    prev_weights_map = {}   # code -> weight
    prev_shares_map = {}    # code -> shares

    records = []
    all_dates = [dataset.dates[env.current_step]]
    all_values = [env.portfolio_value]
    all_turnovers = []

    while not done:
        if normalize_env:
            obs_norm = normalize_env.normalize_obs(obs)
            action, _ = model.predict(obs_norm, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)

        next_obs, reward, terminated, truncated, info = env.step(action)

        t = min(env.current_step, env.n_dates - 1)
        current_prices = dataset.actual_close[t]
        valid_price_mask = ~np.isnan(current_prices) & (current_prices > 0)

        current_total_asset = info["portfolio_value"]
        current_weights = env.weights.copy()
        target_values = current_total_asset * current_weights

        step_realized_pnl = 0.0
        current_weights_map = {}   # 本期有效持仓 code -> weight
        current_shares_map = {}    # 本期有效持仓 code -> shares

        for i in range(len(stock_codes)):
            if not valid_price_mask[i]:
                continue

            code = stock_codes[i]
            price = current_prices[i]
            w = current_weights[i]
            target_val = target_values[i]
            target_share = target_val / price if price > 0 else 0.0

            is_significant = (w >= MIN_WEIGHT_THRESHOLD)

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

        # ── 构建持仓明细 (仅有效持仓且价格有效) ──
        unrealized_pnl = 0.0
        valid_holdings = []

        for code, pos in holdings.items():
            idx = code_to_idx.get(code)
            if idx is None:
                continue
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

        # ── 识别清仓股票: 上期有效持仓 → 本期不再是有效持仓 ──
        cleared_stocks = []
        for code, prev_w in prev_weights_map.items():
            if code not in current_weights_map:
                cleared_stocks.append({
                    'name': code_to_name.get(code, code),
                    'prev_weight': prev_w,
                })
        cleared_stocks.sort(key=lambda x: x['prev_weight'], reverse=True)

        # ── 写入 record ──
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
            "持仓股票数(有效：仓位>0.1%)": len(current_weights_map),
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
    parser = argparse.ArgumentParser(description="RL 交易模型训练 v0.61")
    parser.add_argument("--config", type=str,
                        default=str(project_root / "configs" / "rl_config_v0.6.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    run_ts = datetime.now().strftime("%Y%m%d%H%M%S")

    print("=" * 70)
    print(f"  RL 交易模型 v0.61 (Temporal Attention + 详细回测) — {run_ts}")
    print("=" * 70)
    print_gpu_info()

    # ─── 1. 加载数据 ───
    print("\n[Step 1] 加载合并数据...")
    data_cfg = config["data"]
    data_dir = MERGED_DATA_DIR or data_cfg["merged_data_dir"]
    full_dataset = load_merged_data(
        data_dir=data_dir,
        min_data_points=data_cfg.get("min_data_points", 100),
    )

    # ─── 2. 降采样 ───
    resample_cfg = data_cfg.get("force_resample", {})
    if resample_cfg.get("enabled", False):
        print("\n[Step 2] 降采样...")
        full_dataset = resample_dataset_v2(full_dataset, rule=resample_cfg.get("rule", "W-FRI"))
    else:
        print("\n[Step 2] 未启用降采样")

    # ─── 3. 拆分 (从数据 split 列自动检测) ───
    split_date = full_dataset.auto_split_date
    if not split_date:
        split_date = config.get("split", {}).get("rl_train_end_date")
        print(f"\n[Step 3] 拆分: 训练 <= {split_date} / 测试 > {split_date} (来自配置文件)")
    else:
        print(f"\n[Step 3] 拆分: 训练 <= {split_date} / 测试 > {split_date} (从数据自动检测)")
    if not split_date:
        print("错误: 无法从数据中自动检测分割日期，也未在配置中指定 rl_train_end_date")
        sys.exit(1)
    train_dataset, test_dataset = full_dataset.split_by_date(split_date)

    if train_dataset is None or train_dataset.n_dates < 20:
        print("错误: 训练集数据不足")
        sys.exit(1)

    train_ppy = estimate_periods_per_year(train_dataset.dates)
    test_ppy = estimate_periods_per_year(test_dataset.dates) if test_dataset else train_ppy
    n_stocks = train_dataset.n_stocks

    rl_cfg = config["rl"]
    W = rl_cfg.get("temporal_window", 10)

    print(f"  训练集: {train_dataset.n_dates} 期, 测试集: {test_dataset.n_dates if test_dataset else 0} 期")
    print(f"  股票数: {n_stocks}, 时间窗口: W={W}, 频率: {train_ppy:.0f}/year")

    # ─── 4. 输出目录 ───
    output_dir = project_root / "experiments" / f"rl_ppo_v0.61_{run_ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(exist_ok=True)

    with open(configs_dir / "rl_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    # ─── 5. 创建环境 ───
    print("\n[Step 4] 创建环境...")
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

    # ─── 6. 创建 PPO ───
    print("\n[Step 5] 创建 PPO (Temporal Attention Policy)...")
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

    # ─── 7. 训练 ───
    total_ts = rl_cfg["total_timesteps"]
    print(f"\n[Step 6] 开始训练 ({total_ts:,} timesteps)...")

    log_cb = TrainingLogCallback(total_timesteps=total_ts, n_steps=n_steps)
    hist_cb = TrainingHistoryCallback(save_path=str(results_dir / "training_metrics.xlsx"))
    callbacks = CallbackList([log_cb, hist_cb])

    model.learn(total_timesteps=total_ts, callback=callbacks, progress_bar=False)
    print("  训练完成!")

    # ─── 8. 保存 ───
    model.save(str(model_dir / "ppo_trading"))
    if isinstance(train_vec_env, VecNormalize):
        train_vec_env.save(str(model_dir / "vec_normalize.pkl"))
    print(f"  模型已保存: {model_dir}")

    # ─── 9. 回测 (v0.61: 详细交易明细) ───
    print("\n[Step 7] 回测 (生成详细交易明细)...")

    def process_backtest(dataset, ppy, prefix):
        if not dataset or dataset.n_dates < 3:
            print(f"  {prefix}: 数据不足 ({dataset.n_dates if dataset else 0} 期), 跳过")
            return None

        env = MultiStockTradingEnvV3(dataset=dataset, **env_kwargs)

        norm_env = None
        if isinstance(train_vec_env, VecNormalize):
            _dummy = DummyVecEnv([lambda: env])
            norm_env = VecNormalize.load(str(model_dir / "vec_normalize.pkl"), _dummy)
            norm_env.training = False
            norm_env.norm_reward = False

        df_details, dates, values, turnovers = run_detailed_backtest(env, model, norm_env)

        # 保存详细交易明细 Excel
        detail_path = results_dir / f"{prefix}_trade_details.xlsx"
        df_details.to_excel(str(detail_path), index=False)
        print(f"  {prefix} 交易明细已保存: {detail_path}")

        # 指标计算与绘图
        bh_values = compute_buy_and_hold_v2(env)
        min_len = min(len(values), len(bh_values))

        metrics = compute_metrics(values[:min_len], dates[:min_len], periods_per_year=ppy)
        bh_metrics = compute_metrics(bh_values[:min_len], dates[:min_len], periods_per_year=ppy)

        print_backtest_report(metrics, bh_metrics,
                              label=f"RL Agent ({prefix})", benchmark_label="等权买入持有")
        plot_backtest(
            dates[:min_len], values[:min_len], bh_values[:min_len],
            save_path=str(results_dir / f"{prefix}_backtest.png"),
            title=f"{prefix.capitalize()} Backtest (v0.61 Temporal Attention)",
        )

        return metrics

    process_backtest(train_dataset, train_ppy, "train")
    process_backtest(test_dataset, test_ppy, "test") if test_dataset else None

    print(f"\n{'=' * 70}")
    print(f"  完成! 结果: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
