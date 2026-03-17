#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL 交易模型训练与回测主脚本
版本: v0.5
日期: 20260310

v0.5 变更:
  - 使用合并后的新数据源 (真实收盘价 + 120日远期预测)
  - Stock Scoring Network: 共享参数逐股票打分 + softmax 权重分配
  - 参数量从 ~8M 降至 ~8K
  - 改用 log_return 奖励函数

使用方式:
    python scripts/train_rl_v0.5_20260310.py --config configs/rl_config_v0.5.yaml
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

# ─── 常量 ───────────────────────────────────────────────────────────────────

N_FEATURES_PER_STOCK = 7
N_GLOBAL_FEATURES = 2
LOOKBACK_STEPS = 4


# ═══════════════════════════════════════════════════════════════════════════
# 第一部分: 数据加载与特征计算
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradingDatasetV2:
    """v0.5 交易数据集：真实收盘价 + 预测信号 + 预计算特征"""
    dates: np.ndarray
    stock_ids: list
    stock_codes: list
    stock_names: list
    actual_close: np.ndarray         # (n_dates, n_stocks)
    predicted_return_120d: np.ndarray # (n_dates, n_stocks)
    valid_mask: np.ndarray           # (n_dates, n_stocks) bool
    return_1w: np.ndarray            # (n_dates, n_stocks)
    return_1m: np.ndarray
    return_3m: np.ndarray
    volatility_1m: np.ndarray
    price_vs_ma60: np.ndarray
    step_returns: np.ndarray         # (n_dates, n_stocks) 逐步收益率

    @property
    def n_dates(self):
        return len(self.dates)

    @property
    def n_stocks(self):
        return len(self.stock_ids)

    def split_by_date(self, split_date: str):
        train_mask = self.dates <= split_date
        test_mask = self.dates > split_date
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        def _slice(idx):
            if len(idx) == 0:
                return None
            return TradingDatasetV2(
                dates=self.dates[idx],
                stock_ids=self.stock_ids,
                stock_codes=self.stock_codes,
                stock_names=self.stock_names,
                actual_close=self.actual_close[idx],
                predicted_return_120d=self.predicted_return_120d[idx],
                valid_mask=self.valid_mask[idx],
                return_1w=self.return_1w[idx],
                return_1m=self.return_1m[idx],
                return_3m=self.return_3m[idx],
                volatility_1m=self.volatility_1m[idx],
                price_vs_ma60=self.price_vs_ma60[idx],
                step_returns=self.step_returns[idx],
            )

        return _slice(train_idx), _slice(test_idx)


def _compute_period_return(close, period, valid_mask):
    """向量化计算 N 日收益率"""
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
    """从合并后的 parquet 目录加载数据并计算特征"""
    data_path = Path(data_dir)
    stocks_dir = data_path / "stocks"
    meta_path = data_path / "stock_list.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"未找到 stock_list.csv: {meta_path}")

    meta_df = pd.read_csv(meta_path)
    meta_df = meta_df.sort_values("stock_id").reset_index(drop=True)

    # 过滤数据点不足的股票
    meta_df = meta_df[meta_df["n_raw_dates"] >= min_data_points].reset_index(drop=True)
    print(f"  通过过滤的股票: {len(meta_df)} (min_data_points={min_data_points})")

    # 加载所有股票数据，收集日期
    all_stock_data = {}
    all_dates_set = set()
    for _, row in meta_df.iterrows():
        sid = str(row["stock_id"])
        code = str(row["stock_code"])
        fpath = stocks_dir / f"{sid}_{code}.parquet"
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath, columns=["date", "actual_close", "predicted_return_120d"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        all_stock_data[sid] = df
        all_dates_set.update(df["date"].values)

    # 构建统一日期轴
    dates_dt = np.array(sorted(all_dates_set))
    dates_str = np.array([str(d)[:10] for d in pd.to_datetime(dates_dt)])
    n_dates = len(dates_str)
    n_stocks = len(all_stock_data)

    print(f"  日期范围: {dates_str[0]} ~ {dates_str[-1]}, 共 {n_dates} 天")
    print(f"  股票数量: {n_stocks}")

    # 对齐到统一日期轴
    actual_close = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
    predicted_return_120d = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)

    stock_ids = []
    stock_codes = []
    stock_names = []
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

    # valid_mask: 有真实收盘价的位置
    valid_mask = ~np.isnan(actual_close) & (actual_close > 0)

    # 前向填充 predicted_return_120d (按列)
    pred_df = pd.DataFrame(predicted_return_120d)
    predicted_return_120d = pred_df.ffill().values
    predicted_return_120d = np.nan_to_num(predicted_return_120d, 0.0)

    # ─── 计算特征 ───
    print("  计算特征...")
    return_1w = _compute_period_return(actual_close, 5, valid_mask)
    return_1m = _compute_period_return(actual_close, 20, valid_mask)
    return_3m = _compute_period_return(actual_close, 60, valid_mask)

    # 逐步收益率 (用于环境计算组合收益)
    step_returns = _compute_period_return(actual_close, 1, valid_mask)

    # 日收益率 (用于计算波动率)
    daily_ret = step_returns.copy()

    # 波动率 (20日滚动标准差)
    vol_df = pd.DataFrame(daily_ret)
    volatility_1m = vol_df.rolling(20, min_periods=10).std().values
    volatility_1m = np.nan_to_num(volatility_1m, 0.0)

    # 价格 vs 60日均线
    close_df = pd.DataFrame(actual_close)
    ma60 = close_df.rolling(60, min_periods=30).mean().values
    price_vs_ma60 = np.zeros_like(actual_close)
    valid_ma = (ma60 > 0) & valid_mask
    np.divide(actual_close - ma60, ma60, out=price_vs_ma60, where=valid_ma)
    price_vs_ma60[~valid_ma] = 0.0

    # 截断极端值
    return_1w = np.clip(np.nan_to_num(return_1w, 0.0), -1.0, 1.0)
    return_1m = np.clip(np.nan_to_num(return_1m, 0.0), -1.0, 1.0)
    return_3m = np.clip(np.nan_to_num(return_3m, 0.0), -2.0, 2.0)
    volatility_1m = np.clip(volatility_1m, 0, 1.0)
    price_vs_ma60 = np.clip(np.nan_to_num(price_vs_ma60, 0.0), -1.0, 1.0)
    predicted_return_120d = np.clip(predicted_return_120d, -2.0, 2.0)
    step_returns = np.nan_to_num(step_returns, 0.0)

    dataset = TradingDatasetV2(
        dates=dates_str,
        stock_ids=stock_ids,
        stock_codes=stock_codes,
        stock_names=stock_names,
        actual_close=actual_close,
        predicted_return_120d=predicted_return_120d,
        valid_mask=valid_mask,
        return_1w=return_1w,
        return_1m=return_1m,
        return_3m=return_3m,
        volatility_1m=volatility_1m,
        price_vs_ma60=price_vs_ma60,
        step_returns=step_returns,
    )

    coverage = valid_mask.sum() / valid_mask.size * 100
    print(f"  数据覆盖率: {coverage:.1f}%")
    return dataset


def resample_dataset_v2(dataset: TradingDatasetV2, rule="W-FRI"):
    """将数据集降采样到指定频率"""
    raw_dates = pd.to_datetime(dataset.dates)
    s = pd.Series(np.arange(len(raw_dates)), index=raw_dates)
    resampled = s.resample(rule).last()
    idx = resampled.dropna().astype(int).values

    if len(idx) == 0:
        raise ValueError("降采样后没有剩余数据")

    print(f"  降采样: {len(dataset.dates)} → {len(idx)} ({len(idx)/len(dataset.dates):.1%})")
    return TradingDatasetV2(
        dates=dataset.dates[idx],
        stock_ids=dataset.stock_ids,
        stock_codes=dataset.stock_codes,
        stock_names=dataset.stock_names,
        actual_close=dataset.actual_close[idx],
        predicted_return_120d=dataset.predicted_return_120d[idx],
        valid_mask=dataset.valid_mask[idx],
        return_1w=dataset.return_1w[idx],
        return_1m=dataset.return_1m[idx],
        return_3m=dataset.return_3m[idx],
        volatility_1m=dataset.volatility_1m[idx],
        price_vs_ma60=dataset.price_vs_ma60[idx],
        step_returns=_compute_period_return(
            dataset.actual_close[idx], 1, dataset.valid_mask[idx]
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 第二部分: 交易环境
# ═══════════════════════════════════════════════════════════════════════════

def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class MultiStockTradingEnvV2(gym.Env):
    """
    v0.5 多股票组合交易环境

    观测: (n_stocks × 7 + 2) 展平向量
      per-stock (7): predicted_return_120d, return_1w, return_1m, return_3m,
                     volatility_1m, price_vs_ma60, current_weight
      global   (2): cash_ratio, portfolio_return

    动作: (n_stocks,) 原始分数 → softmax → 组合权重
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset: TradingDatasetV2, initial_cash=10_000_000.0,
                 transaction_cost=0.001, max_weight_per_stock=0.10,
                 softmax_temperature=1.0, reward_type="log_return",
                 turnover_penalty=0.5):
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

        obs_dim = self.n_stocks * N_FEATURES_PER_STOCK + N_GLOBAL_FEATURES
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.n_stocks,), dtype=np.float32
        )

        self.weights = np.zeros(self.n_stocks, dtype=np.float64)
        self.cash_ratio = 1.0
        self.portfolio_value = initial_cash
        self.current_step = LOOKBACK_STEPS

    def _get_obs(self):
        t = self.current_step
        ds = self.dataset
        stock_features = np.column_stack([
            ds.predicted_return_120d[t],
            ds.return_1w[t],
            ds.return_1m[t],
            ds.return_3m[t],
            ds.volatility_1m[t],
            ds.price_vs_ma60[t],
            self.weights,
        ])  # (n_stocks, 7)

        global_features = np.array([
            self.cash_ratio,
            (self.portfolio_value / self.initial_cash) - 1.0,
        ], dtype=np.float64)

        obs = np.concatenate([stock_features.flatten(), global_features])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        obs = np.clip(obs, -10.0, 10.0)
        return obs.astype(np.float32)

    def _action_to_weights(self, action):
        """将原始分数通过 softmax 转为组合权重，无效股票被屏蔽"""
        scores = action.astype(np.float64).copy()
        valid = self.dataset.valid_mask[self.current_step]
        scores[~valid] = -1e9

        if valid.sum() == 0:
            return np.zeros(self.n_stocks, dtype=np.float64)

        weights = _softmax(scores / self.temperature)

        # 限制单只股票最大权重 (迭代裁剪)
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
        self.current_step = LOOKBACK_STEPS
        self.weights = np.zeros(self.n_stocks, dtype=np.float64)
        self.cash_ratio = 1.0
        self.portfolio_value = self.initial_cash
        return self._get_obs(), {}

    def step(self, action):
        new_weights = self._action_to_weights(action)
        new_cash_ratio = 1.0 - new_weights.sum()

        # 换手率 & 交易成本
        turnover = np.abs(new_weights - self.weights).sum()
        turnover += abs(new_cash_ratio - self.cash_ratio)
        trade_cost = turnover * self.transaction_cost

        self.weights = new_weights
        self.cash_ratio = new_cash_ratio

        self.current_step += 1
        terminated = self.current_step >= self.n_dates - 1
        truncated = False

        # 组合收益
        stock_returns = self.dataset.step_returns[self.current_step]
        portfolio_return = np.dot(self.weights, stock_returns) - trade_cost
        self.portfolio_value *= (1.0 + portfolio_return)

        # 奖励
        if self.reward_type == "log_return":
            reward = np.log1p(np.clip(portfolio_return, -0.99, None))
        else:
            reward = portfolio_return
        reward -= self.turnover_penalty * trade_cost

        # 权重漂移 (市场运动导致的自然权重变化)
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
            "date": self.dataset.dates[self.current_step],
            "cash_ratio": self.cash_ratio,
            "n_active_stocks": int((self.weights > 1e-6).sum()),
        }

        return self._get_obs(), float(reward), terminated, truncated, info


# ═══════════════════════════════════════════════════════════════════════════
# 第三部分: 自定义 SB3 策略网络 (Stock Scoring Architecture)
# ═══════════════════════════════════════════════════════════════════════════

class StockScoringMlpExtractor(nn.Module):
    """
    替换 SB3 的 MlpExtractor，实现共享打分网络。

    Policy 路径: obs → 逐股票共享 MLP → per-stock scores (n_stocks,)
    Value 路径:  obs → 逐股票共享 MLP → mean pooling + global → MLP → value features
    """

    def __init__(self, feature_dim, n_stocks, n_features_per_stock, hidden_dim=64):
        super().__init__()
        self.n_stocks = n_stocks
        self.n_features = n_features_per_stock
        self.hidden_dim = hidden_dim
        embed_dim = hidden_dim // 2

        # 共享打分网络 (所有股票用相同权重)
        self.shared_net = nn.Sequential(
            nn.Linear(n_features_per_stock, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )
        self.score_head = nn.Linear(embed_dim, 1)

        # Value 路径: 聚合后的 MLP
        self.value_mlp = nn.Sequential(
            nn.Linear(embed_dim + N_GLOBAL_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

        # SB3 需要的属性
        self.latent_dim_pi = n_stocks
        self.latent_dim_vf = embed_dim

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def _parse_obs(self, features):
        batch = features.shape[0]
        n = self.n_stocks * self.n_features
        stock_obs = features[:, :n].reshape(batch, self.n_stocks, self.n_features)
        global_obs = features[:, n:]
        return stock_obs, global_obs

    def forward_actor(self, features):
        stock_obs, _ = self._parse_obs(features)
        embeddings = self.shared_net(stock_obs)         # (B, n_stocks, embed)
        scores = self.score_head(embeddings).squeeze(-1) # (B, n_stocks)
        return scores

    def forward_critic(self, features):
        stock_obs, global_obs = self._parse_obs(features)
        embeddings = self.shared_net(stock_obs)          # (B, n_stocks, embed)
        pooled = embeddings.mean(dim=1)                  # (B, embed)
        value_input = torch.cat([pooled, global_obs], dim=1)
        return self.value_mlp(value_input)               # (B, embed)


class StockScoringPolicy(ActorCriticPolicy):
    """
    自定义 PPO 策略: 共享打分网络 + Identity action_net

    参数量: ~8K (vs 标准 MLP 的 ~8M)
    """

    def __init__(self, observation_space, action_space, lr_schedule,
                 n_stocks=100, n_features_per_stock=7, hidden_dim=64, **kwargs):
        self._n_stocks = n_stocks
        self._n_features = n_features_per_stock
        self._hidden_dim = hidden_dim
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = StockScoringMlpExtractor(
            self.features_dim, self._n_stocks, self._n_features, self._hidden_dim
        )

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # 用 Identity 替换 Linear(n_stocks, n_stocks) 的 action_net
        # 因为 mlp_extractor 已经输出了 per-stock scores
        self.action_net = nn.Identity()
        # 重建 optimizer (排除被替换掉的旧 action_net 参数)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )


# ═══════════════════════════════════════════════════════════════════════════
# 第四部分: 训练回调
# ═══════════════════════════════════════════════════════════════════════════

class TrainingLogCallback(BaseCallback):
    """训练进度表格回调"""

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

        pbar_w = 30
        filled = int(pbar_w * pct / 100)
        bar = "█" * filled + "░" * (pbar_w - filled)

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
    """记录训练指标并保存为 Excel"""

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
    """等权买入持有基准"""
    ds = env.dataset
    start = LOOKBACK_STEPS
    values = [env.initial_cash]
    for t in range(start + 1, ds.n_dates):
        valid = ds.valid_mask[t] & ds.valid_mask[t - 1]
        n_valid = valid.sum()
        if n_valid > 0:
            ret = np.sum(ds.step_returns[t, valid]) / n_valid
        else:
            ret = 0.0
        values.append(values[-1] * (1.0 + ret))
    return np.array(values)


def run_backtest(env, model, normalize_env=None):
    """运行回测，返回日期和组合净值序列"""
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
    parser = argparse.ArgumentParser(description="RL 交易模型训练 v0.5")
    parser.add_argument("--config", type=str,
                        default=str(project_root / "configs" / "rl_config_v0.5.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    run_ts = datetime.now().strftime("%Y%m%d%H%M%S")

    print("=" * 70)
    print(f"  RL 交易模型 v0.5 (Stock Scoring) — {run_ts}")
    print("=" * 70)
    print_gpu_info()

    # ─── 1. 加载数据 ───
    print("\n[Step 1] 加载合并数据...")
    data_cfg = config["data"]
    full_dataset = load_merged_data(
        data_dir=data_cfg["merged_data_dir"],
        min_data_points=data_cfg.get("min_data_points", 100),
    )

    # ─── 2. 降采样 ───
    resample_cfg = data_cfg.get("force_resample", {})
    if resample_cfg.get("enabled", False):
        print("\n[Step 2] 降采样...")
        full_dataset = resample_dataset_v2(full_dataset, rule=resample_cfg.get("rule", "W-FRI"))
    else:
        print("\n[Step 2] 未启用降采样")

    # ─── 3. 拆分训练/测试 ───
    split_date = config["split"]["rl_train_end_date"]
    print(f"\n[Step 3] 拆分: 训练 <= {split_date} / 测试 > {split_date}")
    train_dataset, test_dataset = full_dataset.split_by_date(split_date)

    if train_dataset is None or train_dataset.n_dates < 20:
        print("错误: 训练集数据不足")
        sys.exit(1)

    train_ppy = estimate_periods_per_year(train_dataset.dates)
    test_ppy = estimate_periods_per_year(test_dataset.dates) if test_dataset else train_ppy
    n_stocks = train_dataset.n_stocks

    print(f"  训练集: {train_dataset.n_dates} 期, 测试集: {test_dataset.n_dates if test_dataset else 0} 期")
    print(f"  股票数: {n_stocks}, 频率: {train_ppy:.0f} periods/year")

    # ─── 4. 创建输出目录 ───
    output_dir = project_root / "experiments" / f"rl_ppo_v0.5_{run_ts}"
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
    rl_cfg = config["rl"]
    trading_cfg = config["trading"]
    reward_cfg = config["reward"]

    device = rl_cfg.get("device", "auto")
    if device == "cuda" and not torch.cuda.is_available():
        print("  警告: cuda 不可用，回退到 cpu")
        device = "cpu"

    def make_env(dataset, seed=0):
        def _init():
            env = MultiStockTradingEnvV2(
                dataset=dataset,
                initial_cash=trading_cfg["initial_cash"],
                transaction_cost=trading_cfg["transaction_cost"],
                max_weight_per_stock=trading_cfg["max_weight_per_stock"],
                softmax_temperature=trading_cfg["softmax_temperature"],
                reward_type=reward_cfg["type"],
                turnover_penalty=reward_cfg["turnover_penalty"],
            )
            env.reset(seed=seed)
            return Monitor(env)
        return _init

    train_vec_env = DummyVecEnv([make_env(train_dataset, seed=42)])

    if rl_cfg.get("normalize_observations", True):
        train_vec_env = VecNormalize(
            train_vec_env,
            norm_obs=True,
            norm_reward=rl_cfg.get("normalize_rewards", False),
            clip_obs=10.0,
            clip_reward=10.0,
        )

    # ─── 6. 创建 PPO ───
    print(f"\n[Step 5] 创建 PPO (Stock Scoring Policy)...")
    episode_len = train_dataset.n_dates - LOOKBACK_STEPS - 1
    n_steps = min(rl_cfg["n_steps"], episode_len)
    batch_size = min(rl_cfg["batch_size"], n_steps)

    policy_kwargs = dict(
        n_stocks=n_stocks,
        n_features_per_stock=N_FEATURES_PER_STOCK,
        hidden_dim=rl_cfg.get("hidden_dim", 64),
    )

    model = PPO(
        policy=StockScoringPolicy,
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
        verbose=0,
        seed=42,
        device=device,
    )

    # 打印参数统计
    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    scoring_params = sum(p.numel() for p in model.policy.mlp_extractor.shared_net.parameters())
    scoring_params += sum(p.numel() for p in model.policy.mlp_extractor.score_head.parameters())
    print(f"  总可训练参数: {total_params:,}")
    print(f"  其中打分网络: {scoring_params:,}")
    print(f"  Batch: {batch_size}, N Steps: {n_steps}, Episode Length: {episode_len}")

    # ─── 7. 训练 ───
    total_ts = rl_cfg["total_timesteps"]
    print(f"\n[Step 6] 开始训练 ({total_ts:,} timesteps)...")

    log_cb = TrainingLogCallback(total_timesteps=total_ts, n_steps=n_steps)
    hist_cb = TrainingHistoryCallback(save_path=str(results_dir / "training_metrics.xlsx"))
    callbacks = CallbackList([log_cb, hist_cb])

    model.learn(total_timesteps=total_ts, callback=callbacks, progress_bar=False)
    print("  训练完成!")

    # ─── 8. 保存模型 ───
    model.save(str(model_dir / "ppo_trading"))
    if isinstance(train_vec_env, VecNormalize):
        train_vec_env.save(str(model_dir / "vec_normalize.pkl"))
    print(f"  模型已保存: {model_dir}")

    # ─── 9. 回测 ───
    print("\n[Step 7] 回测...")

    def process_backtest(dataset, ppy, prefix):
        if not dataset or dataset.n_dates < 10:
            return None

        env = MultiStockTradingEnvV2(
            dataset=dataset,
            initial_cash=trading_cfg["initial_cash"],
            transaction_cost=trading_cfg["transaction_cost"],
            max_weight_per_stock=trading_cfg["max_weight_per_stock"],
            softmax_temperature=trading_cfg["softmax_temperature"],
            reward_type=reward_cfg["type"],
            turnover_penalty=reward_cfg["turnover_penalty"],
        )

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
            title=f"{prefix.capitalize()} Backtest (v0.5 Stock Scoring)",
        )

        # 保存回测详情
        detail_df = pd.DataFrame({
            "date": dates[:min_len],
            "portfolio_value": values[:min_len],
            "benchmark_value": bh_values[:min_len],
            "turnover": np.concatenate([[0], turnovers[:min_len - 1]]),
        })
        detail_df.to_excel(str(results_dir / f"{prefix}_details.xlsx"), index=False)

        return metrics

    train_metrics = process_backtest(train_dataset, train_ppy, "train")
    test_metrics = process_backtest(test_dataset, test_ppy, "test") if test_dataset else None

    print(f"\n{'=' * 70}")
    print(f"  完成! 结果: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
