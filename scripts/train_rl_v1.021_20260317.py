#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL 交易模型训练 v1.021 — GPU 优化 + 显存 OOM 修复
版本: v1.021
日期: 20260317

v1.021 变更 (基于 v1.02):
  - rollout_buffer.reset() 前显式释放旧 GPU 张量并调用 empty_cache，缓解 OOM

使用方式:
    python scripts/train_rl_v1.021_20260317.py --config configs/rl_config_v1.021.yaml
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

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.buffers import RolloutBuffer  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CallbackList  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.policies import ActorCriticPolicy  # noqa: E402
from stable_baselines3.common.utils import obs_as_tensor  # noqa: E402
from stable_baselines3.common.type_aliases import RolloutBufferSamples  # noqa: E402

from src.utils.backtest import (  # noqa: E402
    compute_metrics,
    estimate_periods_per_year,
    print_backtest_report,
    plot_backtest,
)

DATASET_DIR = "/data/projectRL_20260218/data/processed_data/rl_dataset_v0.3_202603111056"

N_MARKET_FEATURES = 9
PREDICTED_CLOSE_SCALE = 1000.0
N_GLOBAL_FEATURES = 2
MAX_HOLDINGS_IN_EXCEL = 30
MAX_CLEARED_IN_EXCEL = 20
TOP_K_STOCKS = 100
MIN_EPISODE_WEEKS = 156


def _slice_dataset(ds, indices):
    """从 TradingDatasetV3 按索引切片"""
    return type(ds)(
        dates=ds.dates[indices],
        stock_ids=ds.stock_ids,
        stock_codes=ds.stock_codes,
        stock_names=ds.stock_names,
        actual_close=ds.actual_close[indices],
        predicted_return_120d=ds.predicted_return_120d[indices],
        valid_mask=ds.valid_mask[indices],
        return_1w=ds.return_1w[indices],
        return_1m=ds.return_1m[indices],
        return_3m=ds.return_3m[indices],
        volatility_1m=ds.volatility_1m[indices],
        price_vs_ma60=ds.price_vs_ma60[indices],
        step_returns=ds.step_returns[indices],
        error_mean=ds.error_mean[indices],
        error_std=ds.error_std[indices],
        predicted_close=ds.predicted_close[indices],
    )


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
    error_mean: np.ndarray
    error_std: np.ndarray
    predicted_close: np.ndarray

    @property
    def n_dates(self):
        return len(self.dates)

    @property
    def n_stocks(self):
        return len(self.stock_ids)


def load_precomputed_dataset(npz_path: str, metadata: dict) -> TradingDatasetV3:
    data = np.load(npz_path)
    if "error_mean" in data:
        error_mean = data["error_mean"]
        error_std = data["error_std"]
    else:
        shape = data["actual_close"].shape
        error_mean = np.zeros(shape, dtype=np.float32)
        error_std = np.zeros(shape, dtype=np.float32)
    if "predicted_close" in data:
        predicted_close = data["predicted_close"]
    else:
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


def merge_datasets(train_ds: TradingDatasetV3, test_ds: TradingDatasetV3) -> TradingDatasetV3:
    """合并 train 和 test，按日期排序"""
    if test_ds is None:
        return train_ds
    dates = np.concatenate([train_ds.dates, test_ds.dates])
    date_objs = pd.to_datetime(dates)
    order = np.argsort(date_objs)
    dates = dates[order]
    return TradingDatasetV3(
        dates=dates,
        stock_ids=train_ds.stock_ids,
        stock_codes=train_ds.stock_codes,
        stock_names=train_ds.stock_names,
        actual_close=np.concatenate([train_ds.actual_close, test_ds.actual_close], axis=0)[order],
        predicted_return_120d=np.concatenate([train_ds.predicted_return_120d, test_ds.predicted_return_120d], axis=0)[order],
        valid_mask=np.concatenate([train_ds.valid_mask, test_ds.valid_mask], axis=0)[order],
        return_1w=np.concatenate([train_ds.return_1w, test_ds.return_1w], axis=0)[order],
        return_1m=np.concatenate([train_ds.return_1m, test_ds.return_1m], axis=0)[order],
        return_3m=np.concatenate([train_ds.return_3m, test_ds.return_3m], axis=0)[order],
        volatility_1m=np.concatenate([train_ds.volatility_1m, test_ds.volatility_1m], axis=0)[order],
        price_vs_ma60=np.concatenate([train_ds.price_vs_ma60, test_ds.price_vs_ma60], axis=0)[order],
        step_returns=np.concatenate([train_ds.step_returns, test_ds.step_returns], axis=0)[order],
        error_mean=np.concatenate([train_ds.error_mean, test_ds.error_mean], axis=0)[order],
        error_std=np.concatenate([train_ds.error_std, test_ds.error_std], axis=0)[order],
        predicted_close=np.concatenate([train_ds.predicted_close, test_ds.predicted_close], axis=0)[order],
    )


def _softmax_with_anchor(scores, anchor_score=0.0):
    max_score = max(np.max(scores), anchor_score)
    e_scores = np.exp(scores - max_score)
    e_anchor = np.exp(anchor_score - max_score)
    total = np.sum(e_scores) + e_anchor
    weights = e_scores / total
    cash_weight = e_anchor / total
    return weights, cash_weight


class MultiStockTradingEnvV10(gym.Env):
    """v1.0: 随机起始、随机长度、仅 episode 末奖励(年化收益)"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset: TradingDatasetV3, initial_cash=10_000_000.0,
                 transaction_cost=0.001, max_weight_per_stock=0.10,
                 softmax_temperature=1.0, turnover_penalty=0.5, temporal_window=10,
                 trade_interval=1, use_random_start=True, max_start_idx=None,
                 min_episode_weeks=156, periods_per_year=52):
        super().__init__()
        self.dataset = dataset
        self.n_stocks = dataset.n_stocks
        self.n_dates = dataset.n_dates
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_weight_per_stock = max_weight_per_stock
        self.temperature = softmax_temperature
        self.turnover_penalty = turnover_penalty
        self.W = temporal_window
        self.lookback = max(temporal_window, 4)
        self.trade_interval = max(1, int(trade_interval))
        self.use_random_start = use_random_start
        self.max_start_idx = max_start_idx
        self.min_episode_weeks = min_episode_weeks
        self.periods_per_year = periods_per_year

        predicted_close_norm = np.nan_to_num(dataset.predicted_close, 0.0) / PREDICTED_CLOSE_SCALE
        self.market_features = np.stack([
            dataset.predicted_return_120d,
            dataset.return_1w,
            dataset.return_1m,
            dataset.return_3m,
            dataset.volatility_1m,
            dataset.price_vs_ma60,
            dataset.error_mean,
            dataset.error_std,
            predicted_close_norm,
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
        self.episode_start_step = self.lookback
        self.episode_end_step = self.n_dates - 1
        self._rng = np.random.default_rng()

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
        stock_weights, _ = _softmax_with_anchor(scores / self.temperature, anchor_score=0.0)
        over = stock_weights > self.max_weight_per_stock
        if over.any():
            stock_weights[over] = self.max_weight_per_stock
        return stock_weights

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.use_random_start and self.max_start_idx is not None:
            max_start = min(self.max_start_idx, self.n_dates - 1 - self.min_episode_weeks)
            if max_start < self.lookback:
                start_step = self.lookback
            else:
                start_step = int(self._rng.integers(self.lookback, max_start + 1))
            max_len = self.n_dates - 1 - start_step
            if max_len >= self.min_episode_weeks:
                episode_len = int(self._rng.integers(self.min_episode_weeks, max_len + 1))
            else:
                episode_len = max_len
            episode_end = start_step + episode_len
        else:
            start_step = self.lookback
            episode_end = self.n_dates - 1

        self.current_step = start_step
        self.episode_start_step = start_step
        self.episode_end_step = episode_end
        self.weights = np.zeros(self.n_stocks, dtype=np.float64)
        self.cash_ratio = 1.0
        self.portfolio_value = self.initial_cash
        return self._get_obs(), {}

    def step(self, action):
        steps_since_start = self.current_step - self.episode_start_step
        is_trade_step = (steps_since_start % self.trade_interval) == 0

        if is_trade_step:
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
        else:
            turnover = 0.0
            trade_cost = 0.0

        self.current_step += 1
        terminated = self.current_step >= self.episode_end_step
        truncated = False
        t = min(self.current_step, self.n_dates - 1)
        stock_returns = self.dataset.step_returns[t]
        portfolio_return = np.dot(self.weights, stock_returns) - trade_cost
        self.portfolio_value *= (1.0 + portfolio_return)
        drifted = self.weights * (1.0 + stock_returns)
        total_after = drifted.sum() + self.cash_ratio
        if total_after > 1e-10:
            self.weights = drifted / total_after
            self.cash_ratio = self.cash_ratio / total_after
        else:
            self.weights = np.zeros(self.n_stocks)
            self.cash_ratio = 1.0

        if terminated:
            years = (self.episode_end_step - self.episode_start_step) / self.periods_per_year
            if years > 1e-6 and self.portfolio_value > 0:
                total_return = (self.portfolio_value - self.initial_cash) / self.initial_cash
                annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0
            else:
                annualized_return = 0.0
            reward = float(annualized_return)
        else:
            reward = 0.0

        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "turnover": turnover,
            "trade_cost": trade_cost,
            "date": self.dataset.dates[t],
            "cash_ratio": self.cash_ratio,
            "n_active_stocks": int((self.weights > 1e-6).sum()),
        }
        return self._get_obs(), reward, terminated, truncated, info


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
        self.pos_encoding = nn.Parameter(torch.randn(1, temporal_window, embed_dim) * 0.02)
        self.temporal_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.0)
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
        return self.score_head(score_in).squeeze(-1)

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


class CompleteEpisodeRolloutBuffer(RolloutBuffer):
    """
    容量为 n_steps*2 的 RolloutBuffer，支持可变长度（pos < buffer_size 时）。
    用于 PPOCompleteEpisodes：每次加入完整 episode，训练时只使用实际收集的样本。
    """

    def compute_returns_and_advantage(self, last_values, dones):
        """仅对实际收集的 pos 个 transition 计算 GAE"""
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0
        n_valid = self.pos
        for step in reversed(range(n_valid)):
            if step == n_valid - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def get(self, batch_size=None):
        """使用实际收集的 pos 个样本进行训练，而非 buffer_size"""
        assert self.full, "Buffer must be full before get()"
        n_valid = self.pos * self.n_envs
        indices = np.random.permutation(n_valid)
        if not self.generator_ready:
            for name in ["observations", "actions", "values", "log_probs", "advantages", "returns"]:
                self.__dict__[name] = self.swap_and_flatten(self.__dict__[name])
            self.generator_ready = True
        if batch_size is None:
            batch_size = n_valid
        start_idx = 0
        while start_idx < n_valid:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size


class GPUCompleteEpisodeRolloutBuffer(CompleteEpisodeRolloutBuffer):
    """
    v1.021: GPU 优化的 RolloutBuffer + 显存 OOM 修复
    - observations 存入 GPU
    - values、log_probs 保持为 tensor，不转 numpy
    - GAE 计算全程在 GPU
    - reset() 前显式释放旧张量并 empty_cache，避免下一轮 rollout 时 OOM
    """

    def reset(self):
        super().reset()
        self.generator_ready = False
        self._values_list = []
        self._log_probs_list = []
        # v1.021: 显式释放旧 GPU 张量，再分配新的，缓解显存碎片化导致的 OOM
        if hasattr(self, "_obs_tensor") and self._obs_tensor is not None:
            del self._obs_tensor
            self._obs_tensor = None
        if hasattr(self, "_obs_flat") and self._obs_flat is not None:
            del self._obs_flat
            self._obs_flat = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        obs_shape = (self.buffer_size, self.n_envs, *self.obs_shape)
        self._obs_tensor = torch.zeros(obs_shape, dtype=torch.float32, device=self.device)
        self._values_tensor = None
        self._log_probs_tensor = None
        self._advantages_tensor = None
        self._returns_tensor = None

    def add(self, obs, action, reward, episode_start, value, log_prob):
        """存储 obs 到 GPU，values/log_probs 保持为 tensor"""
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
        action = action.reshape((self.n_envs, self.action_dim))

        obs_t = torch.from_numpy(np.array(obs)).float().to(self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        self._obs_tensor[self.pos] = obs_t
        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        v_flat = value.flatten() if value.numel() > 1 else value.reshape(1)
        lp_flat = log_prob.flatten() if log_prob.numel() > 1 else log_prob.reshape(-1)
        self.values[self.pos] = value.detach().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.detach().cpu().numpy()
        self._values_list.append(value.detach())
        self._log_probs_list.append(log_prob.detach())
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values, dones):
        """GAE 全程在 GPU 上计算"""
        n_valid = self.pos
        last_vals = last_values.flatten().to(self.device)
        dones_t = torch.from_numpy(dones.astype(np.float32)).to(self.device)

        values_list = []
        for i in range(n_valid):
            v = self._values_list[i]
            v_flat = v.flatten() if v.numel() > 1 else v.reshape(1)
            values_list.append(v_flat[: self.n_envs])
        values_t = torch.stack(values_list).to(self.device)
        if values_t.dim() == 1:
            values_t = values_t.unsqueeze(1)
        rewards_t = torch.from_numpy(self.rewards[:n_valid]).to(self.device)
        ep_starts_t = torch.from_numpy(self.episode_starts[:n_valid]).to(self.device)

        last_gae_lam = torch.zeros(self.n_envs, device=self.device, dtype=torch.float32)
        advantages_t = torch.zeros((n_valid, self.n_envs), device=self.device, dtype=torch.float32)

        for step in reversed(range(n_valid)):
            if step == n_valid - 1:
                next_non_terminal = 1.0 - dones_t
                next_values = last_vals
            else:
                next_non_terminal = 1.0 - ep_starts_t[step + 1]
                next_values = values_t[step + 1].flatten()
            delta = rewards_t[step] + self.gamma * next_values * next_non_terminal - values_t[step].flatten()
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages_t[step] = last_gae_lam

        returns_t = advantages_t + values_t
        self.advantages = advantages_t.cpu().numpy()
        self.returns = returns_t.cpu().numpy()
        self._advantages_tensor = advantages_t
        self._returns_tensor = returns_t
        self._values_tensor = values_t
        lp_list = []
        for i in range(n_valid):
            lp = self._log_probs_list[i]
            lp_flat = lp.flatten() if lp.numel() > 1 else lp.reshape(1)
            lp_list.append(lp_flat[: self.n_envs])
        self._log_probs_tensor = torch.stack(lp_list).to(self.device)
        if self._log_probs_tensor.dim() == 1:
            self._log_probs_tensor = self._log_probs_tensor.unsqueeze(1)

    def get(self, batch_size=None):
        """使用 GPU 上的数据，避免每 batch CPU->GPU 传输"""
        assert self.full, "Buffer must be full before get()"
        n_valid = self.pos * self.n_envs
        indices = np.random.permutation(n_valid)
        if not self.generator_ready:
            self._obs_flat = self._obs_tensor[: self.pos].swapaxes(0, 1).reshape(n_valid, *self.obs_shape)
            self._values_flat = self._values_tensor.swapaxes(0, 1).reshape(-1)
            self._log_probs_flat = self._log_probs_tensor.swapaxes(0, 1).reshape(-1)
            self._advantages_flat = self._advantages_tensor.swapaxes(0, 1).reshape(-1)
            self._returns_flat = self._returns_tensor.swapaxes(0, 1).reshape(-1)
            self.generator_ready = True
        if batch_size is None:
            batch_size = n_valid
        start_idx = 0
        while start_idx < n_valid:
            yield self._get_samples_gpu(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples_gpu(self, batch_inds):
        """直接返回 GPU 上的 tensor，无需 to_torch 转换"""
        batch_inds_t = torch.from_numpy(batch_inds).long().to(self.device)
        obs_batch = self._obs_flat[batch_inds_t]
        actions_batch = torch.from_numpy(
            self.actions[: self.pos].swapaxes(0, 1).reshape(-1, self.action_dim)[batch_inds]
        ).float().to(self.device)
        return RolloutBufferSamples(
            obs_batch,
            actions_batch,
            self._values_flat[batch_inds_t],
            self._log_probs_flat[batch_inds_t],
            self._advantages_flat[batch_inds_t],
            self._returns_flat[batch_inds_t],
        )


class PPOCompleteEpisodes(PPO):
    """v1.01: 仅收集完整 episode 的 transition 用于更新，buffer 容量 2*n_steps"""

    def _setup_model(self):
        super()._setup_model()
        self.rollout_buffer = CompleteEpisodeRolloutBuffer(
            self.n_steps * 2,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.env.num_envs,
        )

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """仅将完整 episode 的 transition 加入 buffer，收集满 n_rollout_steps 后更新"""
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps_target = n_rollout_steps
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        episode_buffer = []
        obs = self._last_obs
        episode_starts = self._last_episode_starts

        while rollout_buffer.size() < n_steps_target:
            if self.use_sde and self.sde_sample_freq > 0 and len(episode_buffer) % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                obs_tensor = obs_as_tensor(obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            obs_np = np.array(obs) if not isinstance(obs, np.ndarray) else obs
            if obs_np.ndim == 1:
                obs_np = obs_np.reshape(1, -1)
            ep_start = np.array(episode_starts, dtype=np.float32)
            if ep_start.ndim == 0:
                ep_start = ep_start.reshape(1)
            vals_t = values.detach()
            if vals_t.dim() == 0:
                vals_t = vals_t.unsqueeze(0)
            log_probs_t = log_probs.detach()
            if log_probs_t.dim() == 0:
                log_probs_t = log_probs_t.unsqueeze(0)

            episode_buffer.append((obs_np.copy(), actions.copy(), rewards.copy(), ep_start.copy(), vals_t, log_probs_t))

            if np.any(dones):
                for o, a, r, es, v, lp in episode_buffer:
                    rollout_buffer.add(o, a, r, es, v, lp)
                episode_buffer = []

                if rollout_buffer.size() >= n_steps_target:
                    obs = env.reset()
                    episode_starts = np.ones(env.num_envs, dtype=bool)
                    break

                obs = env.reset()
                episode_starts = np.ones(env.num_envs, dtype=bool)
            else:
                obs = new_obs
                episode_starts = np.zeros(env.num_envs, dtype=bool)

        self._last_obs = obs
        self._last_episode_starts = episode_starts

        if rollout_buffer.size() < n_steps_target:
            callback.on_rollout_end()
            return True

        rollout_buffer.full = True

        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=np.zeros(env.num_envs, dtype=bool))

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True


class PPOCompleteEpisodesV102(PPOCompleteEpisodes):
    """v1.021: GPU 优化 + 显存 OOM 修复"""

    def _setup_model(self):
        super()._setup_model()
        self.rollout_buffer = GPUCompleteEpisodeRolloutBuffer(
            self.n_steps * 2,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.env.num_envs,
        )

    def train(self):
        """重写 train，增加 epoch 级别进度打印"""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        n_batches = (self.rollout_buffer.pos * self.rollout_buffer.n_envs + self.batch_size - 1) // self.batch_size

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            batch_idx = 0
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fractions.append(torch.mean((torch.abs(ratio - 1) > clip_range).float()).item())

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                batch_idx += 1

            if (epoch + 1) % max(1, self.n_epochs // 5) == 0 or epoch == 0:
                sys.stdout.write(f"\r  [train] epoch {epoch + 1}/{self.n_epochs}\n")
                sys.stdout.flush()

            self._n_updates += 1
            if not continue_training:
                break

        from stable_baselines3.common.utils import explained_variance
        n_valid = self.rollout_buffer.pos * self.rollout_buffer.n_envs
        values_flat = self.rollout_buffer.values[: self.rollout_buffer.pos].flatten()
        returns_flat = self.rollout_buffer.returns.flatten()
        if len(values_flat) > n_valid:
            values_flat = values_flat[:n_valid]
        if len(returns_flat) > n_valid:
            returns_flat = returns_flat[:n_valid]
        explained_var = explained_variance(values_flat, returns_flat)
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


def _evaluate_val_annualized_return(env, model, norm_env=None):
    obs, _ = env.reset()
    done = False
    while not done:
        if norm_env is not None:
            obs_flat = obs.reshape(1, -1) if obs.ndim == 1 else obs
            obs_norm = norm_env.normalize_obs(obs_flat)
            obs_norm = obs_norm.reshape(-1) if obs_norm.ndim == 2 else obs_norm
            action, _ = model.predict(obs_norm, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info["portfolio_value"], env.initial_cash, env.episode_end_step - env.episode_start_step


class EarlyStoppingCallbackV1(BaseCallback):
    def __init__(self, eval_freq, patience, val_dataset, env_kwargs, periods_per_year,
                 model_dir, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.patience = patience
        self.val_dataset = val_dataset
        self.env_kwargs = env_kwargs
        self.periods_per_year = periods_per_year
        self.model_dir = Path(model_dir)
        self.best_score = float("-inf")
        self.best_timesteps = 0
        self.no_improve_count = 0
        self._should_stop = False
        self._last_eval_at = 0

    def _get_norm_env(self):
        env = self.model.get_env()
        if hasattr(env, "normalize_obs"):
            return env
        return None

    def _on_rollout_end(self):
        if self._should_stop:
            return
        # 仅在至少完成一次参数更新后再验证，避免用初始模型做验证
        if getattr(self.model, "_n_updates", 0) == 0:
            return
        if self.num_timesteps < self._last_eval_at + self.eval_freq:
            return
        if self.val_dataset is None or self.val_dataset.n_dates < MIN_EPISODE_WEEKS:
            return
        self._last_eval_at = self.num_timesteps

        norm_env = self._get_norm_env()
        val_kwargs = {k: v for k, v in self.env_kwargs.items()}
        val_kwargs["use_random_start"] = False
        val_kwargs["max_start_idx"] = None
        val_env = MultiStockTradingEnvV10(dataset=self.val_dataset, **val_kwargs)

        final_val, initial_cash, n_steps = _evaluate_val_annualized_return(
            val_env, self.model, norm_env
        )
        years = n_steps / self.periods_per_year
        if years > 1e-6:
            total_ret = (final_val - initial_cash) / initial_cash
            annualized = (1.0 + total_ret) ** (1.0 / years) - 1.0
        else:
            annualized = 0.0

        if annualized > self.best_score:
            self.best_score = annualized
            self.best_timesteps = self.num_timesteps
            self.no_improve_count = 0
            self.model.save(str(self.model_dir / "ppo_trading_best"))
            if self.verbose:
                print(f"  [早停] step={self.num_timesteps} 验证年化={annualized:.4f} *best*")
        else:
            self.no_improve_count += 1
            if self.verbose:
                print(f"  [早停] step={self.num_timesteps} 验证年化={annualized:.4f} no_improve={self.no_improve_count}/{self.patience}")

        if self.no_improve_count >= self.patience:
            self._should_stop = True
            if self.verbose:
                print(f"  [早停] 连续 {self.patience} 次未提升，停止。最佳 step={self.best_timesteps} 年化={self.best_score:.4f}")

    def _on_step(self):
        return not self._should_stop


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
            f"  elapsed={self._fmt_time(elapsed)}  eta={eta}  ep_rew={ep_rew:.4f}  loss={loss:.4f}  ev={ev:+.3f}"
            if not np.isnan(loss) else f"  elapsed={self._fmt_time(elapsed)}  eta={eta}  ep_rew={ep_rew:.4f}",
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


def _add_bh_columns_to_trade_details(df_details, values, bh_values, initial_cash):
    n = len(df_details)
    if n == 0:
        return df_details
    values_arr = np.array(values)
    bh_arr = np.array(bh_values)
    min_len = min(len(values_arr), len(bh_arr), n + 1)
    if min_len < 2:
        return df_details
    vals_rl = values_arr[1 : n + 1]
    vals_bh = bh_arr[1 : n + 1]
    cum_max_rl = np.maximum.accumulate(values_arr[: n + 1])[1:]
    cum_max_bh = np.maximum.accumulate(bh_arr[: n + 1])[1:]
    eps = 1e-12
    drawdown_rl = (vals_rl - cum_max_rl) / (cum_max_rl + eps)
    drawdown_bh = (vals_bh - cum_max_bh) / (cum_max_bh + eps)
    df_details["RL_累计回撤"] = drawdown_rl[:n]
    df_details["等权BH_总资产"] = vals_bh[:n]
    df_details["等权BH_累计盈亏"] = vals_bh[:n] - initial_cash
    df_details["等权BH_累计回撤"] = drawdown_bh[:n]
    cols = list(df_details.columns)
    bh_cols = ["RL_累计回撤", "等权BH_总资产", "等权BH_累计盈亏", "等权BH_累计回撤"]
    for c in bh_cols:
        if c in cols:
            cols.remove(c)
    idx_insert = cols.index("清仓股票数") + 1 if "清仓股票数" in cols else len(cols)
    for i, c in enumerate(bh_cols):
        cols.insert(idx_insert + i, c)
    df_details = df_details[cols]
    return df_details


def _compute_extended_metrics(portfolio_values, dates, periods_per_year):
    base = compute_metrics(portfolio_values, dates, periods_per_year)
    if "error" in base:
        return base
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return base
    ar = base["annualized_return"]
    md = base["max_drawdown"]
    std_p = base["std_return_per_period"]
    annualized_vol = std_p * np.sqrt(periods_per_year) if std_p > 0 else 0.0
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-10
    sortino = ar / (downside_std * np.sqrt(periods_per_year)) if downside_std > 1e-12 else 0.0
    calmar = ar / abs(md) if md != 0 else 0.0
    base["sortino_ratio"] = sortino
    base["calmar_ratio"] = calmar
    base["annualized_volatility"] = annualized_vol
    return base


def _build_evaluation_excel(metrics_train, bh_train, metrics_val, bh_val, save_path):
    rows = [
        ("夏普比率", "sharpe_ratio"),
        ("索提诺比率", "sortino_ratio"),
        ("最大回撤", "max_drawdown"),
        ("卡玛比率", "calmar_ratio"),
        ("累计收益率", "total_return"),
        ("年化收益率", "annualized_return"),
        ("年化波动率", "annualized_volatility"),
        ("胜率", "win_rate"),
        ("盈亏比", "profit_loss_ratio"),
    ]
    cols = ["RL_训练集", "RL_验证集", "等权BH_训练集", "等权BH_验证集", "RL相对BH_训练集", "RL相对BH_验证集"]
    data = {c: [] for c in cols}
    for name, key in rows:
        rl_tr = metrics_train.get(key, np.nan) if metrics_train and "error" not in metrics_train else np.nan
        rl_va = metrics_val.get(key, np.nan) if metrics_val and "error" not in metrics_val else np.nan
        bh_tr = bh_train.get(key, np.nan) if bh_train and "error" not in bh_train else np.nan
        bh_va = bh_val.get(key, np.nan) if bh_val and "error" not in bh_val else np.nan
        diff_tr = rl_tr - bh_tr if not (np.isnan(rl_tr) or np.isnan(bh_tr)) else np.nan
        diff_va = rl_va - bh_va if not (np.isnan(rl_va) or np.isnan(bh_va)) else np.nan
        data["RL_训练集"].append(rl_tr)
        data["RL_验证集"].append(rl_va)
        data["等权BH_训练集"].append(bh_tr)
        data["等权BH_验证集"].append(bh_va)
        data["RL相对BH_训练集"].append(diff_tr)
        data["RL相对BH_验证集"].append(diff_va)
    df = pd.DataFrame(data, index=[r[0] for r in rows])
    df.index.name = "指标"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(str(save_path))
    print(f"  策略评价指标已保存: {save_path}")


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
    parser = argparse.ArgumentParser(description="RL 交易模型训练 v1.021 (显存OOM修复)")
    parser.add_argument("--config", type=str,
                        default=str(project_root / "configs" / "rl_config_v1.021.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    run_ts = datetime.now().strftime("%Y%m%d%H%M%S")

    print("=" * 70)
    print(f"  RL 交易模型 v1.021 (显存OOM修复) — {run_ts}")
    print("=" * 70)
    print_gpu_info()

    dataset_dir_str = DATASET_DIR
    if not dataset_dir_str or not Path(dataset_dir_str).exists():
        config_dir = config["data"].get("dataset_dir", "")
        if config_dir and Path(config_dir).exists():
            dataset_dir_str = config_dir
        else:
            raise FileNotFoundError(
                f"请先在训练脚本中设置 DATASET_DIR (当前: {dataset_dir_str})"
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

    train_raw = load_precomputed_dataset(str(train_npz), metadata)
    test_raw = load_precomputed_dataset(str(test_npz), metadata) if test_npz.exists() else None

    full_dataset = merge_datasets(train_raw, test_raw)
    val_years = config["data"].get("val_years", 5)
    periods_per_year = config["data"].get("periods_per_year", 52)
    val_weeks = val_years * periods_per_year

    if full_dataset.n_dates <= val_weeks:
        raise ValueError(f"全量数据仅 {full_dataset.n_dates} 期，不足验证集 {val_weeks} 期")

    val_indices = np.arange(full_dataset.n_dates - val_weeks, full_dataset.n_dates)
    val_dataset = _slice_dataset(full_dataset, val_indices)
    train_dataset = full_dataset

    W = config["rl"].get("temporal_window", 30)
    lookback = max(W, 4)

    val_start_date = str(val_dataset.dates[0])
    train_dates = train_dataset.dates
    idx_before_val = np.where(train_dates < val_start_date)[0]
    if len(idx_before_val) == 0:
        max_train_start_idx = lookback
    else:
        max_train_start_idx = int(idx_before_val[-1])
    max_train_start_idx = min(max_train_start_idx, train_dataset.n_dates - 1 - MIN_EPISODE_WEEKS)
    if max_train_start_idx < lookback:
        max_train_start_idx = lookback

    print(f"  股票数: {train_dataset.n_stocks}")
    print(f"  全量数据: {full_dataset.n_dates} 期")
    print(f"  验证集: {val_dataset.n_dates} 期 ({val_start_date} ~ {val_dataset.dates[-1]})")
    print(f"  训练随机起点上限: {max_train_start_idx} (对应日期 < {val_start_date})")
    print(f"  v1.021: GPU buffer + reset 前显存释放")

    early_stop_cfg = config.get("early_stopping", {})
    early_stop_enabled = early_stop_cfg.get("enabled", False) and val_dataset.n_dates >= MIN_EPISODE_WEEKS

    train_ppy = estimate_periods_per_year(train_dataset.dates)
    val_ppy = estimate_periods_per_year(val_dataset.dates)
    n_stocks = train_dataset.n_stocks
    rl_cfg = config["rl"]
    trade_interval = config["trading"].get("trade_interval", 1)
    min_episode_years = rl_cfg.get("min_episode_years", 3)
    min_episode_weeks = min_episode_years * periods_per_year

    print(f"  时间窗口: W={W}, 交易间隔: {trade_interval}, 最少episode: {min_episode_weeks} 周")
    if early_stop_enabled:
        print(f"  早停: eval_freq={early_stop_cfg.get('eval_freq', 2560)}, patience={early_stop_cfg.get('patience', 5)}")

    output_dir = project_root / "experiments" / f"rl_ppo_v1.021_{run_ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    with open(configs_dir / "rl_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    trading_cfg = config["trading"]
    device = rl_cfg.get("device", "auto")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    env_kwargs = dict(
        initial_cash=trading_cfg["initial_cash"],
        transaction_cost=trading_cfg["transaction_cost"],
        max_weight_per_stock=trading_cfg["max_weight_per_stock"],
        softmax_temperature=trading_cfg["softmax_temperature"],
        turnover_penalty=0.0,
        temporal_window=W,
        trade_interval=trade_interval,
        use_random_start=True,
        max_start_idx=max_train_start_idx,
        min_episode_weeks=min_episode_weeks,
        periods_per_year=periods_per_year,
    )

    train_seed = int(datetime.now().timestamp() * 1_000_000) % (2**31)
    print(f"  随机种子: {train_seed} (每次运行不同，可复现时请指定)")

    def make_env(seed=0):
        def _init():
            env = MultiStockTradingEnvV10(dataset=train_dataset, **env_kwargs)
            env.reset(seed=seed)
            return Monitor(env)
        return _init

    train_vec_env = DummyVecEnv([make_env(seed=train_seed)])
    if rl_cfg.get("normalize_observations", True):
        train_vec_env = VecNormalize(
            train_vec_env, norm_obs=True,
            norm_reward=False,
            clip_obs=10.0, clip_reward=10.0,
        )

    n_steps = rl_cfg["n_steps"]
    batch_size = rl_cfg["batch_size"]

    model = PPOCompleteEpisodesV102(
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
        policy_kwargs=dict(
            n_stocks=n_stocks,
            temporal_window=W,
            n_market_features=N_MARKET_FEATURES,
            hidden_dim=rl_cfg.get("hidden_dim", 64),
            n_heads=rl_cfg.get("n_heads", 2),
        ),
        verbose=0, seed=train_seed, device=device,
    )

    total_ts = rl_cfg["total_timesteps"]
    print(f"\n[Step 4] 开始训练 ({total_ts:,} timesteps)...")
    log_cb = TrainingLogCallback(total_timesteps=total_ts, n_steps=n_steps)
    hist_cb = TrainingHistoryCallback(save_path=str(results_dir / "training_metrics.xlsx"))

    callbacks = [log_cb, hist_cb]
    if early_stop_enabled:
        early_env_kwargs = {k: v for k, v in env_kwargs.items() if k not in ("use_random_start", "max_start_idx")}
        early_cb = EarlyStoppingCallbackV1(
            eval_freq=early_stop_cfg.get("eval_freq", 2560),
            patience=early_stop_cfg.get("patience", 5),
            val_dataset=val_dataset,
            env_kwargs=early_env_kwargs,
            periods_per_year=periods_per_year,
            model_dir=str(model_dir),
            verbose=1,
        )
        callbacks.append(early_cb)

    model.learn(total_timesteps=total_ts, callback=CallbackList(callbacks), progress_bar=False)
    print("  训练完成!")

    best_path = model_dir / "ppo_trading_best.zip"
    if best_path.exists():
        print("  恢复早停最佳模型...")
        model = PPOCompleteEpisodesV102.load(str(model_dir / "ppo_trading_best"), env=train_vec_env)

    model.save(str(model_dir / "ppo_trading"))
    if isinstance(train_vec_env, VecNormalize):
        train_vec_env.save(str(model_dir / "vec_normalize.pkl"))
    print(f"  模型已保存: {model_dir}")

    print("\n[Step 5] 回测...")

    def process_backtest(dataset, ppy, prefix, use_random_start=False):
        if not dataset or dataset.n_dates < MIN_EPISODE_WEEKS:
            print(f"  {prefix}: 数据不足, 跳过")
            return None, None
        bk_env_kwargs = {k: v for k, v in env_kwargs.items() if k not in ("use_random_start", "max_start_idx")}
        bk_env_kwargs["use_random_start"] = use_random_start
        bk_env_kwargs["max_start_idx"] = None
        env = MultiStockTradingEnvV10(dataset=dataset, **bk_env_kwargs)
        norm_env = None
        if isinstance(train_vec_env, VecNormalize):
            _dummy = DummyVecEnv([lambda: env])
            norm_env = VecNormalize.load(str(model_dir / "vec_normalize.pkl"), _dummy)
            norm_env.training = False
            norm_env.norm_reward = False

        df_details, dates, values, turnovers = run_detailed_backtest(env, model, norm_env)
        bh_values = compute_buy_and_hold_v2(env)
        initial_cash = env.initial_cash

        df_details = _add_bh_columns_to_trade_details(df_details, values, bh_values, initial_cash)

        detail_path = results_dir / f"{prefix}_trade_details.xlsx"
        df_details.to_excel(str(detail_path), index=False)
        print(f"  {prefix} 交易明细已保存: {detail_path}")

        min_len = min(len(values), len(bh_values))
        metrics = _compute_extended_metrics(values[:min_len], list(dates[:min_len]), ppy)
        bh_metrics = _compute_extended_metrics(bh_values[:min_len], list(dates[:min_len]), ppy)
        print_backtest_report(metrics, bh_metrics,
                              label=f"RL Agent ({prefix})", benchmark_label="等权买入持有")
        plot_backtest(
            dates[:min_len], values[:min_len], bh_values[:min_len],
            save_path=str(results_dir / f"{prefix}_backtest.png"),
            title=f"{prefix.capitalize()} Backtest (v1.021)",
        )
        return metrics, bh_metrics

    train_metrics, train_bh = process_backtest(train_dataset, train_ppy, "train", use_random_start=False)
    val_metrics, val_bh = process_backtest(val_dataset, val_ppy, "val", use_random_start=False)
    _build_evaluation_excel(
        train_metrics, train_bh, val_metrics, val_bh,
        save_path=str(results_dir / "evaluation_metrics.xlsx"),
    )

    print(f"\n{'=' * 70}")
    print(f"  完成! 结果: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
