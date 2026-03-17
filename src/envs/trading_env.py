#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多股票组合交易环境 (Gymnasium 接口)
版本: v0.1
日期: 20260218

State:  per-stock features + global features
Action: 连续仓位权重 (0~1 per stock)
Reward: 组合收益率 - 换手成本
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.data_loader import TradingDataset


N_FEATURES_PER_STOCK = 6
N_GLOBAL_FEATURES = 2
LOOKBACK_STEPS = 4


class MultiStockTradingEnv(gym.Env):
    """
    多股票组合交易环境。

    观测空间 (per stock × n_stocks + global):
        0: predicted_return     - TimeXer 预测收益率
        1: actual_return_1step  - 上一步实际收益率
        2: prediction_error     - 当期预测误差
        3: rolling_pred_error   - 滚动平均预测误差 (4期)
        4: rolling_return       - 滚动平均实际收益 (4期)
        5: current_weight       - 当前持仓权重
      global:
        -2: cash_ratio          - 现金比例
        -1: portfolio_return    - 累计组合收益率

    动作空间: Box(low=-1, high=1, shape=(n_stocks,))
        映射到 [0, 1] 后归一化为组合权重，剩余部分为现金。
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dataset: TradingDataset,
        initial_cash: float = 10_000_000.0,
        transaction_cost: float = 0.001,
        max_weight_per_stock: float = 0.2,
        reward_type: str = "log_return",
        turnover_penalty: float = 1.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.dataset = dataset
        self.n_stocks = dataset.n_stocks
        self.n_dates = dataset.n_dates
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_weight_per_stock = max_weight_per_stock
        self.reward_type = reward_type
        self.turnover_penalty = turnover_penalty
        self.render_mode = render_mode

        self._precompute_features()

        obs_dim = self.n_stocks * N_FEATURES_PER_STOCK + N_GLOBAL_FEATURES
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_stocks,), dtype=np.float32
        )

        self.current_step = 0
        self.weights = np.zeros(self.n_stocks, dtype=np.float64)
        self.cash_ratio = 1.0
        self.portfolio_value = initial_cash
        self.portfolio_values_history = []

    def _precompute_features(self):
        """预计算所有日期的股票特征，加速 step()"""
        ds = self.dataset
        n, m = ds.n_dates, ds.n_stocks

        self.returns = np.zeros((n, m))
        for t in range(1, n):
            prev = ds.actual_close[t - 1]
            curr = ds.actual_close[t]
            valid = (prev > 0) & (~np.isnan(prev)) & (~np.isnan(curr))
            self.returns[t, valid] = (curr[valid] - prev[valid]) / prev[valid]

        self.predicted_return = np.zeros((n, m))
        for t in range(n):
            base = ds.actual_close[t]
            pred = ds.predicted_close[t]
            valid = (base > 0) & (~np.isnan(base)) & (~np.isnan(pred))
            self.predicted_return[t, valid] = (pred[valid] - base[valid]) / base[valid]

        self.pred_error = np.zeros((n, m))
        for t in range(n):
            act = ds.actual_close[t]
            pred = ds.predicted_close[t]
            valid = (act > 0) & (~np.isnan(act)) & (~np.isnan(pred))
            self.pred_error[t, valid] = (pred[valid] - act[valid]) / act[valid]

        self.rolling_pred_error = np.zeros((n, m))
        self.rolling_return = np.zeros((n, m))
        for t in range(n):
            start = max(0, t - LOOKBACK_STEPS + 1)
            window = t - start + 1
            self.rolling_pred_error[t] = np.nanmean(self.pred_error[start:t + 1], axis=0)
            self.rolling_return[t] = np.nanmean(self.returns[start:t + 1], axis=0)

        self.rolling_pred_error = np.nan_to_num(self.rolling_pred_error, 0.0)
        self.rolling_return = np.nan_to_num(self.rolling_return, 0.0)
        self.predicted_return = np.nan_to_num(self.predicted_return, 0.0)
        self.returns = np.nan_to_num(self.returns, 0.0)
        self.pred_error = np.nan_to_num(self.pred_error, 0.0)

    def _get_obs(self) -> np.ndarray:
        t = self.current_step
        stock_features = np.column_stack([
            self.predicted_return[t],
            self.returns[t],
            self.pred_error[t],
            self.rolling_pred_error[t],
            self.rolling_return[t],
            self.weights,
        ])  # (n_stocks, 6)

        global_features = np.array([
            self.cash_ratio,
            (self.portfolio_value / self.initial_cash) - 1.0,
        ], dtype=np.float64)

        obs = np.concatenate([stock_features.flatten(), global_features])
        
        # 检查 NaN 并替换为 0
        if np.isnan(obs).any() or np.isinf(obs).any():
             # 使用 np.nan_to_num 将 NaN 替换为 0.0，同时处理 Infinity
             obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 增加硬截断，防止极端值破坏 VecNormalize 或 PPO
        obs = np.clip(obs, -20.0, 20.0)
             
        return obs.astype(np.float32)

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """将 PPO 原始动作转换为组合权重"""
        raw = (action + 1.0) / 2.0  # [-1,1] → [0,1]
        raw = np.clip(raw, 0.0, 1.0)

        # Top-K 截断 (v0.4.2 change: 强制保留 Top-30)
        K = 30
        if self.n_stocks > K:
            # np.partition 会把数组分为两部分：前 n-K 个是较小的，后 K 个是较大的
            # 但不保证顺序。我们只需要用第 -K 个元素作为阈值即可。
            # 注意：如果有很多相同的元素，可能会有问题，但对于连续动作通常还好。
            
            # 使用 argpartition 找到分界线索引
            k_indices = np.argpartition(raw, -K)[-K:]
            
            # 创建新的掩码，只保留这些索引
            mask = np.zeros_like(raw, dtype=bool)
            mask[k_indices] = True
            
            # 将不在 Top-K 中的置零
            raw[~mask] = 0.0

        raw = np.minimum(raw, self.max_weight_per_stock)

        total = raw.sum()
        if total > 1.0:
            raw = raw / total

        return raw

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = LOOKBACK_STEPS
        self.weights = np.zeros(self.n_stocks, dtype=np.float64)
        self.cash_ratio = 1.0
        self.portfolio_value = self.initial_cash
        self.portfolio_values_history = [self.initial_cash]
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
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

        stock_returns = self.returns[self.current_step]
        portfolio_return = np.dot(self.weights, stock_returns) - trade_cost

        self.portfolio_value *= (1.0 + portfolio_return)
        self.portfolio_values_history.append(self.portfolio_value)

        if self.reward_type == "log_return":
            reward = np.log1p(np.clip(portfolio_return, -0.99, None))
        elif self.reward_type == "excess_return":
            # 计算市场平均收益率 (等权)
            market_return = np.mean(stock_returns)
            # 超额收益 = 组合收益 - 市场收益
            # 不再放大 10 倍，防止梯度爆炸 (v0.4.1 fix)
            reward = (portfolio_return - market_return) * 1.0
        else:
            reward = portfolio_return

        reward -= self.turnover_penalty * trade_cost

        new_weights_after_return = self.weights * (1.0 + stock_returns)
        total_after = new_weights_after_return.sum() + self.cash_ratio
        if total_after > 1e-10:
            self.weights = new_weights_after_return / total_after
            self.cash_ratio = self.cash_ratio / total_after
        else:
            self.weights = np.zeros(self.n_stocks)
            self.cash_ratio = 1.0

        obs = self._get_obs()
        # 获取持仓明细 (强制显示权重最大的前 100 只股票)
        # 1. 找到所有权重 > 0 的股票索引
        all_active_indices = np.where(self.weights > 0)[0]
        
        positions = {}
        if len(all_active_indices) > 0:
            # 2. 获取对应的股票代码和权重
            all_codes = np.array(self.dataset.stock_codes)[all_active_indices]
            all_weights = self.weights[all_active_indices]
            
            # 3. 按权重降序排列
            sorted_indices = np.argsort(all_weights)[::-1]
            
            # 4. 取前 100 个 (如果不足 100 个则取全部)
            top_k = min(100, len(sorted_indices))
            top_indices = sorted_indices[:top_k]
            
            for i, idx in enumerate(top_indices):
                # idx 是在 all_weights/all_codes 中的索引
                positions[all_codes[idx]] = float(all_weights[idx])

        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "turnover": turnover,
            "trade_cost": trade_cost,
            "date": self.dataset.dates[self.current_step],
            "cash_ratio": self.cash_ratio,
            "invested_ratio": self.weights.sum(),
            "n_active_stocks": (self.weights > 1e-6).sum(),  # 统计只要有微量持仓就算
            "positions": positions,
        }

        return obs, float(reward), terminated, truncated, info
