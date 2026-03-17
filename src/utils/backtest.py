#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测评估模块
版本: v0.1
日期: 20260218
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def evaluate_agent(env, model, deterministic: bool = True) -> dict:
    """
    运行一次完整回测。

    Args:
        env: 交易环境（原始环境，非 VecEnv 包装）
        model: 训练好的 SB3 模型
        deterministic: 是否使用确定性策略

    Returns:
        dict 包含: dates, portfolio_values, actions, infos
    """
    obs, _ = env.reset()
    done = False

    dates = []
    portfolio_values = []
    actions_history = []
    infos_history = []
    rewards_history = []

    portfolio_values.append(env.portfolio_value)
    dates.append(env.dataset.dates[env.current_step])

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        dates.append(info["date"])
        portfolio_values.append(info["portfolio_value"])
        actions_history.append(action.copy())
        infos_history.append(info)
        rewards_history.append(reward)

    return {
        "dates": dates,
        "portfolio_values": np.array(portfolio_values),
        "actions": actions_history,
        "infos": infos_history,
        "rewards": np.array(rewards_history),
    }


def estimate_periods_per_year(dates) -> float:
    """根据日期序列的实际间隔自动推算年化因子"""
    if len(dates) < 2:
        return 52.0
    date_objs = pd.to_datetime(dates)
    diffs_days = np.diff(date_objs).astype("timedelta64[D]").astype(float)
    median_interval = np.median(diffs_days)
    if median_interval > 20:
        return 12.0   # 月频
    elif median_interval > 5:
        return 52.0   # 周频
    else:
        return 252.0  # 日频


def compute_buy_and_hold(env) -> np.ndarray:
    """计算等权买入持有的组合价值曲线"""
    ds = env.dataset
    start = 4  # LOOKBACK_STEPS
    n = ds.n_dates

    stock_returns = np.zeros((n, ds.n_stocks))
    for t in range(1, n):
        prev = ds.actual_close[t - 1]
        curr = ds.actual_close[t]
        valid = (prev > 0) & (~np.isnan(prev)) & (~np.isnan(curr))
        stock_returns[t, valid] = (curr[valid] - prev[valid]) / prev[valid]

    stock_returns = np.nan_to_num(stock_returns, 0.0)
    equal_weight = 1.0 / ds.n_stocks

    values = [env.initial_cash]
    for t in range(start + 1, n):
        port_ret = np.sum(stock_returns[t] * equal_weight)
        values.append(values[-1] * (1.0 + port_ret))

    return np.array(values)


def compute_metrics(portfolio_values: np.ndarray, dates: list, periods_per_year: float = 52.0) -> dict:
    """
    计算回测指标。

    Args:
        portfolio_values: 组合净值序列
        dates: 日期序列
        periods_per_year: 年化因子（周频=52, 月频=12, 日频=252）
    """
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        return {"error": "no valid returns"}

    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1.0

    n_periods = len(returns)
    annualized_return = (1.0 + total_return) ** (periods_per_year / max(n_periods, 1)) - 1.0

    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    cum_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - cum_max) / cum_max
    max_drawdown = drawdowns.min()

    dd_end = np.argmin(drawdowns)
    dd_start = np.argmax(portfolio_values[:dd_end + 1]) if dd_end > 0 else 0

    win_rate = (returns > 0).sum() / max(len(returns), 1)

    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
    avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 1e-10
    profit_loss_ratio = avg_win / max(avg_loss, 1e-10)

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "max_drawdown_start": dates[dd_start] if dd_start < len(dates) else None,
        "max_drawdown_end": dates[dd_end] if dd_end < len(dates) else None,
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss_ratio,
        "n_periods": n_periods,
        "avg_return_per_period": returns.mean(),
        "std_return_per_period": returns.std(),
        "final_value": portfolio_values[-1],
        "initial_value": portfolio_values[0],
    }


def print_backtest_report(
    metrics: dict,
    benchmark_metrics: Optional[dict] = None,
    label: str = "RL Agent",
    benchmark_label: str = "Buy & Hold",
):
    """打印回测报告"""
    print("\n" + "=" * 70)
    print(f"  回测报告: {label}")
    print("=" * 70)

    def _fmt(key, fmt_str, metrics_dict):
        v = metrics_dict.get(key)
        if v is None:
            return "N/A"
        return fmt_str.format(v)

    rows = [
        ("初始资金", _fmt("initial_value", "${:,.0f}", metrics)),
        ("最终资金", _fmt("final_value", "${:,.0f}", metrics)),
        ("总收益率", _fmt("total_return", "{:.2%}", metrics)),
        ("年化收益率", _fmt("annualized_return", "{:.2%}", metrics)),
        ("夏普比率", _fmt("sharpe_ratio", "{:.3f}", metrics)),
        ("最大回撤", _fmt("max_drawdown", "{:.2%}", metrics)),
        ("胜率", _fmt("win_rate", "{:.2%}", metrics)),
        ("盈亏比", _fmt("profit_loss_ratio", "{:.2f}", metrics)),
        ("交易周期数", _fmt("n_periods", "{:d}", metrics)),
    ]

    for name, val in rows:
        print(f"  {name:<15s}: {val}")

    if benchmark_metrics and "error" not in benchmark_metrics:
        print(f"\n  --- 对比: {benchmark_label} ---")
        bm_rows = [
            ("总收益率", _fmt("total_return", "{:.2%}", benchmark_metrics)),
            ("年化收益率", _fmt("annualized_return", "{:.2%}", benchmark_metrics)),
            ("夏普比率", _fmt("sharpe_ratio", "{:.3f}", benchmark_metrics)),
            ("最大回撤", _fmt("max_drawdown", "{:.2%}", benchmark_metrics)),
        ]
        for name, val in bm_rows:
            print(f"  {name:<15s}: {val}")

        alpha = metrics.get("annualized_return", 0) - benchmark_metrics.get("annualized_return", 0)
        print(f"\n  超额年化收益 (Alpha): {alpha:.2%}")

    print("=" * 70)


def plot_backtest(
    dates: list,
    portfolio_values: np.ndarray,
    benchmark_values: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "RL Portfolio Backtest",
):
    """绘制回测净值曲线"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    x = range(len(portfolio_values))
    dates_labels = [str(d) for d in dates]

    ax1 = axes[0]
    ax1.plot(x, portfolio_values / portfolio_values[0], label="RL Agent", linewidth=1.5, color="steelblue")
    if benchmark_values is not None and len(benchmark_values) == len(portfolio_values):
        ax1.plot(x, benchmark_values / benchmark_values[0], label="Buy & Hold (Equal Weight)",
                 linewidth=1.2, color="gray", linestyle="--")
    ax1.set_ylabel("Normalized Portfolio Value")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    n_labels = min(10, len(dates_labels))
    step = max(1, len(dates_labels) // n_labels)
    ax1.set_xticks(range(0, len(dates_labels), step))
    ax1.set_xticklabels([dates_labels[i] for i in range(0, len(dates_labels), step)], rotation=45, fontsize=8)

    ax2 = axes[1]
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    colors = ["green" if r > 0 else "red" for r in returns]
    ax2.bar(range(len(returns)), returns * 100, color=colors, alpha=0.7, width=0.8)
    ax2.set_ylabel("Period Return (%)")
    ax2.set_xlabel("Trading Period")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"回测图表已保存: {save_path}")
    plt.close(fig)
