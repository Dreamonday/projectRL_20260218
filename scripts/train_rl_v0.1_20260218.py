#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL 交易模型训练与回测主脚本
版本: v0.1
日期: 20260218

基于 TimeXer 模型的预测结果，使用 PPO 算法训练多股票组合交易 Agent。

使用方式:
    python scripts/train_rl_v0.1_20260218.py
    python scripts/train_rl_v0.1_20260218.py --config configs/rl_config.yaml
"""

import sys
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import load_prediction_data, build_trading_dataset
from src.envs.trading_env import MultiStockTradingEnv
from src.utils.backtest import (
    evaluate_agent,
    compute_buy_and_hold,
    compute_metrics,
    print_backtest_report,
    plot_backtest,
)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback


class TrainingLogCallback(BaseCallback):
    """训练过程日志回调"""

    def __init__(self, log_interval: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info:
                self.episode_rewards.append(ep_info["r"])
                self.episode_lengths.append(ep_info["l"])

        if self.num_timesteps % (self.log_interval * 256) == 0:
            if self.episode_rewards:
                recent = self.episode_rewards[-20:]
                print(
                    f"  [Step {self.num_timesteps:>8d}] "
                    f"episodes={len(self.episode_rewards)}, "
                    f"avg_reward={np.mean(recent):.4f}, "
                    f"std_reward={np.std(recent):.4f}"
                )
        return True


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(dataset, config, seed=0):
    """创建环境工厂函数（用于 DummyVecEnv）"""
    def _init():
        env = MultiStockTradingEnv(
            dataset=dataset,
            initial_cash=config["trading"]["initial_cash"],
            transaction_cost=config["trading"]["transaction_cost"],
            max_weight_per_stock=config["trading"]["max_weight_per_stock"],
            reward_type=config["reward"]["type"],
            turnover_penalty=config["reward"]["turnover_penalty"],
        )
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="RL 交易模型训练")
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "configs" / "rl_config.yaml"),
        help="配置文件路径",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    print("=" * 70)
    print(f"  RL 交易模型训练 v0.1 — {run_timestamp}")
    print("=" * 70)

    # ======== 1. 加载数据 ========
    print("\n[Step 1] 加载 TimeXer 预测数据...")
    data_cfg = config["data"]
    stock_data = load_prediction_data(
        data_dir=data_cfg["prediction_dir"],
        market_filter=data_cfg["market_filter"],
        max_stocks=data_cfg["max_stocks"],
        use_train_data=data_cfg["use_train_data"],
        use_val_data=data_cfg["use_val_data"],
        min_data_points=data_cfg["min_data_points"],
    )

    print(f"\n[Step 2] 构建交易数据集...")
    full_dataset = build_trading_dataset(stock_data)

    # ======== 2. 拆分训练/测试 ========
    split_cfg = config["split"]
    train_end = split_cfg["rl_train_end_date"]
    print(f"\n[Step 3] 按日期拆分: 训练 <= {train_end} / 测试 > {train_end}")

    train_dataset, test_dataset = full_dataset.split_by_date(train_end)

    if train_dataset is None or train_dataset.n_dates < 10:
        print("错误: 训练集日期太少，请调整 rl_train_end_date")
        sys.exit(1)
    if test_dataset is None or test_dataset.n_dates < 5:
        print("警告: 测试集日期太少，回测结果可能不可靠")

    print(f"  训练集: {train_dataset.n_dates} 日期, {train_dataset.dates[0]} ~ {train_dataset.dates[-1]}")
    if test_dataset:
        print(f"  测试集: {test_dataset.n_dates} 日期, {test_dataset.dates[0]} ~ {test_dataset.dates[-1]}")

    # ======== 3. 创建环境 ========
    print(f"\n[Step 4] 创建训练环境...")
    rl_cfg = config["rl"]

    train_vec_env = DummyVecEnv([make_env(train_dataset, config, seed=42)])
    if rl_cfg.get("normalize_observations", True) or rl_cfg.get("normalize_rewards", True):
        train_vec_env = VecNormalize(
            train_vec_env,
            norm_obs=rl_cfg.get("normalize_observations", True),
            norm_reward=rl_cfg.get("normalize_rewards", True),
            clip_obs=10.0,
            clip_reward=10.0,
        )
    print(f"  观测维度: {train_vec_env.observation_space.shape}")
    print(f"  动作维度: {train_vec_env.action_space.shape}")

    # ======== 4. 创建并训练 PPO ========
    print(f"\n[Step 5] 创建 PPO 模型...")
    policy_kwargs = dict(
        net_arch=dict(
            pi=rl_cfg.get("policy_net", [256, 256]),
            vf=rl_cfg.get("policy_net", [256, 256]),
        )
    )

    episode_len = train_dataset.n_dates - 5
    n_steps = min(rl_cfg["n_steps"], episode_len)
    batch_size = min(rl_cfg["batch_size"], n_steps)
    print(f"  Episode 长度: {episode_len}, n_steps={n_steps}, batch_size={batch_size}")

    model = PPO(
        policy="MlpPolicy",
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
        verbose=1,
        seed=42,
        device="cpu",
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  策略网络参数量: {total_params:,}")
    print(f"  总训练步数: {rl_cfg['total_timesteps']:,}")

    print(f"\n[Step 6] 开始训练...")
    print("-" * 70)
    callback = TrainingLogCallback(log_interval=10)
    model.learn(total_timesteps=rl_cfg["total_timesteps"], callback=callback, progress_bar=True)
    print("-" * 70)
    print("训练完成!")

    # ======== 5. 保存模型 ========
    output_dir = project_root / "experiments" / f"rl_ppo_v0.1_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(exist_ok=True)

    model.save(str(model_dir / "ppo_trading"))
    if isinstance(train_vec_env, VecNormalize):
        train_vec_env.save(str(model_dir / "vec_normalize.pkl"))
    print(f"\n模型已保存: {model_dir}")

    with open(configs_dir / "rl_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    # ======== 6. 训练集回测 ========
    print(f"\n[Step 7] 训练集回测...")
    train_eval_env = MultiStockTradingEnv(
        dataset=train_dataset,
        initial_cash=config["trading"]["initial_cash"],
        transaction_cost=config["trading"]["transaction_cost"],
        max_weight_per_stock=config["trading"]["max_weight_per_stock"],
        reward_type=config["reward"]["type"],
        turnover_penalty=config["reward"]["turnover_penalty"],
    )

    if isinstance(train_vec_env, VecNormalize):
        _eval_vec = DummyVecEnv([lambda: train_eval_env])
        _eval_vec = VecNormalize.load(str(model_dir / "vec_normalize.pkl"), _eval_vec)
        _eval_vec.training = False
        _eval_vec.norm_reward = False

        obs, _ = train_eval_env.reset()
        done = False
        dates_train = [train_dataset.dates[train_eval_env.current_step]]
        values_train = [train_eval_env.portfolio_value]

        while not done:
            obs_norm = _eval_vec.normalize_obs(obs)
            action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = train_eval_env.step(action)
            done = terminated or truncated
            dates_train.append(info["date"])
            values_train.append(info["portfolio_value"])

        values_train = np.array(values_train)
    else:
        train_result = evaluate_agent(train_eval_env, model, deterministic=True)
        dates_train = train_result["dates"]
        values_train = train_result["portfolio_values"]

    bh_train = compute_buy_and_hold(train_eval_env)
    min_len = min(len(values_train), len(bh_train))
    values_train = values_train[:min_len]
    bh_train = bh_train[:min_len]
    dates_train = dates_train[:min_len]

    train_metrics = compute_metrics(values_train, dates_train)
    train_bh_metrics = compute_metrics(bh_train, dates_train)
    print_backtest_report(train_metrics, train_bh_metrics, label="RL Agent (训练集)", benchmark_label="等权买入持有")

    plot_backtest(
        dates_train, values_train, bh_train,
        save_path=str(results_dir / "train_backtest.png"),
        title="Training Period Backtest",
    )

    # ======== 7. 测试集回测 ========
    if test_dataset and test_dataset.n_dates >= 5:
        print(f"\n[Step 8] 测试集回测...")
        test_env = MultiStockTradingEnv(
            dataset=test_dataset,
            initial_cash=config["trading"]["initial_cash"],
            transaction_cost=config["trading"]["transaction_cost"],
            max_weight_per_stock=config["trading"]["max_weight_per_stock"],
            reward_type=config["reward"]["type"],
            turnover_penalty=config["reward"]["turnover_penalty"],
        )

        if isinstance(train_vec_env, VecNormalize):
            _eval_vec_test = DummyVecEnv([lambda: test_env])
            _eval_vec_test = VecNormalize.load(str(model_dir / "vec_normalize.pkl"), _eval_vec_test)
            _eval_vec_test.training = False
            _eval_vec_test.norm_reward = False

            obs, _ = test_env.reset()
            done = False
            dates_test = [test_dataset.dates[test_env.current_step]]
            values_test = [test_env.portfolio_value]
            infos_test = []

            while not done:
                obs_norm = _eval_vec_test.normalize_obs(obs)
                action, _ = model.predict(obs_norm, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                dates_test.append(info["date"])
                values_test.append(info["portfolio_value"])
                infos_test.append(info)

            values_test = np.array(values_test)
        else:
            test_result = evaluate_agent(test_env, model, deterministic=True)
            dates_test = test_result["dates"]
            values_test = test_result["portfolio_values"]
            infos_test = test_result["infos"]

        bh_test = compute_buy_and_hold(test_env)
        min_len = min(len(values_test), len(bh_test))
        values_test = values_test[:min_len]
        bh_test = bh_test[:min_len]
        dates_test = dates_test[:min_len]

        test_metrics = compute_metrics(values_test, dates_test)
        test_bh_metrics = compute_metrics(bh_test, dates_test)
        print_backtest_report(test_metrics, test_bh_metrics, label="RL Agent (测试集)", benchmark_label="等权买入持有")

        plot_backtest(
            dates_test, values_test, bh_test,
            save_path=str(results_dir / "test_backtest.png"),
            title="Test Period Backtest",
        )

        # 保存测试集交易明细
        if infos_test:
            trade_details = []
            for info in infos_test:
                trade_details.append({
                    "date": info["date"],
                    "portfolio_value": info["portfolio_value"],
                    "portfolio_return": info["portfolio_return"],
                    "turnover": info["turnover"],
                    "trade_cost": info["trade_cost"],
                    "cash_ratio": info["cash_ratio"],
                    "invested_ratio": info["invested_ratio"],
                    "n_active_stocks": int(info["n_active_stocks"]),
                })
            pd_details = pd.DataFrame(trade_details)
            pd_details.to_csv(results_dir / "test_trade_details.csv", index=False)
            print(f"交易明细已保存: {results_dir / 'test_trade_details.csv'}")

    # ======== 8. 保存汇总 ========
    summary = {
        "run_timestamp": run_timestamp,
        "n_stocks": full_dataset.n_stocks,
        "stock_codes": full_dataset.stock_codes,
        "train_dates": f"{train_dataset.dates[0]} ~ {train_dataset.dates[-1]}",
        "train_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in train_metrics.items()},
    }
    if test_dataset and test_dataset.n_dates >= 5:
        summary["test_dates"] = f"{test_dataset.dates[0]} ~ {test_dataset.dates[-1]}"
        summary["test_metrics"] = {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in test_metrics.items()}

    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  全部完成! 结果保存在: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
