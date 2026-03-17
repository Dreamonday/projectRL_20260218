#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL 交易模型训练与回测主脚本
版本: v0.2
日期: 20260309

v0.2 变更:
  - 去掉 max_stocks 限制，使用所有通过过滤的股票
  - 单只股票最大仓位从 20% 提升至 50%
  - 策略网络增大 [1024, 512]，适配更大动作/观测空间
  - 使用 GPU (cuda) 训练
  - 根据数据实际频率自动计算年化因子 (月频=12, 周频=52)

使用方式:
    python scripts/train_rl_v0.2_20260309104747.py
    python scripts/train_rl_v0.2_20260309104747.py --config configs/rl_config.yaml
"""

import sys
import time
import yaml
import json
import argparse
import numpy as np
import pandas as pd
import torch
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
    estimate_periods_per_year,
    print_backtest_report,
    plot_backtest,
)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback


class TrainingLogCallback(BaseCallback):
    """ANSI 多行表格覆写训练进度回调"""

    def __init__(self, total_timesteps: int, n_steps: int = 512, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps
        self.total_iters = total_timesteps // n_steps
        self._start_time = None
        self._last_n_lines = 0

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _fmt_kl(self, v):
        if np.isnan(v):
            return "---"
        if abs(v) >= 1e6:
            return f"{v:.1e}"
        if abs(v) >= 1000:
            return f"{v:.0f}"
        return f"{v:.2f}"

    def _fmt_time(self, seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h{m:02d}m{s:02d}s"
        return f"{m}m{s:02d}s"

    def _build_table(self):
        cur_iter = self.num_timesteps // self.n_steps
        elapsed = time.time() - self._start_time if self._start_time else 0
        fps = self.num_timesteps / max(elapsed, 1e-6)
        pct = self.num_timesteps / self.total_timesteps * 100

        if elapsed > 0 and pct > 0:
            eta_sec = elapsed / pct * (100 - pct)
            eta_str = self._fmt_time(eta_sec)
        else:
            eta_str = "---"

        logger = {}
        try:
            logger = self.model.logger.name_to_value
        except Exception:
            pass

        loss = logger.get("train/loss", float("nan"))
        vl = logger.get("train/value_loss", float("nan"))
        pg = logger.get("train/policy_gradient_loss", float("nan"))
        kl = logger.get("train/approx_kl", float("nan"))
        clip = logger.get("train/clip_fraction", float("nan"))
        ev = logger.get("train/explained_variance", float("nan"))
        ent = logger.get("train/entropy_loss", float("nan"))
        lr = logger.get("train/learning_rate", float("nan"))
        std = logger.get("train/std", float("nan"))
        n_upd = logger.get("train/n_updates", float("nan"))

        pbar_w = 30
        filled = int(pbar_w * pct / 100)
        bar = "█" * filled + "░" * (pbar_w - filled)

        rows = [
            ("progress/", ""),
            ("   iterations", f"{cur_iter} / {self.total_iters}"),
            ("   timesteps", f"{self.num_timesteps} / {self.total_timesteps}"),
            ("   progress", f"{bar} {pct:5.1f}%"),
            ("time/", ""),
            ("   fps", f"{fps:.0f}"),
            ("   elapsed", self._fmt_time(elapsed)),
            ("   eta", eta_str),
            ("train/", ""),
            ("   loss", f"{loss:.4f}" if not np.isnan(loss) else "---"),
            ("   value_loss", f"{vl:.4f}" if not np.isnan(vl) else "---"),
            ("   policy_grad_loss", f"{pg:.4f}" if not np.isnan(pg) else "---"),
            ("   approx_kl", self._fmt_kl(kl)),
            ("   clip_fraction", f"{clip:.3f}" if not np.isnan(clip) else "---"),
            ("   entropy_loss", f"{ent:.1f}" if not np.isnan(ent) else "---"),
            ("   explained_var", f"{ev:+.3f}" if not np.isnan(ev) else "---"),
            ("   learning_rate", f"{lr:.6f}" if not np.isnan(lr) else "---"),
            ("   std", f"{std:.4f}" if not np.isnan(std) else "---"),
            ("   n_updates", f"{n_upd:.0f}" if not np.isnan(n_upd) else "---"),
        ]

        key_w = max(len(r[0]) for r in rows) + 1
        val_w = max(len(r[1]) for r in rows) + 1
        total_w = key_w + val_w + 5
        sep = "-" * total_w

        lines = [sep]
        for k, v in rows:
            if v == "":
                lines.append(f"| {k:<{key_w}}|{' ' * (val_w + 2)}|")
            else:
                lines.append(f"| {k:<{key_w}}| {v:>{val_w}} |")
        lines.append(sep)
        return lines

    def _on_rollout_end(self) -> None:
        lines = self._build_table()
        n = len(lines)

        buf = ""
        if self._last_n_lines > 0:
            buf += f"\033[{self._last_n_lines}A\r"
        for line in lines:
            buf += f"\033[2K{line}\n"

        sys.stdout.write(buf)
        sys.stdout.flush()
        self._last_n_lines = n

    def _on_training_end(self) -> None:
        sys.stdout.write("\n")

    def _on_step(self) -> bool:
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


def print_gpu_info():
    """打印 GPU 信息"""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"  GPU: {gpu.name}")
        print(f"  显存: {gpu.total_memory / 1024**3:.1f} GB")
        print(f"  CUDA: {torch.version.cuda}")
    else:
        print("  未检测到 GPU，将使用 CPU 训练")


def save_trade_details(infos, initial_cash, save_path):
    """保存交易明细到 Excel"""
    if not infos:
        return

    details = []
    for info in infos:
        # 格式化持仓字符串: "CODE:0.20; CODE2:0.10"
        pos_str = ""
        if "positions" in info and info["positions"]:
            items = [f"{k}:{v:.4f}" for k, v in info["positions"].items()]
            pos_str = "; ".join(items)

        current_value = info["portfolio_value"]
        details.append({
            "日期": info["date"],
            "总资产": current_value,
            "累计盈亏": current_value - initial_cash,
            "当日收益率": info["portfolio_return"],
            "换手率": info["turnover"],
            "交易成本": info["trade_cost"],
            "现金比例": info["cash_ratio"],
            "现金价值": current_value * info["cash_ratio"],
            "股票仓位": info["invested_ratio"],
            "持股数量": int(info["n_active_stocks"]),
            "持仓明细": pos_str
        })

    df = pd.DataFrame(details)
    # 确保 save_path 以 .xlsx 结尾
    if str(save_path).endswith(".csv"):
        save_path = str(save_path).replace(".csv", ".xlsx")
    
    df.to_excel(save_path, index=False)
    print(f"交易明细已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="RL 交易模型训练 v0.2")
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
    print(f"  RL 交易模型训练 v0.2 — {run_timestamp}")
    print("=" * 70)
    print_gpu_info()

    # ======== 1. 加载数据 ========
    print("\n[Step 1] 加载 TimeXer 预测数据...")
    data_cfg = config["data"]
    stock_data = load_prediction_data(
        data_dir=data_cfg["prediction_dir"],
        market_filter=data_cfg["market_filter"],
        max_stocks=data_cfg.get("max_stocks"),
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

    train_ppy = estimate_periods_per_year(train_dataset.dates)
    print(f"  训练集: {train_dataset.n_dates} 日期, {train_dataset.dates[0]} ~ {train_dataset.dates[-1]}")
    print(f"  训练集频率: ~{train_ppy:.0f} 期/年")
    if test_dataset:
        test_ppy = estimate_periods_per_year(test_dataset.dates)
        print(f"  测试集: {test_dataset.n_dates} 日期, {test_dataset.dates[0]} ~ {test_dataset.dates[-1]}")
        print(f"  测试集频率: ~{test_ppy:.0f} 期/年")

    # ======== 3. 创建环境 ========
    print(f"\n[Step 4] 创建训练环境...")
    rl_cfg = config["rl"]
    device = rl_cfg.get("device", "auto")
    if device == "cuda" and not torch.cuda.is_available():
        print("  警告: 配置为 cuda 但未检测到 GPU，回退到 cpu")
        device = "cpu"

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
    print(f"  训练设备: {device}")

    # ======== 4. 创建并训练 PPO ========
    print(f"\n[Step 5] 创建 PPO 模型...")
    policy_kwargs = dict(
        net_arch=dict(
            pi=rl_cfg.get("policy_net", [1024, 512]),
            vf=rl_cfg.get("policy_net", [1024, 512]),
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
        verbose=0,
        seed=42,
        device=device,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  策略网络参数量: {total_params:,}")
    print(f"  总训练步数: {rl_cfg['total_timesteps']:,}")

    if device == "cuda":
        mem_allocated = torch.cuda.memory_allocated() / 1024**2
        print(f"  GPU 显存占用 (模型加载后): {mem_allocated:.1f} MB")

    total_ts = rl_cfg["total_timesteps"]
    total_iters = total_ts // n_steps
    print(f"\n[Step 6] 开始训练 ({total_iters} iterations)...")
    callback = TrainingLogCallback(total_timesteps=total_ts, n_steps=n_steps)
    model.learn(total_timesteps=total_ts, callback=callback, progress_bar=False)
    print("训练完成!")

    if device == "cuda":
        mem_peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  GPU 峰值显存: {mem_peak:.1f} MB")

    # ======== 5. 保存模型 ========
    output_dir = project_root / "experiments" / f"rl_ppo_v0.2_{run_timestamp}"
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
        infos_train = []

        while not done:
            obs_norm = _eval_vec.normalize_obs(obs)
            action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = train_eval_env.step(action)
            done = terminated or truncated
            dates_train.append(info["date"])
            values_train.append(info["portfolio_value"])
            infos_train.append(info)

        values_train = np.array(values_train)
    else:
        train_result = evaluate_agent(train_eval_env, model, deterministic=True)
        dates_train = train_result["dates"]
        values_train = train_result["portfolio_values"]
        infos_train = train_result["infos"]

    bh_train = compute_buy_and_hold(train_eval_env)
    min_len = min(len(values_train), len(bh_train))
    values_train = values_train[:min_len]
    bh_train = bh_train[:min_len]
    dates_train = dates_train[:min_len]

    train_metrics = compute_metrics(values_train, dates_train, periods_per_year=train_ppy)
    train_bh_metrics = compute_metrics(bh_train, dates_train, periods_per_year=train_ppy)
    print_backtest_report(train_metrics, train_bh_metrics, label="RL Agent (训练集)", benchmark_label="等权买入持有")

    plot_backtest(
        dates_train, values_train, bh_train,
        save_path=str(results_dir / "train_backtest.png"),
        title="Training Period Backtest",
    )

    if infos_train:
        save_trade_details(infos_train, config["trading"]["initial_cash"], results_dir / "train_trade_details.xlsx")

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

        test_metrics = compute_metrics(values_test, dates_test, periods_per_year=test_ppy)
        test_bh_metrics = compute_metrics(bh_test, dates_test, periods_per_year=test_ppy)
        print_backtest_report(test_metrics, test_bh_metrics, label="RL Agent (测试集)", benchmark_label="等权买入持有")

        plot_backtest(
            dates_test, values_test, bh_test,
            save_path=str(results_dir / "test_backtest.png"),
            title="Test Period Backtest",
        )

        if infos_test:
            save_trade_details(infos_test, config["trading"]["initial_cash"], results_dir / "test_trade_details.xlsx")

    # ======== 8. 保存汇总 ========
    summary = {
        "version": "v0.2",
        "run_timestamp": run_timestamp,
        "device": device,
        "n_stocks": full_dataset.n_stocks,
        "stock_codes": full_dataset.stock_codes,
        "train_dates": f"{train_dataset.dates[0]} ~ {train_dataset.dates[-1]}",
        "train_periods_per_year": train_ppy,
        "train_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in train_metrics.items()},
    }
    if test_dataset and test_dataset.n_dates >= 5:
        summary["test_dates"] = f"{test_dataset.dates[0]} ~ {test_dataset.dates[-1]}"
        summary["test_periods_per_year"] = test_ppy
        summary["test_metrics"] = {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in test_metrics.items()}

    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  全部完成! 结果保存在: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
