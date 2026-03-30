#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证脚本：分析 v1.1 训练时第一个早停 step 为何总是 2280
不修改任何训练代码，仅做信息验证。

逻辑说明：
- 早停评估在 _on_rollout_end 时触发
- 第一次 rollout end 发生在 buffer 收集满 n_steps(2048) 个 transition 时
- 由于按完整 episode 收集，buffer 满的时刻 = 完成某个 episode 后累计 transition 数 >= 2048
- 因此第一次 rollout end 的 num_timesteps = 第一个使得累计 >= 2048 的 episode 序列总长度

若 env 和 model 都使用固定 seed，则 episode 长度序列确定，导致第一次 rollout end 的 step 总是相同。
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 动态加载 train_rl_v1.1（文件名含点，无法直接 import）
_train_spec = importlib.util.spec_from_file_location(
    "train_rl_v11",
    project_root / "scripts" / "train_rl_v1.1_20260317.py",
)
_train_mod = importlib.util.module_from_spec(_train_spec)
_train_spec.loader.exec_module(_train_mod)

load_precomputed_dataset = _train_mod.load_precomputed_dataset
merge_datasets = _train_mod.merge_datasets
_slice_dataset = _train_mod._slice_dataset
MultiStockTradingEnvV10 = _train_mod.MultiStockTradingEnvV10
MIN_EPISODE_WEEKS = _train_mod.MIN_EPISODE_WEEKS

DATASET_DIR = "/data/projectRL_20260218/data/processed_data/rl_dataset_v0.3_202603111056"
N_STEPS = 2048  # 与 rl_config_v1.1 一致


def main():
    dataset_dir = Path(DATASET_DIR)
    meta_path = dataset_dir / "metadata.json"
    train_npz = dataset_dir / "train_dataset.npz"
    test_npz = dataset_dir / "test_dataset.npz"

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    train_raw = load_precomputed_dataset(str(train_npz), metadata)
    test_raw = load_precomputed_dataset(str(test_npz), metadata) if test_npz.exists() else None
    full_dataset = merge_datasets(train_raw, test_raw)

    val_weeks = 5 * 52
    val_indices = np.arange(full_dataset.n_dates - val_weeks, full_dataset.n_dates)
    val_dataset = _slice_dataset(full_dataset, val_indices)
    train_dataset = full_dataset

    W = 30
    lookback = max(W, 4)
    val_start_date = str(val_dataset.dates[0])
    train_dates = train_dataset.dates
    idx_before_val = np.where(train_dates < val_start_date)[0]
    max_train_start_idx = int(idx_before_val[-1]) if len(idx_before_val) > 0 else lookback
    max_train_start_idx = min(max_train_start_idx, train_dataset.n_dates - 1 - MIN_EPISODE_WEEKS)
    if max_train_start_idx < lookback:
        max_train_start_idx = lookback

    periods_per_year = 52
    min_episode_weeks = 3 * periods_per_year

    env_kwargs = dict(
        initial_cash=10_000_000.0,
        transaction_cost=0.001,
        max_weight_per_stock=0.10,
        softmax_temperature=1.0,
        turnover_penalty=0.0,
        temporal_window=W,
        trade_interval=1,
        use_random_start=True,
        max_start_idx=max_train_start_idx,
        min_episode_weeks=min_episode_weeks,
        periods_per_year=periods_per_year,
    )

    def run_first_rollout_steps(seed: int):
        """模拟第一次 rollout 收集：按完整 episode 批量加入 buffer，返回填满时的总 step 数"""
        env = MultiStockTradingEnvV10(dataset=train_dataset, **env_kwargs)
        obs, _ = env.reset(seed=seed)
        total_steps = 0
        buffer_count = 0  # 仅在 episode 结束时增加
        episode_lengths = []
        ep_start = 0

        rng = np.random.default_rng(seed)
        while buffer_count < N_STEPS:
            # 随机 action（episode 长度由 reset 时的 RNG 决定，与 action 无关）
            action = rng.uniform(-1, 1, size=env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1

            if terminated or truncated:
                ep_len = total_steps - ep_start
                buffer_count += ep_len  # 按完整 episode 批量加入，与 collect_rollouts 一致
                episode_lengths.append(ep_len)
                ep_start = total_steps
                obs, _ = env.reset()  # 不传 seed，使用 env 内部 RNG

        return total_steps, episode_lengths

    print("=" * 60)
    print("验证：第一个 rollout end 的 step 是否固定")
    print("=" * 60)
    print(f"n_steps 目标: {N_STEPS}")
    print()

    # 使用与训练相同的 seed=42
    steps_42, lens_42 = run_first_rollout_steps(42)
    print(f"[seed=42] 第一次 rollout end 时 num_timesteps = {steps_42}")
    print(f"  episode 长度序列: {lens_42}")
    print()

    # 再跑一次 seed=42，验证确定性
    steps_42_2, lens_42_2 = run_first_rollout_steps(42)
    print(f"[seed=42 重复] num_timesteps = {steps_42_2}")
    print(f"  与第一次相同: {steps_42 == steps_42_2}")
    print()

    # 换一个 seed 看是否不同
    steps_123, lens_123 = run_first_rollout_steps(123)
    print(f"[seed=123] num_timesteps = {steps_123}")
    print(f"  episode 长度序列: {lens_123}")
    print(f"  与 seed=42 不同: {steps_42 != steps_123}")
    print()

    # 尝试多个 seed 看能否得到 2280
    for s in [0, 1, 42, 123, 456, 999]:
        st, _ = run_first_rollout_steps(s)
        marker = " <-- 接近 2280" if 2270 <= st <= 2290 else ""
        print(f"[seed={s}] 第一次 rollout end = {st}{marker}")

    print()
    print("结论:")
    print(f"  - 固定 seed 时，第一次 rollout end 恒为同一 step（确定性）")
    print(f"  - 早停打印的 step = 首次 _on_rollout_end 时的 num_timesteps")
    print(f"  - 原因：env.reset(seed=42) + model seed=42 使 episode 长度序列确定")
    print(f"  - 具体数值(如 2280)取决于 seed、数据、max_train_start_idx 等")
    print("=" * 60)


if __name__ == "__main__":
    main()
