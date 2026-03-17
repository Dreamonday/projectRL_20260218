#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断脚本: 检查训练数据/观测/网络输出中的 NaN/Inf 问题
"""

import sys
import yaml
import numpy as np
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import load_prediction_data, build_trading_dataset
from src.envs.trading_env import MultiStockTradingEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_array(name, arr):
    """检查数组的 NaN/Inf/极端值"""
    has_nan = np.isnan(arr).sum()
    has_inf = np.isinf(arr).sum()
    finite = arr[np.isfinite(arr)]
    if len(finite) > 0:
        print(f"  {name:30s} | shape={str(arr.shape):20s} | "
              f"NaN={has_nan:8d} Inf={has_inf:8d} | "
              f"min={finite.min():12.4f} max={finite.max():12.4f} "
              f"mean={finite.mean():12.4f} std={finite.std():12.4f}")
    else:
        print(f"  {name:30s} | shape={str(arr.shape):20s} | ALL NaN/Inf!")
    return has_nan, has_inf


def main():
    config = load_config(str(project_root / "configs" / "rl_config.yaml"))

    print("=" * 90)
    print("  诊断 1: 加载原始数据")
    print("=" * 90)
    data_cfg = config["data"]
    stock_data = load_prediction_data(
        data_dir=data_cfg["prediction_dir"],
        market_filter=data_cfg["market_filter"],
        max_stocks=data_cfg.get("max_stocks"),
        use_train_data=data_cfg["use_train_data"],
        use_val_data=data_cfg["use_val_data"],
        min_data_points=data_cfg["min_data_points"],
    )
    full_dataset = build_trading_dataset(stock_data)

    print("\n--- 原始数据集数组检查 ---")
    check_array("predicted_close", full_dataset.predicted_close)
    check_array("actual_close", full_dataset.actual_close)
    check_array("last_input_close", full_dataset.last_input_close)
    check_array("has_data", full_dataset.has_data.astype(float))

    split_cfg = config["split"]
    train_dataset, test_dataset = full_dataset.split_by_date(split_cfg["rl_train_end_date"])

    print(f"\n  训练集: {train_dataset.n_dates} 日期, {train_dataset.n_stocks} 股票")
    check_array("train predicted_close", train_dataset.predicted_close)
    check_array("train actual_close", train_dataset.actual_close)

    print("\n" + "=" * 90)
    print("  诊断 2: 检查环境预计算特征")
    print("=" * 90)
    env = MultiStockTradingEnv(
        dataset=train_dataset,
        initial_cash=config["trading"]["initial_cash"],
        transaction_cost=config["trading"]["transaction_cost"],
        max_weight_per_stock=config["trading"]["max_weight_per_stock"],
        reward_type=config["reward"]["type"],
        turnover_penalty=config["reward"]["turnover_penalty"],
    )
    check_array("env.returns", env.returns)
    check_array("env.predicted_return", env.predicted_return)
    check_array("env.pred_error", env.pred_error)
    check_array("env.rolling_pred_error", env.rolling_pred_error)
    check_array("env.rolling_return", env.rolling_return)

    extreme_threshold = 10.0
    for name, arr in [("returns", env.returns), ("predicted_return", env.predicted_return),
                      ("pred_error", env.pred_error)]:
        extreme_count = (np.abs(arr) > extreme_threshold).sum()
        if extreme_count > 0:
            extreme_vals = arr[np.abs(arr) > extreme_threshold]
            print(f"\n  警告: {name} 中有 {extreme_count} 个极端值 (|x|>{extreme_threshold})")
            print(f"    示例: {extreme_vals[:20]}")
            idxs = np.argwhere(np.abs(arr) > extreme_threshold)
            for idx in idxs[:5]:
                t, s = idx
                print(f"    [date={train_dataset.dates[t]}, stock={train_dataset.stock_codes[s]}] "
                      f"value={arr[t, s]:.4f}, "
                      f"actual_close={train_dataset.actual_close[t, s]:.4f}, "
                      f"predicted_close={train_dataset.predicted_close[t, s]:.4f}")

    print("\n" + "=" * 90)
    print("  诊断 3: 检查环境观测值")
    print("=" * 90)
    obs, _ = env.reset()
    print(f"  初始观测 shape: {obs.shape}")
    check_array("initial obs", obs)

    all_obs = [obs.copy()]
    for step in range(min(50, train_dataset.n_dates - 10)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        all_obs.append(obs.copy())
        if terminated or truncated:
            break

    all_obs = np.array(all_obs)
    check_array("50步观测汇总", all_obs)

    nan_steps = np.any(np.isnan(all_obs), axis=1)
    inf_steps = np.any(np.isinf(all_obs), axis=1)
    if nan_steps.any():
        print(f"  含NaN的步数: {np.where(nan_steps)[0]}")
    if inf_steps.any():
        print(f"  含Inf的步数: {np.where(inf_steps)[0]}")

    per_dim_max = np.max(np.abs(all_obs), axis=0)
    extreme_dims = np.where(per_dim_max > 100)[0]
    if len(extreme_dims) > 0:
        print(f"\n  有 {len(extreme_dims)} 个维度最大绝对值 > 100:")
        for d in extreme_dims[:20]:
            print(f"    dim {d}: max_abs={per_dim_max[d]:.4f}")

    print("\n" + "=" * 90)
    print("  诊断 4: 检查 VecNormalize 归一化后的观测")
    print("=" * 90)

    def make_env_fn():
        e = MultiStockTradingEnv(
            dataset=train_dataset,
            initial_cash=config["trading"]["initial_cash"],
            transaction_cost=config["trading"]["transaction_cost"],
            max_weight_per_stock=config["trading"]["max_weight_per_stock"],
            reward_type=config["reward"]["type"],
            turnover_penalty=config["reward"]["turnover_penalty"],
        )
        e.reset(seed=42)
        return e

    vec_env = DummyVecEnv([make_env_fn])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0)

    obs = vec_env.reset()
    print(f"  VecNormalize 初始观测:")
    check_array("vnorm obs[0]", obs)

    all_norm_obs = [obs.copy()]
    for step in range(min(50, train_dataset.n_dates - 10)):
        action = [vec_env.action_space.sample()]
        obs, rewards, dones, infos = vec_env.step(action)
        all_norm_obs.append(obs.copy())
        if dones[0]:
            break

    all_norm_obs = np.concatenate(all_norm_obs, axis=0)
    check_array("VecNormalize 50步汇总", all_norm_obs)

    nan_norm = np.any(np.isnan(all_norm_obs), axis=1)
    inf_norm = np.any(np.isinf(all_norm_obs), axis=1)
    if nan_norm.any():
        print(f"  归一化后含NaN步数: {np.where(nan_norm)[0]}")
    if inf_norm.any():
        print(f"  归一化后含Inf步数: {np.where(inf_norm)[0]}")

    print("\n" + "=" * 90)
    print("  诊断 5: 创建 PPO 模型并测试前向传播")
    print("=" * 90)

    vec_env2 = DummyVecEnv([make_env_fn])
    vec_env2 = VecNormalize(vec_env2, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0)

    rl_cfg = config["rl"]
    policy_kwargs = dict(
        net_arch=dict(
            pi=rl_cfg.get("policy_net", [1024, 512]),
            vf=rl_cfg.get("policy_net", [1024, 512]),
        )
    )

    episode_len = train_dataset.n_dates - 5
    n_steps = min(rl_cfg["n_steps"], episode_len)
    batch_size = min(rl_cfg["batch_size"], n_steps)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env2,
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
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  策略网络参数量: {total_params:,}")

    for name, param in model.policy.named_parameters():
        pdata = param.data
        has_nan = torch.isnan(pdata).sum().item()
        has_inf = torch.isinf(pdata).sum().item()
        if has_nan > 0 or has_inf > 0:
            print(f"  初始参数异常! {name}: NaN={has_nan}, Inf={has_inf}")

    obs_tensor = torch.FloatTensor(all_norm_obs[:5]).to(model.device)
    print(f"\n  测试前向传播 (5个观测):")
    check_array("input obs", obs_tensor.cpu().numpy())

    with torch.no_grad():
        features = model.policy.extract_features(obs_tensor)
        if isinstance(features, tuple):
            pi_features, vf_features = features
        else:
            pi_features = features
            vf_features = features
        print(f"  features shape: {pi_features.shape}")
        check_array("pi_features", pi_features.cpu().numpy())

        latent_pi = model.policy.mlp_extractor.forward_actor(pi_features)
        print(f"  latent_pi shape: {latent_pi.shape}")
        check_array("latent_pi", latent_pi.cpu().numpy())

        mean_actions = model.policy.action_net(latent_pi)
        print(f"  mean_actions shape: {mean_actions.shape}")
        check_array("mean_actions", mean_actions.cpu().numpy())

    print("\n  模拟 512 步 rollout 后训练 (单步 learn):")
    try:
        model.learn(total_timesteps=n_steps, progress_bar=False)
        print("  第1轮 learn 成功!")
    except Exception as e:
        print(f"  第1轮 learn 失败: {e}")

        print("\n  检查 rollout buffer 中的观测:")
        rb = model.rollout_buffer
        rb_obs = rb.observations
        check_array("rollout_buffer obs", rb_obs)

        rb_nan = np.isnan(rb_obs).any(axis=(0, 2))
        rb_inf = np.isinf(rb_obs).any(axis=(0, 2))
        print(f"  Buffer中含NaN的env: {np.where(rb_nan)[0]}")
        print(f"  Buffer中含Inf的env: {np.where(rb_inf)[0]}")

        rb_returns = rb.returns
        rb_advantages = rb.advantages
        check_array("rollout returns", rb_returns)
        check_array("rollout advantages", rb_advantages)

        print("\n  检查训练时的网络参数:")
        for name, param in model.policy.named_parameters():
            pdata = param.data
            has_nan = torch.isnan(pdata).sum().item()
            has_inf = torch.isinf(pdata).sum().item()
            if has_nan > 0 or has_inf > 0:
                print(f"    参数异常: {name}: NaN={has_nan}, Inf={has_inf}, shape={pdata.shape}")

    print("\n" + "=" * 90)
    print("  诊断完成")
    print("=" * 90)


if __name__ == "__main__":
    main()
