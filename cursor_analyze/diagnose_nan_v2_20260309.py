#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断脚本 v2: 深入检查 PPO 训练阶段 NaN 产生的具体环节
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


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_array(name, arr):
    has_nan = np.isnan(arr).sum()
    has_inf = np.isinf(arr).sum()
    finite = arr[np.isfinite(arr)]
    if len(finite) > 0:
        print(f"  {name:30s} | shape={str(arr.shape):20s} | "
              f"NaN={has_nan:6d} Inf={has_inf:6d} | "
              f"min={finite.min():12.4f} max={finite.max():12.4f} "
              f"mean={finite.mean():12.4f} std={finite.std():12.4f}")
    else:
        print(f"  {name:30s} | ALL NaN/Inf!")


def main():
    config = load_config(str(project_root / "configs" / "rl_config.yaml"))

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
    split_cfg = config["split"]
    train_dataset, _ = full_dataset.split_by_date(split_cfg["rl_train_end_date"])

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

    rl_cfg = config["rl"]
    episode_len = train_dataset.n_dates - 5
    n_steps = min(rl_cfg["n_steps"], episode_len)
    batch_size = min(rl_cfg["batch_size"], n_steps)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_kwargs = dict(
        net_arch=dict(
            pi=rl_cfg.get("policy_net", [1024, 512]),
            vf=rl_cfg.get("policy_net", [1024, 512]),
        )
    )

    model = PPO(
        policy="MlpPolicy", env=vec_env,
        learning_rate=rl_cfg["learning_rate"],
        n_steps=n_steps, batch_size=batch_size,
        n_epochs=rl_cfg["n_epochs"],
        gamma=rl_cfg["gamma"], gae_lambda=rl_cfg["gae_lambda"],
        clip_range=rl_cfg["clip_range"],
        ent_coef=rl_cfg["ent_coef"], vf_coef=rl_cfg["vf_coef"],
        max_grad_norm=rl_cfg["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        verbose=0, seed=42, device=device,
    )

    print("=" * 90)
    print("  步骤 1: 手动收集 rollout")
    print("=" * 90)

    from stable_baselines3.common.callbacks import CallbackList
    model._setup_learn(total_timesteps=n_steps, callback=None)
    cb = CallbackList([])
    cb.init_callback(model)
    model.rollout_buffer.reset()
    cb.on_rollout_start()
    model.collect_rollouts(model.env, cb, model.rollout_buffer, n_steps)

    rb = model.rollout_buffer
    print(f"\n  Rollout buffer 容量: {rb.buffer_size}, pos: {rb.pos}")
    check_array("observations", rb.observations)
    check_array("actions", rb.actions)
    check_array("rewards", rb.rewards)
    check_array("values", rb.values)
    check_array("log_probs", rb.log_probs)
    check_array("advantages", rb.advantages)
    check_array("returns", rb.returns)

    print("\n" + "=" * 90)
    print("  步骤 2: 开启 anomaly detection 手动训练 1 个 mini-batch")
    print("=" * 90)

    torch.autograd.set_detect_anomaly(True)

    for batch_idx, rollout_data in enumerate(rb.get(batch_size)):
        print(f"\n--- Mini-batch {batch_idx} ---")

        obs_t = rollout_data.observations
        act_t = rollout_data.actions
        old_values = rollout_data.old_values
        old_log_prob = rollout_data.old_log_prob
        advantages = rollout_data.advantages
        returns_t = rollout_data.returns

        check_array(f"  batch obs", obs_t.cpu().numpy())
        check_array(f"  batch actions", act_t.cpu().numpy())
        check_array(f"  batch advantages", advantages.cpu().numpy())
        check_array(f"  batch returns", returns_t.cpu().numpy())
        check_array(f"  batch old_log_prob", old_log_prob.cpu().numpy())
        check_array(f"  batch old_values", old_values.cpu().numpy())

        print(f"\n  前向传播:")
        try:
            features = model.policy.extract_features(obs_t)
            if isinstance(features, tuple):
                pi_features, vf_features = features
            else:
                pi_features, vf_features = features, features

            latent_pi = model.policy.mlp_extractor.forward_actor(pi_features)
            latent_vf = model.policy.mlp_extractor.forward_critic(vf_features)

            check_array("  latent_pi", latent_pi.detach().cpu().numpy())
            check_array("  latent_vf", latent_vf.detach().cpu().numpy())

            mean_actions = model.policy.action_net(latent_pi)
            check_array("  mean_actions", mean_actions.detach().cpu().numpy())

            values_pred = model.policy.value_net(latent_vf)
            check_array("  values_pred", values_pred.detach().cpu().numpy())

            values, log_prob, entropy = model.policy.evaluate_actions(obs_t, act_t)
            check_array("  new_values", values.detach().cpu().numpy())
            check_array("  new_log_prob", log_prob.detach().cpu().numpy())
            check_array("  entropy", entropy.detach().cpu().numpy())

        except Exception as e:
            print(f"  前向传播失败: {e}")
            break

        advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        check_array("  advantages_norm", advantages_norm.detach().cpu().numpy())

        ratio = torch.exp(log_prob - old_log_prob)
        check_array("  ratio", ratio.detach().cpu().numpy())

        policy_loss_1 = advantages_norm * ratio
        policy_loss_2 = advantages_norm * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        values_pred_clipped = old_values + torch.clamp(values - old_values, -0.2, 0.2)
        value_loss_1 = (values - returns_t) ** 2
        value_loss_2 = (values_pred_clipped - returns_t) ** 2
        value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()

        entropy_loss = -entropy.mean()

        loss = policy_loss + 0.01 * entropy_loss + 0.5 * value_loss

        print(f"\n  损失值:")
        print(f"    policy_loss  = {policy_loss.item():.6f}")
        print(f"    value_loss   = {value_loss.item():.6f}")
        print(f"    entropy_loss = {entropy_loss.item():.6f}")
        print(f"    total_loss   = {loss.item():.6f}")

        print(f"\n  反向传播:")
        model.policy.optimizer.zero_grad()
        try:
            loss.backward()
        except Exception as e:
            print(f"  backward 失败: {e}")
            break

        total_grad_norm = 0.0
        max_grad_norm = 0.0
        max_grad_param = ""
        nan_grad_params = []
        for name, param in model.policy.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm ** 2
                if torch.isnan(param.grad).any():
                    nan_grad_params.append(name)
                if grad_norm > max_grad_norm:
                    max_grad_norm = grad_norm
                    max_grad_param = name

        total_grad_norm = total_grad_norm ** 0.5
        print(f"    总梯度范数 (clipping前): {total_grad_norm:.6f}")
        print(f"    最大梯度参数: {max_grad_param} (norm={max_grad_norm:.6f})")
        if nan_grad_params:
            print(f"    含NaN梯度的参数: {nan_grad_params}")

        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
        total_grad_norm_after = 0.0
        for _, param in model.policy.named_parameters():
            if param.grad is not None:
                total_grad_norm_after += param.grad.data.norm(2).item() ** 2
        total_grad_norm_after = total_grad_norm_after ** 0.5
        print(f"    总梯度范数 (clipping后): {total_grad_norm_after:.6f}")

        model.policy.optimizer.step()

        nan_params_after = []
        for name, param in model.policy.named_parameters():
            if torch.isnan(param.data).any():
                nan_params_after.append(name)
        if nan_params_after:
            print(f"\n  更新后含NaN的参数: {nan_params_after}")
        else:
            print(f"\n  更新后所有参数正常")

        if batch_idx >= 1:
            break

    print("\n" + "=" * 90)
    print("  步骤 3: 检查 log_prob 数值")
    print("=" * 90)
    print(f"  动作维度: {model.action_space.shape[0]}")
    log_std = model.policy.log_std.data.cpu().numpy()
    check_array("log_std", log_std)
    print(f"  对应 std 范围: [{np.exp(log_std.min()):.6f}, {np.exp(log_std.max()):.6f}]")

    n_action_dims = model.action_space.shape[0]
    expected_logprob = -0.5 * n_action_dims * np.log(2 * np.pi) - n_action_dims * log_std.mean()
    print(f"  预期 log_prob 量级: ~{expected_logprob:.1f}")
    print(f"  (1395维高斯的 log_prob 会是一个很大的负数)")

    torch.autograd.set_detect_anomaly(False)
    print("\n  诊断完成!")


if __name__ == "__main__":
    main()
