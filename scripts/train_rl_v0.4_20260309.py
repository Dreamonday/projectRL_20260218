#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL 交易模型训练与回测主脚本
版本: v0.4
日期: 20260309

v0.4 变更:
  - 强制进行数据降采样 (Resampling)，实现真正的周频交易
  - 配置加载: configs/rl_config_v0.4.yaml
  - 调整训练参数以适应减少的数据量

使用方式:
    python scripts/train_rl_v0.4_20260309.py --config configs/rl_config_v0.4.yaml
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

from src.data.data_loader import load_prediction_data, build_trading_dataset, TradingDataset
from src.envs.trading_env import MultiStockTradingEnv
from src.utils.backtest import (
    compute_buy_and_hold,
    compute_metrics,
    estimate_periods_per_year,
    print_backtest_report,
    plot_backtest,
)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor


class TrainingHistoryCallback(BaseCallback):
    """
    记录训练过程中的各项指标，并在训练结束时保存为 Excel 文件
    """
    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.history = []
        self._start_time = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_rollout_end(self) -> None:
        # 在每次 rollout 结束（即一次迭代结束）时收集数据
        
        # 1. 基础信息
        elapsed = time.time() - self._start_time if self._start_time else 0
        current_step = self.num_timesteps
        
        # 2. 从 logger 获取训练指标
        logger = self.model.logger.name_to_value
        
        # 3. 构造记录字典
        record = {
            "timesteps": current_step,
            "elapsed_time": elapsed,
            "fps": logger.get("time/fps", 0),
            
            # 业务指标
            "ep_rew_mean": logger.get("rollout/ep_rew_mean", float("nan")),
            "ep_len_mean": logger.get("rollout/ep_len_mean", float("nan")),
            
            # 训练损失
            "loss": logger.get("train/loss", float("nan")),
            "value_loss": logger.get("train/value_loss", float("nan")),
            "policy_gradient_loss": logger.get("train/policy_gradient_loss", float("nan")),
            "entropy_loss": logger.get("train/entropy_loss", float("nan")),
            
            # 其他指标
            "approx_kl": logger.get("train/approx_kl", float("nan")),
            "clip_fraction": logger.get("train/clip_fraction", float("nan")),
            "explained_variance": logger.get("train/explained_variance", float("nan")),
            "learning_rate": logger.get("train/learning_rate", float("nan")),
            "std": logger.get("train/std", float("nan")),
        }
        
        # 尝试从 Monitor buffer 获取更实时的 ep_rew_mean (如果 logger 还没更新)
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            import numpy as np
            real_ep_rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            if not np.isnan(real_ep_rew):
                record["ep_rew_mean_realtime"] = real_ep_rew
        
        self.history.append(record)

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        # 保存为 Excel
        if self.history:
            df = pd.DataFrame(self.history)
            try:
                # 确保目录存在
                Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_excel(self.save_path, index=False)
                print(f"\n[TrainingHistory] 训练历史数据已保存至: {self.save_path}")
            except Exception as e:
                print(f"\n[TrainingHistory] 保存 Excel 失败: {e}")


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

        # 尝试获取 Monitor 输出的 rollout/ep_rew_mean
        # 优先从 model.ep_info_buffer 中直接计算，比 logger 更实时
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            ep_rew_mean = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            ep_len_mean = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
        else:
            ep_rew_mean = logger.get("rollout/ep_rew_mean", float("nan"))
            ep_len_mean = logger.get("rollout/ep_len_mean", float("nan"))

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
            ("rollout/ (Mean Reward)", ""),
            ("   ep_rew_mean", f"{ep_rew_mean:.4f}" if not np.isnan(ep_rew_mean) else "---"),
            ("   ep_len_mean", f"{ep_len_mean:.0f}" if not np.isnan(ep_len_mean) else "---"),
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


def resample_dataset(dataset: TradingDataset, rule="W-FRI") -> TradingDataset:
    """
    将 TradingDataset 降采样到指定频率。
    
    Args:
        dataset: 原始数据集
        rule: Pandas resample rule (e.g. 'W-FRI')
        
    Returns:
        降采样后的新数据集
    """
    print(f"  正在执行降采样: {rule} ...")
    
    # 1. 创建日期索引
    # dates 是字符串列表，先转为 datetime
    raw_dates = pd.to_datetime(dataset.dates)
    
    # 创建一个 Series，值为原始索引 (0, 1, 2, ...)
    # 这样 resample 后就能找到对应周期的“最后一个原始索引”
    s = pd.Series(np.arange(len(raw_dates)), index=raw_dates)
    
    # 2. 执行 Resample
    # .last() 取该周期内最后一个有效值
    # 例如：如果一周有 Mon, Tue, Wed，那么 .last() 会取 Wed 的索引
    resampled = s.resample(rule).last()
    
    # 3. 过滤掉 NaN (那些完全没有数据的周期)
    # 并转为整数索引
    valid_indices = resampled.dropna().astype(int).values
    
    if len(valid_indices) == 0:
        raise ValueError("降采样后没有剩余数据！请检查频率设置。")
        
    print(f"  原始数据点: {len(dataset.dates)}")
    print(f"  降采样后: {len(valid_indices)}")
    print(f"  缩减比例: {len(valid_indices) / len(dataset.dates):.2%}")
    
    # 4. 构建新数据集 (切片)
    # 注意：需要切片 TradingDataset 中的所有 numpy 数组
    
    new_dates = dataset.dates[valid_indices]
    
    # 切片各个数组
    # dataset.dates: (n_dates,)
    # dataset.predicted_close: (n_dates, n_stocks)
    # dataset.actual_close: (n_dates, n_stocks)
    # dataset.last_input_close: (n_dates, n_stocks)
    # dataset.has_data: (n_dates, n_stocks)
    
    new_dataset = TradingDataset(
        dates=new_dates,
        stock_ids=dataset.stock_ids,
        stock_codes=dataset.stock_codes,
        stock_names=dataset.stock_names,
        predicted_close=dataset.predicted_close[valid_indices],
        actual_close=dataset.actual_close[valid_indices],
        last_input_close=dataset.last_input_close[valid_indices],
        has_data=dataset.has_data[valid_indices],
        val_avg_error=dataset.val_avg_error,
    )
    
    return new_dataset


def run_detailed_backtest(env, model, dataset, initial_cash, normalize_env=None):
    """
    运行详细回测，计算真实/浮动盈亏，并生成详细报表
    """
    obs, _ = env.reset()
    done = False
    
    # 影子账户状态
    shadow_cash = initial_cash
    # holdings: {stock_code: {'shares': float, 'avg_cost': float}}
    holdings = {}
    realized_pnl_cum = 0.0
    
    # 记录列表
    records = []
    
    # 上一期的状态 (code -> weight, code -> shares)
    prev_weights_map = {}
    prev_shares_map = {}
    
    # 获取股票名称映射: code -> name
    code_to_name = {}
    if hasattr(dataset, 'stock_codes') and hasattr(dataset, 'stock_names'):
        for c, n in zip(dataset.stock_codes, dataset.stock_names):
            code_to_name[c] = n
            
    stock_codes = np.array(dataset.stock_codes)
    
    # 回测循环
    while not done:
        current_step = env.current_step
        date = dataset.dates[current_step]
        
        # 1. 预测动作
        if normalize_env:
            obs_norm = normalize_env.normalize_obs(obs)
            action, _ = model.predict(obs_norm, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
            
        # 2. 执行环境 Step
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. 影子账户计算 (在 Step 之后，利用 info 中的 weights)
        # 注意：env.step 已经更新了 portfolio_value。我们需要反推交易
        # 为了简化，我们直接使用 env 的 weights 来计算“理论持仓”，并更新 avg_cost
        
        # 获取当前所有股票价格 (使用实际收盘价)
        current_prices = dataset.actual_close[current_step]
        valid_price_mask = ~np.isnan(current_prices) & (current_prices > 0)
        
        # 当前总资产 (来自 Env，包含了交易成本扣除)
        current_total_asset = info["portfolio_value"]
        current_weights = env.weights
        
        # 计算理论上的目标持仓价值和份额
        # Target Value_i = Total Asset * Weight_i
        target_values = current_total_asset * current_weights
        
        # 计算交易产生的 Realized PnL
        # 我们假设交易发生在收盘价 (Env也是这么假设的)
        
        step_realized_pnl = 0.0
        
        # 临时记录本期的持仓 (供下一期使用)
        current_shares_map = {}
        current_weights_map = {}
        
        # 遍历所有股票 (只处理有效价格的)
        for i in range(len(stock_codes)):
            if not valid_price_mask[i]:
                continue
                
            code = stock_codes[i]
            price = current_prices[i]
            target_val = target_values[i]
            target_share = target_val / price if price > 0 else 0
            
            # 记录本期状态
            if target_share > 1e-6:
                current_shares_map[code] = target_share
                current_weights_map[code] = current_weights[i]
            
            # 获取旧持仓
            old_pos = holdings.get(code, {'shares': 0.0, 'avg_cost': 0.0})
            old_share = old_pos['shares']
            
            diff_share = target_share - old_share
            
            if abs(diff_share) < 1e-6:
                continue
                
            if diff_share > 0: # 买入
                # 更新平均成本: (OldCost * OldShare + CurrentPrice * BuyShare) / NewShare
                cost_basis = old_pos['avg_cost'] * old_share + price * diff_share
                new_avg_cost = cost_basis / target_share
                holdings[code] = {'shares': target_share, 'avg_cost': new_avg_cost}
                
            else: # 卖出
                # 计算已实现盈亏: (Price - AvgCost) * SellShare
                sell_share = abs(diff_share)
                # 确保不超过持有量 (浮点误差保护)
                sell_share = min(sell_share, old_share)
                
                pnl = (price - old_pos['avg_cost']) * sell_share
                step_realized_pnl += pnl
                
                # 卖出不改变剩余持仓的单位成本
                remaining_share = old_share - sell_share
                if remaining_share < 1e-6:
                    if code in holdings:
                        del holdings[code]
                else:
                    holdings[code]['shares'] = remaining_share
        
        realized_pnl_cum += step_realized_pnl
        
        # 4. 计算当前的浮动盈亏 (Unrealized PnL)
        unrealized_pnl = 0.0
        pos_details_flat = {} # 用于Excel的宽表格式
        
        # 排序持仓，为了在Excel中展示权重最大的前N个
        sorted_holdings = []
        
        for code, pos in holdings.items():
            # 找到当前价格
            # 这是一个笨办法，但为了准确性
            try:
                idx = np.where(stock_codes == code)[0][0]
                curr_price = current_prices[idx]
            except:
                curr_price = pos['avg_cost'] # Fallback
            
            market_val = pos['shares'] * curr_price
            u_pnl = (curr_price - pos['avg_cost']) * pos['shares']
            unrealized_pnl += u_pnl
            
            stock_name = code_to_name.get(code, code)
            weight = market_val / current_total_asset if current_total_asset > 0 else 0
            
            # 计算相对于上期的变化
            prev_weight = prev_weights_map.get(code, 0.0)
            prev_share = prev_shares_map.get(code, 0.0)
            
            delta_weight = weight - prev_weight
            delta_share = pos['shares'] - prev_share
            delta_amount = delta_share * curr_price # 正数买入金额，负数卖出金额
            
            sorted_holdings.append({
                'name': stock_name,
                'weight': weight,
                'market_val': market_val,
                'pnl': u_pnl,
                'avg_cost': pos['avg_cost'],
                'curr_price': curr_price,
                'delta_weight': delta_weight,
                'delta_amount': delta_amount
            })

        # 4.5 补充已清仓的股票 (v0.4.4 feature)
        # 遍历上一期持有的股票，如果本期不在 holdings 中，说明被清仓了
        for code, prev_share in prev_shares_map.items():
            if prev_share > 1e-6 and (code not in holdings or holdings[code]['shares'] < 1e-6):
                # 该股票已被清仓
                try:
                    idx = np.where(stock_codes == code)[0][0]
                    curr_price = current_prices[idx]
                except:
                    curr_price = 0.0
                
                prev_weight = prev_weights_map.get(code, 0.0)
                
                # 变化量
                delta_weight = 0.0 - prev_weight
                delta_amount = (0.0 - prev_share) * curr_price # 卖出金额 (负数)
                
                stock_name = code_to_name.get(code, code)
                
                sorted_holdings.append({
                    'name': f"{stock_name} (清仓)", # 在名称中标记
                    'weight': 0.0,
                    'market_val': 0.0,
                    'pnl': 0.0, # 已实现盈亏已计入总账，此处浮动盈亏为0
                    'avg_cost': 0.0,
                    'curr_price': curr_price,
                    'delta_weight': delta_weight,
                    'delta_amount': delta_amount
                })
            
        # 按权重降序
        sorted_holdings.sort(key=lambda x: x['weight'], reverse=True)
        
        # 填充 Excel 列 (前 50 只)
        for rank, h in enumerate(sorted_holdings[:50]): # 限制列数防止爆炸
            prefix = f"持仓_{rank+1}"
            pos_details_flat[f"{prefix}_名称"] = h['name']
            pos_details_flat[f"{prefix}_权重"] = h['weight']
            pos_details_flat[f"{prefix}_权重变化"] = h['delta_weight']
            pos_details_flat[f"{prefix}_仓位变化金额"] = h['delta_amount']
            # pos_details_flat[f"{prefix}_盈亏"] = h['pnl'] # 可选
        
        # 5. 记录
        total_pnl = realized_pnl_cum + unrealized_pnl
        # 校验：Total PnL 应该接近 (current_total_asset - initial_cash)
        # 差异主要来自交易成本 (Env扣除了，但RealizedPnL没扣除交易成本)
        # 为了让报表自洽，我们将 交易成本 也单独列出
        # 实际上：Asset - Init = Realized + Unrealized - TransactionCosts
        
        record = {
            "日期": date,
            "总资产": current_total_asset,
            "总盈亏(Asset-Init)": current_total_asset - initial_cash,
            "已实现盈亏(累计)": realized_pnl_cum,
            "浮动盈亏(当前)": unrealized_pnl,
            "交易成本(当日)": info['trade_cost'],
            "换手率": info['turnover'],
            "现金比例": info['cash_ratio'],
            "持仓股票数": len(holdings)
        }
        record.update(pos_details_flat)
        records.append(record)
        
        # 更新上一期状态
        prev_weights_map = current_weights_map
        prev_shares_map = current_shares_map
        
        obs = next_obs
        done = terminated or truncated

    return pd.DataFrame(records), info # 返回最后的 info


def main():
    parser = argparse.ArgumentParser(description="RL 交易模型训练 v0.4 (Weekly)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "configs" / "rl_config_v0.4.yaml"),
        help="配置文件路径",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    print("=" * 70)
    print(f"  RL 交易模型训练 v0.4 (周频版) — {run_timestamp}")
    print("=" * 70)
    print_gpu_info()

    # ======== 1. 加载数据 ========
    print("\n[Step 1] 加载 TimeXer 预测数据 (原始日频)...")
    data_cfg = config["data"]
    stock_data = load_prediction_data(
        data_dir=data_cfg["prediction_dir"],
        market_filter=data_cfg["market_filter"],
        max_stocks=data_cfg.get("max_stocks"),
        use_train_data=data_cfg["use_train_data"],
        use_val_data=data_cfg["use_val_data"],
        min_data_points=data_cfg["min_data_points"],
    )

    print(f"\n[Step 2] 构建初始交易数据集...")
    full_dataset_daily = build_trading_dataset(stock_data)

    # ======== 1.5 降采样 (v0.4 新增) ========
    force_resample = data_cfg.get("force_resample", {})
    if force_resample.get("enabled", False):
        print(f"\n[Step 2.5] 执行强制降采样 (Resampling)...")
        rule = force_resample.get("rule", "W-FRI")
        full_dataset = resample_dataset(full_dataset_daily, rule=rule)
    else:
        print("\n[Step 2.5] 未启用降采样，使用原始频率 (Daily)")
        full_dataset = full_dataset_daily

    # ======== 2. 拆分训练/测试 ========
    split_cfg = config["split"]
    train_end = split_cfg["rl_train_end_date"]
    print(f"\n[Step 3] 按日期拆分: 训练 <= {train_end} / 测试 > {train_end}")

    train_dataset, test_dataset = full_dataset.split_by_date(train_end)

    if train_dataset is None or train_dataset.n_dates < 10:
        print("错误: 训练集日期太少，请调整 rl_train_end_date")
        sys.exit(1)

    train_ppy = estimate_periods_per_year(train_dataset.dates)
    if test_dataset:
        test_ppy = estimate_periods_per_year(test_dataset.dates)
    
    print(f"  训练集频率估算 (Periods per Year): {train_ppy:.1f}")

    # ======== 3. 创建环境 ========
    print(f"\n[Step 4] 创建训练环境...")
    rl_cfg = config["rl"]
    device = rl_cfg.get("device", "auto")
    if device == "cuda" and not torch.cuda.is_available():
        print("  警告: 配置为 cuda 但未检测到 GPU，回退到 cpu")
        device = "cpu"

    train_vec_env = DummyVecEnv([make_env(train_dataset, config, seed=42)])
    # 使用 Monitor 包装环境以记录 ep_rew_mean
    # 注意：DummyVecEnv 里面的环境通常没有 Monitor，我们需要在 make_env 内部包，或者在这里包
    # 但 VecEnv 已经是一层 Wrapper 了。
    # 正确做法：make_env 返回的 env 应该被 Monitor 包裹。
    
    # 由于 make_env 已经定义好了，我们重新修改 make_env 逻辑比较麻烦
    # 不如直接在这里重新定义 train_vec_env 的构造
    
    def make_monitored_env(dataset, config, seed=0):
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
            # 包装 Monitor
            return Monitor(env)
        return _init

    train_vec_env = DummyVecEnv([make_monitored_env(train_dataset, config, seed=42)])
    
    if rl_cfg.get("normalize_observations", True) or rl_cfg.get("normalize_rewards", True):
        train_vec_env = VecNormalize(
            train_vec_env,
            norm_obs=rl_cfg.get("normalize_observations", True),
            norm_reward=rl_cfg.get("normalize_rewards", True),
            clip_obs=10.0,
            clip_reward=10.0,
        )

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
    
    print(f"  Batch Size: {batch_size}, N Steps: {n_steps}")
    
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

    total_ts = rl_cfg["total_timesteps"]
    # ======== 5. 保存模型 ========
    output_dir = project_root / "experiments" / f"rl_ppo_v0.4_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(exist_ok=True)

    print(f"\n[Step 6] 开始训练 ({total_ts} timesteps)...")
    
    # 组合 Callbacks
    log_callback = TrainingLogCallback(total_timesteps=total_ts, n_steps=n_steps)
    history_callback = TrainingHistoryCallback(save_path=str(results_dir / "training_metrics.xlsx"))
    
    callbacks = CallbackList([log_callback, history_callback])
    
    model.learn(total_timesteps=total_ts, callback=callbacks, progress_bar=False)
    print("训练完成!")

    model.save(str(model_dir / "ppo_trading"))
    if isinstance(train_vec_env, VecNormalize):
        train_vec_env.save(str(model_dir / "vec_normalize.pkl"))
    
    with open(configs_dir / "rl_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    # ======== 6. 详细回测 ========
    print(f"\n[Step 7] 执行详细回测 (包含 PnL 分解)...")
    
    # 辅助函数：运行回测并保存结果
    def process_backtest(dataset, ppy, prefix):
        if not dataset or dataset.n_dates < 5:
            return None
            
        print(f"  正在回测 {prefix} 集 (Size: {dataset.n_dates})...")
        env = MultiStockTradingEnv(
            dataset=dataset,
            initial_cash=config["trading"]["initial_cash"],
            transaction_cost=config["trading"]["transaction_cost"],
            max_weight_per_stock=config["trading"]["max_weight_per_stock"],
            reward_type=config["reward"]["type"],
            turnover_penalty=config["reward"]["turnover_penalty"],
        )
        
        # 加载 Normalizer
        norm_env = None
        if isinstance(train_vec_env, VecNormalize):
            _dummy = DummyVecEnv([lambda: env])
            norm_env = VecNormalize.load(str(model_dir / "vec_normalize.pkl"), _dummy)
            norm_env.training = False
            norm_env.norm_reward = False
        
        # 运行详细回测
        df_details, final_info = run_detailed_backtest(
            env=env,
            model=model,
            dataset=dataset,
            initial_cash=config["trading"]["initial_cash"],
            normalize_env=norm_env
        )
        
        # 保存详细 Excel
        excel_path = results_dir / f"{prefix}_trade_details.xlsx"
        df_details.to_excel(excel_path, index=False)
        print(f"  {prefix} 交易明细已保存: {excel_path}")
        
        # 计算并保存 Metrics & Plot
        dates = df_details["日期"].values
        values = df_details["总资产"].values
        
        # 计算 Buy & Hold 基准
        bh_values = compute_buy_and_hold(env)
        min_len = min(len(values), len(bh_values))
        
        metrics = compute_metrics(values[:min_len], dates[:min_len], periods_per_year=ppy)
        bh_metrics = compute_metrics(bh_values[:min_len], dates[:min_len], periods_per_year=ppy)
        
        print_backtest_report(metrics, bh_metrics, label=f"RL Agent ({prefix})", benchmark_label="等权买入持有")
        
        plot_backtest(
            dates[:min_len], values[:min_len], bh_values[:min_len],
            save_path=str(results_dir / f"{prefix}_backtest.png"),
            title=f"{prefix.capitalize()} Period Backtest (Weekly)",
        )
        return metrics

    # 训练集回测
    train_metrics = process_backtest(train_dataset, train_ppy, "train")
    
    # 测试集回测
    test_metrics = None
    if test_dataset:
        test_metrics = process_backtest(test_dataset, test_ppy, "test")

    print(f"\n{'=' * 70}")
    print(f"  全部完成! 结果保存在: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
