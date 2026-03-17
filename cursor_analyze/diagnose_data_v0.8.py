import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path

# 设置路径
DATASET_DIR = "/data/projectRL_20260218/data/processed_data/rl_dataset_202603101235"

def diagnose_dataset():
    print(f"Loading dataset from {DATASET_DIR}...")
    
    try:
        data = np.load(f"{DATASET_DIR}/train_dataset.npz")
        with open(f"{DATASET_DIR}/metadata.json", "r") as f:
            metadata = json.load(f)
            
        dates = data["dates"]
        actual_close = data["actual_close"]
        valid_mask = data["valid_mask"]
        
        print(f"Total dates: {len(dates)}")
        print(f"Total stocks: {len(metadata['stock_ids'])}")
        
        # 1. 检查全市场休市/无数据的情况
        print("\n--- Checking for Market-wide Missing Data ---")
        missing_days = []
        for t, date in enumerate(dates):
            # 检查当日是否有任何一只股票有有效价格
            prices = actual_close[t]
            valid_prices = ~np.isnan(prices) & (prices > 0)
            n_valid = valid_prices.sum()
            
            # 检查 valid_mask
            mask = valid_mask[t]
            n_masked = mask.sum()
            
            if n_valid == 0:
                missing_days.append({
                    "index": t,
                    "date": date,
                    "reason": "No valid prices",
                    "n_valid_prices": n_valid,
                    "n_valid_mask": n_masked
                })
            elif n_valid < 10: # 极少股票有数据
                missing_days.append({
                    "index": t,
                    "date": date,
                    "reason": "Few valid prices",
                    "n_valid_prices": n_valid,
                    "n_valid_mask": n_masked
                })
                
        if missing_days:
            print(f"Found {len(missing_days)} days with missing/insufficient data:")
            df_missing = pd.DataFrame(missing_days)
            print(df_missing)
            
            # 重点检查 20210405
            target = "2021-04-05"
            if target in df_missing['date'].values:
                print(f"\nConfirmed: {target} is in the missing list.")
        else:
            print("No completely missing days found.")

        # 2. 检查价格数据的稀疏度
        print("\n--- Data Sparsity Stats ---")
        avg_valid_per_day = np.mean([ (~np.isnan(actual_close[t]) & (actual_close[t] > 0)).sum() for t in range(len(dates)) ])
        print(f"Average valid stocks per day: {avg_valid_per_day:.1f}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    diagnose_dataset()
