import numpy as np
import json
import sys

DATASET_DIR = "/data/projectRL_20260218/data/processed_data/rl_dataset_202603101235"

def verify_data_integrity():
    print(f"Loading dataset from {DATASET_DIR}...")
    try:
        data = np.load(f"{DATASET_DIR}/train_dataset.npz")
        with open(f"{DATASET_DIR}/metadata.json", "r") as f:
            metadata = json.load(f)
            
        dates = data["dates"]
        actual_close = data["actual_close"]
        valid_mask = data["valid_mask"]
        stock_codes = np.array(metadata["stock_codes"])
        
        print(f"Checking {len(dates)} dates x {len(stock_codes)} stocks...")
        
        # 1. 查找 valid=True 但 price=NaN 的异常点
        # valid_mask is boolean, actual_close is float
        
        # 构造异常 Mask
        price_invalid = np.isnan(actual_close) | (actual_close <= 1e-6)
        anomaly_mask = valid_mask & price_invalid
        
        n_anomalies = anomaly_mask.sum()
        
        if n_anomalies > 0:
            print(f"\n[CRITICAL] Found {n_anomalies} anomalies where valid_mask=True but price is invalid!")
            
            # 找出受影响的日期
            affected_date_indices = np.where(anomaly_mask.any(axis=1))[0]
            print(f"Affected dates count: {len(affected_date_indices)}")
            
            # 详细列出前几个异常日期
            print("\nTop 5 Affected Dates:")
            for idx in affected_date_indices[:5]:
                date = dates[idx]
                n_bad_stocks = anomaly_mask[idx].sum()
                print(f"  Date: {date} (Index {idx}) - {n_bad_stocks} stocks with valid=True but no price")
                
            # 检查特定的嫌疑日期
            target_dates = ['2015-04-06', '2017-05-30', '2021-04-05']
            print("\nChecking Specific Suspect Dates:")
            for target in target_dates:
                if target in dates:
                    idx = np.where(dates == target)[0][0]
                    n_bad = anomaly_mask[idx].sum()
                    print(f"  {target}: Found {n_bad} anomalies (Total Stocks: {len(stock_codes)})")
                    if n_bad > 0:
                        # 打印几个具体的股票代码
                        bad_stock_indices = np.where(anomaly_mask[idx])[0]
                        print(f"    Sample stocks: {stock_codes[bad_stock_indices[:5]]}")
                else:
                    print(f"  {target}: Not found in dataset")
        else:
            print("\n[OK] No anomalies found. valid_mask aligns perfectly with valid prices.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_data_integrity()
