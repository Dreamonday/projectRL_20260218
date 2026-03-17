import numpy as np
import json
import sys

DATASET_DIR = "/data/projectRL_20260218/data/processed_data/rl_dataset_202603101235"

def diagnose_specific_date():
    print(f"Loading dataset...")
    data = np.load(f"{DATASET_DIR}/train_dataset.npz")
    with open(f"{DATASET_DIR}/metadata.json", "r") as f:
        metadata = json.load(f)
        
    dates = data["dates"]
    actual_close = data["actual_close"]
    valid_mask = data["valid_mask"]
    stock_codes = np.array(metadata["stock_codes"])

    target_date = "2021-04-05"
    if target_date not in dates:
        print(f"Date {target_date} NOT found in dataset.")
        # Find closest date
        return

    idx = np.where(dates == target_date)[0][0]
    print(f"\n--- Analysis for {target_date} (Index {idx}) ---")
    
    # 1. Check Prices
    prices = actual_close[idx]
    valid_prices = ~np.isnan(prices) & (prices > 0)
    n_valid_prices = valid_prices.sum()
    print(f"Valid Prices Count: {n_valid_prices} / {len(stock_codes)}")
    
    # 2. Check Valid Mask (Tradable)
    mask = valid_mask[idx]
    n_tradable = mask.sum()
    print(f"Tradable Stocks Count (Valid Mask): {n_tradable}")
    
    # 3. Intersection
    valid_and_tradable = valid_prices & mask
    n_both = valid_and_tradable.sum()
    print(f"Both Valid Price & Tradable: {n_both}")
    
    if n_valid_prices == 0:
        print("CRITICAL: No valid prices for this date. This explains why holdings are 0.")
        # Check previous day
        if idx > 0:
            prev_date = dates[idx-1]
            prev_prices = actual_close[idx-1]
            print(f"Previous Date ({prev_date}) Valid Prices: {(~np.isnan(prev_prices) & (prev_prices > 0)).sum()}")
    else:
        print("Prices exist. Checking if holdings were filtered out...")
        # If there are valid prices, why did run_detailed_backtest fail to record holdings?
        # Maybe the specific stocks held were suspended?
        pass

if __name__ == "__main__":
    diagnose_specific_date()
