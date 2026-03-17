
import pandas as pd
import sys

try:
    df = pd.read_excel('experiments/rl_ppo_v0.3_20260309170731/results/test_trade_details.xlsx')
    print(df['日期'].head(20).to_string())
    print("\n--- Train Data ---")
    df_train = pd.read_excel('experiments/rl_ppo_v0.3_20260309170731/results/train_trade_details.xlsx')
    print(df_train['日期'].head(20).to_string())
except Exception as e:
    print(e)
