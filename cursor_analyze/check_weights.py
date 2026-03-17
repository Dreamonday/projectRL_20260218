import pandas as pd
import sys

try:
    df_test = pd.read_excel('experiments/rl_ppo_v0.4_20260309190004/results/test_trade_details.xlsx')
    
    # 打印前几行的持仓权重列，看看是否真的很平均
    weight_cols = [c for c in df_test.columns if '权重' in c]
    if weight_cols:
        print("\n--- Test Weights (Top 5 Stocks) ---")
        print(df_test[['日期'] + weight_cols[:5]].head(10).to_string())
        
        # 计算每一行权重的标准差，如果标准差很小，说明确实很平均
        # 注意：这里只取了前50只的权重，如果都接近 1/N，标准差会很小
        # 我们假设全市场的 N 很大，如果它只买了 Top 50 且权重不均，标准差会大
        
        # 简单统计一下第一行非零权重的数量和最大权重
        row0 = df_test.iloc[0]
        weights = []
        for c in weight_cols:
            w = row0[c]
            if pd.notna(w) and w > 0:
                weights.append(w)
        
        print(f"\nRow 0 Active Positions: {len(weights)}")
        if weights:
            print(f"Max Weight: {max(weights):.4f}")
            print(f"Min Weight: {min(weights):.4f}")
            print(f"Avg Weight: {sum(weights)/len(weights):.4f}")
            
except Exception as e:
    print(e)
