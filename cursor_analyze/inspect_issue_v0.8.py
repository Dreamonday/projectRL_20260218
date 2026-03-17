import pandas as pd

file_path = "/data/projectRL_20260218/experiments/rl_ppo_v0.8_20260310175146/results/train_trade_details.xlsx"

try:
    df = pd.read_excel(file_path)
    # 找到 20210405 这一行 (假设 '日期' 列存在且格式匹配，或者是 2370 行附近)
    # 先尝试按索引访问，因为用户提到了 2370 行 (注意 pandas 是 0-indexed，Excel 是 1-indexed)
    
    # 为了保险，我们先打印 2365 到 2375 行的数据，重点关注 20210405
    target_date = 20210405
    
    # 查找特定日期的行
    row = df[df['日期'].astype(str) == '2021-04-05']
    if row.empty:
         row = df[df['日期'].astype(str).str.contains('20210405')]
    if row.empty:
         # 按索引
         row = df.iloc[[2368]]

    if not row.empty:
        print("--- Row for 20210405 ---")
        print(row.iloc[0])
        
        # 检查上下文：前一天和后一天
        idx = row.index[0]
        if idx > 0:
            print("\n--- Previous Day ---")
            print(df.iloc[idx-1][['日期', '总资产', '现金比例', '持仓股票数(有效)']])
        if idx < len(df) - 1:
            print("\n--- Next Day ---")
            print(df.iloc[idx+1][['日期', '总资产', '现金比例', '持仓股票数(有效)']])
            
    else:
        print(f"Date {target_date} not found. Showing lines 2368-2372:")
        print(df.iloc[2368:2373][['日期', '总资产', '现金比例', '持仓股票数(有效)']])

except Exception as e:
    print(f"Error: {e}")
