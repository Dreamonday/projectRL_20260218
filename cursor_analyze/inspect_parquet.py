import pandas as pd
import sys

file_path = "/data/projectRL_20260218/data/predict_and_evaluate/timexer_v0.45_20260214124756_20260214123004_processed_data_20260214_v0.01_20260214163210/预测结果_完整报告.parquet"

try:
    df = pd.read_parquet(file_path)
    print("Columns:")
    for col in df.columns:
        print(f"  {col}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
except Exception as e:
    print(f"Error reading file: {e}")
