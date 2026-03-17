import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = "/data/projectRL_20260218/experiments/rl_ppo_v0.8_20260310175146/results"

def analyze_training(path):
    print(f"\n--- Training Metrics ({path}) ---")
    try:
        df = pd.read_excel(f"{BASE_DIR}/{path}")
        print(df.tail(5)[['timesteps', 'ep_rew_mean', 'loss', 'explained_variance']])
    except Exception as e:
        print(f"Error reading training metrics: {e}")

def analyze_trades(path, label):
    print(f"\n--- {label} Trade Details ({path}) ---")
    try:
        df = pd.read_excel(f"{BASE_DIR}/{path}")
        if df.empty:
            print("Empty dataframe")
            return

        # Basic Stats
        n_days = len(df)
        final_pnl = df.iloc[-1]['总盈亏(Asset-Init)']
        final_asset = df.iloc[-1]['总资产']
        
        # Cash & Holdings
        avg_cash_ratio = df['现金比例'].mean()
        max_cash_ratio = df['现金比例'].max()
        min_cash_ratio = df['现金比例'].min()
        
        avg_holdings = df['持仓股票数(有效)'].mean()
        max_holdings = df['持仓股票数(有效)'].max()
        
        # Turnover
        avg_turnover = df['换手率'].mean()
        
        # Returns
        df['daily_return'] = df['总资产'].pct_change().fillna(0)
        sharpe = (df['daily_return'].mean() / df['daily_return'].std()) * np.sqrt(252) if df['daily_return'].std() > 0 else 0
        
        # Max Drawdown
        cum_ret = (1 + df['daily_return']).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_dd = drawdown.min()

        print(f"Days: {n_days}")
        print(f"Final PnL: {final_pnl:,.2f}")
        print(f"Final Asset: {final_asset:,.2f}")
        print(f"Cash Ratio: Avg={avg_cash_ratio:.1%}, Max={max_cash_ratio:.1%}, Min={min_cash_ratio:.1%}")
        print(f"Holdings Count: Avg={avg_holdings:.1f}, Max={max_holdings}")
        print(f"Turnover: Avg={avg_turnover:.1%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.1%}")

    except Exception as e:
        print(f"Error reading trade details: {e}")

if __name__ == "__main__":
    analyze_training("training_metrics.xlsx")
    analyze_trades("train_trade_details.xlsx", "TRAIN")
    analyze_trades("test_trade_details.xlsx", "TEST")
