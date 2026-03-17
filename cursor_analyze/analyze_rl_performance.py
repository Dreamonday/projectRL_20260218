import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Path to the results file
results_file = Path("/data/projectRL_20260218/experiments/rl_ppo_v0.3_20260309170731/results/test_trade_details.xlsx")

if not results_file.exists():
    print(f"Error: File not found: {results_file}")
    sys.exit(1)

print(f"Loading data from: {results_file}")
df = pd.read_excel(results_file)

# Basic stats
n_days = len(df)
initial_capital = 10000000.0  # Hardcoded from config
final_capital = df["总资产"].iloc[-1]
total_return = (final_capital - initial_capital) / initial_capital

# Transaction Costs Analysis
# Note: "交易成本(当日)" in the excel is the cost ratio (cost / portfolio_value).
# We need to multiply by portfolio value to get dollar amount.
daily_cost_dollars = df["交易成本(当日)"] * df["总资产"]
total_tx_cost = daily_cost_dollars.sum()
tx_cost_impact = total_tx_cost / initial_capital
# If we add back transaction costs
hypothetical_final_capital = final_capital + total_tx_cost
hypothetical_return = (hypothetical_final_capital - initial_capital) / initial_capital

# Exposure Analysis
avg_cash_ratio = df["现金比例"].mean()
avg_exposure = 1.0 - avg_cash_ratio
# Calculate correlation between exposure and daily return (market timing ability?)
# Need market return proxy (using equal weight benchmark implied by user description or previous results)
# Here we just check exposure levels.

# Turnover Analysis
avg_turnover = df["换手率"].mean()

# Concentration Analysis
avg_positions = df["持仓股票数"].mean()
max_positions = df["持仓股票数"].max()
min_positions = df["持仓股票数"].min()

# Weight Concentration Analysis
# Check the top 1 weight to see if the model has strong convictions
avg_top1_weight = df["持仓_1_权重"].mean()
# Theoretical equal weight
theoretical_eq_weight = 1.0 / max(1, avg_positions)

# PnL Analysis
realized_pnl = df["已实现盈亏(累计)"].iloc[-1]
unrealized_pnl = df["浮动盈亏(当前)"].iloc[-1]
total_pnl = realized_pnl + unrealized_pnl

# Output Report
print("-" * 50)
print("RL Model Performance Analysis Report")
print("-" * 50)
print(f"Total Trading Days: {n_days}")
print(f"Final Return: {total_return:.2%}")
print(f"\n[1. Transaction Costs Impact]")
print(f"  Total Transaction Costs: ${total_tx_cost:,.2f}")
print(f"  Cost Impact on Return: -{tx_cost_impact:.2%}")
print(f"  Return WITHOUT Costs: {hypothetical_return:.2%}")
print(f"  (Compared to Benchmark: ~5.77%)")

if hypothetical_return > 0.0577:
    print("  -> Conclusion: The model OUTPERFORMS benchmark before fees.")
    print("     The underperformance is purely due to high trading costs.")
else:
    print("  -> Conclusion: Even without fees, the model UNDERPERFORMS benchmark.")
    print("     The issue is likely stock selection or timing.")

print(f"\n[2. Exposure Analysis (Market Timing)]")
print(f"  Average Cash Ratio: {avg_cash_ratio:.2%}")
print(f"  Average Market Exposure: {avg_exposure:.2%}")
if avg_exposure < 0.95:
    print("  -> Observation: The model held significant cash.")
    print("     In a bull market (Benchmark +5.77%), holding cash drags performance.")
    # Calculate drag: roughly avg_cash * benchmark_return
    cash_drag = avg_cash_ratio * 0.0577
    print(f"     Estimated Cash Drag: -{cash_drag:.2%}")

print(f"\n[3. Trading Activity]")
print(f"  Average Daily Turnover: {avg_turnover:.2%}")
print(f"  (Implies full portfolio turnover every {1/avg_turnover:.1f} days)")

print(f"\n[4. Portfolio Concentration]")
print(f"  Average Positions Held: {avg_positions:.1f}")
print(f"  Max Positions: {max_positions}")
print(f"  Min Positions: {min_positions}")
print(f"  Average Top-1 Holding Weight: {avg_top1_weight:.4%}")
print(f"  Theoretical Equal Weight: {theoretical_eq_weight:.4%}")
if avg_top1_weight < theoretical_eq_weight * 2:
    print("  -> Observation: The model is acting like a 'Closet Indexer'.")
    print("     It holds nearly equal weights, very similar to the benchmark.")
    print("     This explains why its performance (5.37%) is so close to benchmark (5.77%).")
    print("     The difference is mostly transaction costs.")
elif avg_top1_weight > 0.05:
    print("  -> Observation: Highly concentrated top picks.")
else:
    print("  -> Observation: Some active weight allocation.")

print(f"\n[5. PnL Breakdown]")
print(f"  Realized PnL: ${realized_pnl:,.2f}")
print(f"  Unrealized PnL: ${unrealized_pnl:,.2f}")
if realized_pnl < 0 and unrealized_pnl > 0:
    print("  -> Pattern: The model takes losses quickly but lets profits run (Good).")
elif realized_pnl > 0 and unrealized_pnl < 0:
    print("  -> Pattern: The model takes profits quickly but holds onto losers (Bad - Disposition Effect).")

print("-" * 50)
