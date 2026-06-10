"""
05 · Real-world data: a company's money  💸
============================================
Everything so far used made-up numbers. Let's draw a REAL company's income
statement — revenue flowing down through costs to net profit — straight from
Yahoo Finance.

This needs one extra package:
    pip install yfinance

Run it:
    python 05_real_company_data.py            # defaults to MSFT
    python 05_real_company_data.py AAPL       # or any ticker

Writes  income_<TICKER>.png

The one idea worth remembering: a Sankey link can't be negative, so we plot the
*magnitude* and put the real (signed) number in the label. We also derive the
"cost" flows as residuals (Revenue − Gross = COGS, etc.) so the waterfall always
balances. That's the whole trick the pros use.
"""
import sys

import pandas as pd
import yfinance as yf
from gradient_sankey import SankeyRaceMultiLayerParallel as Sankey

ticker = (sys.argv[1] if len(sys.argv) > 1 else "MSFT").upper()

# 1) Pull four clean lines from the latest quarterly income statement.
stmt = yf.Ticker(ticker).quarterly_income_stmt
need = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income"]
latest = stmt[stmt.columns[0]]                      # most recent quarter
rev, gross, op, net = (float(latest[r]) / 1e9 for r in need)   # in $ billions

# 2) Derive the "leak" at each stage as the residual, so inflow == outflow.
cogs, opex, tax = rev - gross, gross - op, op - net

# 3) Build the waterfall: Revenue → {Gross, COGS} → {Operating, OpEx} → {Net, Tax}.
flows = {
    ("Revenue", "Gross Profit"): gross, ("Revenue", "Cost of Sales"): cogs,
    ("Gross Profit", "Operating Income"): op, ("Gross Profit", "Operating Expenses"): opex,
    ("Operating Income", "Net Income"): net, ("Operating Income", "Tax & Other"): tax,
}
df = pd.DataFrame([{"t": 0, "s": s, "d": d, "v": abs(v)} for (s, d), v in flows.items()])
layers = [["Revenue"], ["Gross Profit", "Cost of Sales"],
          ["Operating Income", "Operating Expenses"], ["Net Income", "Tax & Other"]]

# Colour each track: the green "profit spine" vs the magenta "leaks".
KEPT, LEAK = "#33E08A", "#FF2E97"
colors = {n: (KEPT if i == 0 else LEAK) for layer in layers for i, n in enumerate(layer)}
colors["Revenue"] = "#4CC9F0"


def money(v):                      # 4.78 -> "4.8" ; -4.78 -> "(4.8)" (accounting style)
    s = f"{abs(v):.1f}" if abs(v) < 10 else f"{abs(v):.0f}"
    return f"({s})" if v < 0 else s


labels = {n: money(v) for n, v in [
    ("Revenue", rev), ("Gross Profit", gross), ("Cost of Sales", cogs),
    ("Operating Income", op), ("Operating Expenses", opex),
    ("Net Income", net), ("Tax & Other", tax)]}

Sankey.from_dataframe(
    df, layers, time_col="t", source_col="s", target_col="d", value_col="v",
    node_colors=colors,
).save_frame(
    f"income_{ticker}.png",
    title=f"{ticker} — income statement ($B, latest quarter)",
    theme="dark", link_glow=1, stacked_mode=True, ranking_mode=False,
    node_value_labels=labels,           # show the real signed numbers
    yaxis_node="Revenue", yaxis_suffix="B",   # a dynamic $ axis on Revenue
    figsize=(13, 7), dpi=140, padding=1.8,
)
print(f"Done! Open income_{ticker}.png")
print("Want the full animated version with 18 years of history + a stock overlay?")
print("-> see advanced/nvidia_reel.py. Next tutorial -> 06_background_music.py")
