"""
05 - Real-world data: a company's income statement
==================================================
Everything so far used numbers we made up. Now we'll draw a REAL company's
income statement - revenue flowing down through costs to net profit - pulled
live from Yahoo Finance. This lesson teaches the two ideas that make financial
flows actually work, which trip up almost everyone the first time.

Install the one extra dependency:
    pip install yfinance


IDEA 1 - A WATERFALL MUST BALANCE, SO DERIVE THE COSTS AS RESIDUALS
------------------------------------------------------------------
A Sankey is conservative: whatever enters a node must leave it. An income
statement naturally has this shape if you think of it as "what you keep" vs
"what leaks out" at each stage:

    Revenue ----+--> Gross Profit ----+--> Operating Income ----+--> Net Income
                |                      |                         |
                +--> Cost of Sales     +--> Operating Expenses    +--> Tax & Other

You will NOT find clean "Cost of Sales" / "Operating Expenses" / "Tax" lines that
tie out perfectly. The robust trick is to pull only the four "kept" figures
(Revenue, Gross Profit, Operating Income, Net Income) and DERIVE each leak as the
residual:

    Cost of Sales       = Revenue          - Gross Profit
    Operating Expenses  = Gross Profit     - Operating Income
    Tax & Other         = Operating Income - Net Income

Because each leak is defined as "parent minus child", the waterfall balances by
construction - every node's inflow equals its outflow, automatically.


IDEA 2 - LINKS CAN'T BE NEGATIVE, SO SPLIT "SIZE" FROM "LABEL"
-------------------------------------------------------------
A bar can't have negative width, so the library sizes links by MAGNITUDE. But
real statements have negatives (a tax benefit, an operating loss). The pattern:
plot the magnitude for the *size*, and pass the real, signed number as the
*label* via `node_value_labels`. Accountants write negatives in (parentheses),
so our little `money()` helper does the same.


Run it:
    python 05_real_company_data.py            # defaults to MSFT
    python 05_real_company_data.py AAPL       # or any ticker
Writes  income_<TICKER>.png

Want this animated across 18 years with a live stock price riding along the
bottom? That's exactly what advanced/nvidia_reel.py does - you now understand
every piece of it.
"""
import sys

import pandas as pd
import yfinance as yf
from gradient_sankey import SankeyRaceMultiLayerParallel as Sankey

ticker = (sys.argv[1] if len(sys.argv) > 1 else "MSFT").upper()

# --- pull four clean lines from the latest quarterly income statement ---------
stmt = yf.Ticker(ticker).quarterly_income_stmt
latest = stmt[stmt.columns[0]]                       # most recent quarter (a column)
rev, gross, op, net = (float(latest[row]) / 1e9      # convert to $ billions
                       for row in ["Total Revenue", "Gross Profit",
                                   "Operating Income", "Net Income"])

# --- IDEA 1: derive the three leaks as residuals so it balances ---------------
cogs = rev - gross      # Cost of Sales
opex = gross - op       # Operating Expenses
tax = op - net          # Tax & Other

# --- build the four-layer waterfall (same from_dataframe pattern as always) ----
flows = {
    ("Revenue", "Gross Profit"): gross,        ("Revenue", "Cost of Sales"): cogs,
    ("Gross Profit", "Operating Income"): op,  ("Gross Profit", "Operating Expenses"): opex,
    ("Operating Income", "Net Income"): net,   ("Operating Income", "Tax & Other"): tax,
}
# IDEA 2: the *size* is abs(value). The signed value goes into the label later.
df = pd.DataFrame([{"t": 0, "s": s, "d": d, "v": abs(v)} for (s, d), v in flows.items()])
layers = [["Revenue"],
          ["Gross Profit", "Cost of Sales"],
          ["Operating Income", "Operating Expenses"],
          ["Net Income", "Tax & Other"]]

# Colour by position (lesson 04): a green "profit spine" vs magenta "leaks",
# with Revenue picked out in blue as the single source.
KEPT, LEAK = "#33E08A", "#FF2E97"
colors = {n: (KEPT if i == 0 else LEAK) for layer in layers for i, n in enumerate(layer)}
colors["Revenue"] = "#4CC9F0"


def money(v):
    """4.78 -> '4.8' ; -4.78 -> '(4.8)'  (accounting style: negatives in parens)."""
    s = f"{abs(v):.1f}" if abs(v) < 10 else f"{abs(v):.0f}"
    return f"({s})" if v < 0 else s


# IDEA 2 again: a label per node, carrying the REAL signed figure.
labels = {n: money(v) for n, v in [
    ("Revenue", rev), ("Gross Profit", gross), ("Cost of Sales", cogs),
    ("Operating Income", op), ("Operating Expenses", opex),
    ("Net Income", net), ("Tax & Other", tax)]}

Sankey.from_dataframe(
    df, layers, time_col="t", source_col="s", target_col="d", value_col="v",
    node_colors=colors,
).save_frame(
    f"income_{ticker}.png",
    title=f"{ticker} - income statement ($B, latest quarter)",
    theme="dark", link_glow=1, stacked_mode=True, ranking_mode=False,
    node_value_labels=labels,                  # <- show the real signed numbers
    yaxis_node="Revenue", yaxis_suffix="B",    # <- a discreet $ axis next to Revenue
    figsize=(13, 7), dpi=140, padding=1.8,
)
print(f"Done! Open income_{ticker}.png")
print("You derived the leaks as residuals (so it balances) and labelled the real figures.")
print("Next up: 06_background_music.py (add a soundtrack).")
