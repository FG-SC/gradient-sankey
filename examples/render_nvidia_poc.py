"""PoC: single dark/neon frame of NVIDIA's DRE waterfall (latest quarter),
with accounting-style (parentheses) labels for negative figures."""
import sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sankey_race_multi_layers_parallel import SankeyRaceMultiLayerParallel
from nvidia_dre import build, to_flows

HERE = os.path.dirname(os.path.abspath(__file__))

wide = build()
flows = to_flows(wide)
flows["value"] = flows["value"] / 1e9   # -> $ billions

SHORT = {
    "Revenue": "Revenue", "Gross Profit": "Gross", "Cost of Revenue": "COGS",
    "Operating Income": "Op Income", "Operating Expenses": "OpEx",
    "Net Income": "Net Income", "Tax & Other": "Tax+Other",
}
flows["source"] = flows["source"].map(lambda s: SHORT.get(s, s))
flows["target"] = flows["target"].map(lambda s: SHORT.get(s, s))

layers = [["Revenue"], ["Gross", "COGS"], ["Op Income", "OpEx"], ["Net Income", "Tax+Other"]]
# Color by position within each layer (uniform per track): 0 = profit spine, 1 = leak
POS_COLORS = ["#33E08A", "#FF2E97"]
node_colors = {n: POS_COLORS[i] for layer in layers for i, n in enumerate(layer)}


def fmt(v_b: float) -> str:
    """$B label; negatives in (parentheses), accounting style."""
    s = f"{abs(v_b):.0f}"
    return f"({s})" if v_b < 0 else s


# Signed $B labels for the latest quarter -> correct values incl. negatives
last = wide.iloc[-1]
labels = {
    "Revenue":    fmt(last["revenue"]    / 1e9),
    "Gross":      fmt(last["gross_profit"] / 1e9),
    "COGS":       fmt(last["cogs"]       / 1e9),
    "Op Income":  fmt(last["op_income"]  / 1e9),
    "OpEx":       fmt(last["opex"]       / 1e9),
    "Net Income": fmt(last["net_income"] / 1e9),
    "Tax+Other":  fmt(last["tax_other"]  / 1e9),
}

sankey = SankeyRaceMultiLayerParallel.from_dataframe(
    df=flows, layers=layers,
    time_col="quarter", source_col="source", target_col="target", value_col="value",
    node_colors=node_colors,
)

n = flows["quarter"].nunique()
sankey.save_frame(
    output_path=os.path.join(HERE, "nvidia_poc.png"),
    frame_index=n - 1,
    title="NVIDIA  -  income statement  ($B / quarter)",
    figsize=(16, 9), dpi=130, font_size=12, title_fontsize=20,
    padding=2.0, theme="dark", link_glow=1, link_alpha=0.5,
    ranking_mode=False, stacked_mode=True,
    node_value_labels=labels,
)
print("done:", labels)
