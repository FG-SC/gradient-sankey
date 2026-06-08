"""
NVIDIA income-statement (DRE) reel: dark/neon, 9:16 vertical, ~20s.
Real SEC EDGAR data 2008->2026. Bars grow in ABSOLUTE value (global scale),
fixed waterfall order (no ranking), accounting-style (parentheses) labels.
Output: examples/nvidia_dre.mp4
"""
import sys, os
import argparse
import multiprocessing as mp
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sankey_race_multi_layers_parallel import SankeyRaceMultiLayerParallel
from nvidia_dre import build, to_flows

HERE = os.path.dirname(os.path.abspath(__file__))

SHORT = {
    "Revenue": "Revenue", "Gross Profit": "Gross", "Cost of Revenue": "COGS",
    "Operating Income": "Op Income", "Operating Expenses": "OpEx",
    "Net Income": "Net Income", "Tax & Other": "Tax+Other",
}
# Color by POSITION within each layer, consistent across all layers, so each
# "track" of the flow is uniform: position 0 = profit spine (green),
# position 1 = leak (magenta). Gradient then only shows at the green->magenta splits.
POS_COLORS = ["#33E08A", "#FF2E97"]
# node -> wide-df column for the SIGNED $B label
LABEL_COL = {
    "Revenue": "revenue", "Gross": "gross_profit", "COGS": "cogs",
    "Op Income": "op_income", "OpEx": "opex",
    "Net Income": "net_income", "Tax+Other": "tax_other",
}


def fmt(v_b: float) -> str:
    a = abs(v_b)
    s = f"{a:.1f}" if a < 10 else f"{a:.0f}"   # 1 decimal while small, integer when big
    return f"({s})" if v_b < 0 else s


def main(args):
    wide = build()
    # Start the time series at a given year (default 2015)
    wide = wide[wide["period"] >= f"{args.start_year}-01-01"].reset_index(drop=True)
    flows = to_flows(wide)
    flows["value"] = flows["value"] / 1e9
    flows["source"] = flows["source"].map(lambda s: SHORT.get(s, s))
    flows["target"] = flows["target"].map(lambda s: SHORT.get(s, s))

    layers = [["Revenue"], ["Gross", "COGS"], ["Op Income", "OpEx"], ["Net Income", "Tax+Other"]]
    node_colors = {n: POS_COLORS[i] for layer in layers for i, n in enumerate(layer)}

    # Per-quarter parenthesized $B labels, aligned to the library's sorted order
    wide = wide.assign(quarter=flows.drop_duplicates("period").set_index("period")
                       .reindex(wide["period"])["quarter"].values)
    order = sorted(flows["quarter"].unique())
    by_q = {r["quarter"]: r for _, r in wide.iterrows()}
    labels_per_frame = [
        {node: fmt(by_q[q][col] / 1e9) for node, col in LABEL_COL.items()}
        for q in order
    ]

    # NVDA stock (adjusted close), sampled at each quarter-end -> growth overlay
    import yfinance as yf
    stock = yf.Ticker("NVDA").history(start=f"{args.start_year}-01-01", auto_adjust=True)["Close"]
    stock.index = stock.index.tz_localize(None)
    overlay = [float(stock.asof(pd.Timestamp(by_q[q]["period"]))) for q in order]

    sankey = SankeyRaceMultiLayerParallel.from_dataframe(
        df=flows, layers=layers,
        time_col="quarter", source_col="source", target_col="target", value_col="value",
        node_colors=node_colors,
    )

    sankey.animate(
        output_path=os.path.join(HERE, "nvidia_dre.mp4"),
        title="NVIDIA  income statement  ($B / quarter)",
        figsize=(16, 9), fps=30, duration_seconds=args.duration, quality="high",  # landscape, ~45s
        font_size=12, title_fontsize=20, padding=2.4,
        margin_bottom=0.20,                           # reserve bottom band for the stock overlay
        theme="dark", link_glow=1, link_alpha=0.5,   # softer
        ranking_mode=False, stacked_mode=True,        # fixed waterfall order
        absolute_scale=False,                         # per-frame fill -> early quarters stay visible
        node_value_labels_per_frame=labels_per_frame,
        # Dynamic $ y-axis on the Revenue bar (evolving scale -> shows the dimension)
        yaxis_node="Revenue", yaxis_suffix="B",
        # Growth overlay: NVDA stock climbing (mini-chart + discreet big number)
        overlay_series=overlay,
        overlay_label="NVDA stock  ($, split-adj.)",
        overlay_color="#7CFF6B",
        overlay_value_suffix="",
        overlay_x_labels=order,            # year ticks on the stock chart's x-axis
        # Background music (optional): a local mp3 (--audio) OR a YouTube URL (--audio-url).
        audio_path=args.audio,
        audio_url=args.audio_url,
        audio_start=args.audio_start,
        n_workers=6,
    )


if __name__ == "__main__":
    mp.freeze_support()
    ap = argparse.ArgumentParser(description="Render the NVIDIA DRE reel.")
    ap.add_argument("--start-year", type=int, default=2015,
                    help="First year of the time series (default 2015).")
    ap.add_argument("--duration", type=float, default=45.0,
                    help="Video duration in seconds (default 45).")
    ap.add_argument("--audio", type=str, default=None,
                    help="Path to a local MP3 to use as background music.")
    ap.add_argument("--audio-url", type=str, default=None,
                    help="YouTube URL to download as background music (needs yt-dlp). "
                         "Timestamps in the URL are ignored.")
    ap.add_argument("--audio-start", type=float, default=0.0,
                    help="Seconds into the track where playback begins (e.g. start at the drop).")
    main(ap.parse_args())
