"""
04 · Colour with meaning  🌈
=============================
Three ways to control colour, from simplest to fanciest:

  A) a built-in palette,
  B) "colour by position" — a clean, uniform look where each track keeps one hue,
  C) "dynamic colours" — nodes recolour every frame based on their value
     (great for showing growth). Dynamic colour is animation-only, so part C
     writes a short MP4 (needs FFmpeg).

Run it:
    python 04_colors.py
Writes  colors_palette.png, colors_by_position.png, colors_dynamic.mp4
"""
import multiprocessing as mp

import pandas as pd
from gradient_sankey import SankeyRaceMultiLayerParallel as Sankey
from gradient_sankey import ColorPalette, get_palette_colors

layers = [["Revenue"], ["Profit", "Costs"]]


def one_period(values):
    """Build a one-frame Sankey from {(source, target): value}."""
    df = pd.DataFrame([{"t": 0, "s": s, "d": d, "v": v} for (s, d), v in values.items()])
    return df


# --- A) a built-in palette ------------------------------------------------------
# `get_palette_colors` turns any palette into N evenly-spaced hex colours.
# It accepts a ColorPalette, any matplotlib colormap name, or your own hex list.
names = ["Revenue", "Profit", "Costs"]
palette = get_palette_colors(ColorPalette.OCEAN, len(names))
colors = dict(zip(names, palette))
df = one_period({("Revenue", "Profit"): 70, ("Revenue", "Costs"): 30})
Sankey.from_dataframe(df, layers, time_col="t", source_col="s", target_col="d",
                      value_col="v", node_colors=colors
                      ).save_frame("colors_palette.png", title="OCEAN palette",
                                   figsize=(9, 5), dpi=130)

# --- B) colour by position ------------------------------------------------------
# Give the i-th node of EVERY layer the same colour, so each "track" of the flow
# is one consistent hue and the gradient only appears where tracks split.
POS = ["#33E08A", "#FF2E97"]   # position 0 (kept) = green, position 1 (leak) = magenta
by_pos = {n: POS[i % len(POS)] for layer in layers for i, n in enumerate(layer)}
Sankey.from_dataframe(df, layers, time_col="t", source_col="s", target_col="d",
                      value_col="v", node_colors=by_pos
                      ).save_frame("colors_by_position.png", theme="dark", link_glow=1,
                                   title="colour by position", figsize=(9, 5), dpi=130)


# --- C) dynamic colours (animation-only) ---------------------------------------
def main():
    # Profit grows over four periods; "intensity" mode keeps each node's hue but
    # brightens it as the value rises, so the chart literally lights up.
    rows = []
    for t, profit in enumerate([20, 40, 65, 90]):
        rows += [{"t": t, "s": "Revenue", "d": "Profit", "v": profit},
                 {"t": t, "s": "Revenue", "d": "Costs",  "v": 100 - profit}]
    sk = Sankey.from_dataframe(pd.DataFrame(rows), layers, time_col="t", source_col="s",
                               target_col="d", value_col="v", node_colors=by_pos)
    sk.animate("colors_dynamic.mp4", title="dynamic colour (intensity)",
               theme="dark", link_glow=1, dynamic_color_mode="intensity",
               fps=24, duration_seconds=5)
    print("Done! See colors_palette.png, colors_by_position.png, colors_dynamic.mp4")
    print('Other modes to try: "ranking", "value", "percentile". Next: 05_real_company_data.py')


if __name__ == "__main__":
    mp.freeze_support()
    main()
