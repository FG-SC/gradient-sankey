"""
04 - Colour with meaning
========================
Colour isn't just decoration here - the gradient on every link goes FROM the
source node's colour TO the target node's colour. So the colours you assign to
nodes are what make a busy diagram readable. This lesson covers the three ways
to control them, from simplest to fanciest.


HOW NODE COLOURS WORK
---------------------
You pass `node_colors`, a dict mapping each node name to a hex colour:

    node_colors = {"Revenue": "#4CC9F0", "Profit": "#33E08A", "Costs": "#FF2E97"}

Every node you put in a layer should have an entry. A link from "Revenue" to
"Profit" then fades from cyan to green. That fade is the whole trick - your eye
follows a flow by watching its colour travel.


A) FROM A PALETTE
-----------------
Hand-picking hex codes gets old. `get_palette_colors(palette, n)` gives you n
evenly-spaced colours from a palette. It accepts a built-in `ColorPalette`, any
matplotlib colormap name ("viridis", "RdYlGn", ...), or your own list of hex
colours (which it interpolates). Zip the result onto your node names.


B) COLOUR BY POSITION (a pro idiom for waterfalls)
--------------------------------------------------
For income statements and other "spine + leaks" flows, a clean look is to give
the i-th node of EVERY layer the same colour. Then each "track" of the flow is
one consistent hue and the gradient only appears where a track splits:

    POS = ["#33E08A", "#FF2E97"]   # position 0 = the kept "spine", 1 = the leak
    node_colors = {n: POS[i % len(POS)]
                   for layer in layers for i, n in enumerate(layer)}

PITFALL worth memorising: write `POS[i % len(POS)]`, not `POS[i]`. If a layer
ever has more nodes than you have colours, plain `POS[i]` raises IndexError; the
`% len(POS)` simply cycles the palette and stays safe for any layer width.


C) DYNAMIC COLOURS (animation-only)
-----------------------------------
The fanciest option: recolour nodes EVERY FRAME based on their value or rank, so
the chart reacts as the data moves. Because it depends on motion, it's an
`animate()` feature only (it does nothing on a static save_frame). Modes:

    "static"   fixed colours (the default)
    "ranking"  colour by rank within the layer (1st, 2nd, ...)
    "value"    colour by value, normalised within each layer
    "intensity" keep each node's hue but brighten it as the value grows
                -> nodes literally "light up" as they get bigger

We use "intensity" below on a Profit figure that climbs over four periods.


Run it:
    python 04_colors.py
Writes  colors_palette.png, colors_by_position.png, colors_dynamic.mp4
"""
import multiprocessing as mp

import pandas as pd
from gradient_sankey import SankeyRaceMultiLayerParallel as Sankey
from gradient_sankey import ColorPalette, get_palette_colors

# A tiny two-layer "revenue splits into profit and costs" flow we'll reuse.
layers = [["Revenue"], ["Profit", "Costs"]]


def frame(values):
    """Helper: build a one-period table from {(source, target): value}."""
    return pd.DataFrame([{"t": 0, "s": s, "d": d, "v": v} for (s, d), v in values.items()])


# --- A) from a palette --------------------------------------------------------
names = ["Revenue", "Profit", "Costs"]
colors_from_palette = dict(zip(names, get_palette_colors(ColorPalette.OCEAN, len(names))))
df = frame({("Revenue", "Profit"): 70, ("Revenue", "Costs"): 30})
Sankey.from_dataframe(df, layers, time_col="t", source_col="s", target_col="d",
                      value_col="v", node_colors=colors_from_palette
                      ).save_frame("colors_palette.png", title="OCEAN palette",
                                   figsize=(9, 5), dpi=130)

# --- B) colour by position (note the `% len(POS)` safety) ---------------------
POS = ["#33E08A", "#FF2E97"]    # spine = green, leak = magenta
by_position = {n: POS[i % len(POS)] for layer in layers for i, n in enumerate(layer)}
Sankey.from_dataframe(df, layers, time_col="t", source_col="s", target_col="d",
                      value_col="v", node_colors=by_position
                      ).save_frame("colors_by_position.png", theme="dark", link_glow=1,
                                   title="colour by position", figsize=(9, 5), dpi=130)


# --- C) dynamic colours, animation-only ---------------------------------------
def main():
    # Profit climbs 20 -> 90 over four periods; costs are the remainder of 100.
    rows = []
    for t, profit in enumerate([20, 40, 65, 90]):
        rows += [{"t": t, "s": "Revenue", "d": "Profit", "v": profit},
                 {"t": t, "s": "Revenue", "d": "Costs",  "v": 100 - profit}]
    sankey = Sankey.from_dataframe(pd.DataFrame(rows), layers, time_col="t",
                                   source_col="s", target_col="d", value_col="v",
                                   node_colors=by_position)
    sankey.animate(
        "colors_dynamic.mp4", title="dynamic colour (intensity)",
        theme="dark", link_glow=1,
        dynamic_color_mode="intensity",     # <- the magic; try "ranking" or "value" too
        fps=24, duration_seconds=5,
    )
    print("Done! See colors_palette.png, colors_by_position.png, colors_dynamic.mp4")
    print("Watch Profit brighten as it grows - that's dynamic colour reacting to the data.")
    print("Next up: 05_real_company_data.py (real numbers, the residual trick).")


if __name__ == "__main__":
    mp.freeze_support()
    main()
