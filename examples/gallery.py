"""
Gallery / Cookbook
==================
A single, self-contained tour of the main features, using a small synthetic
3-layer "energy transition" dataset (no external API needed).

It shows, end to end:
  1. The exact INPUT OBJECTS the library expects (a tidy DataFrame + `layers`).
  2. PALETTE SELECTION:
       - inter-layer  : a different palette per layer
       - intra-layer  : how a palette spreads across the nodes of one layer
       - by-position  : the same color per index across layers (uniform "tracks")
       - explicit map : hand-picked hex per node
  3. POSITIONING MODES: stacked+ranking / ranking / stacked / fixed.
  4. DYNAMIC NODE COLORS: ranking / value / intensity (animation only).

Run it:
    python examples/gallery.py            # static PNGs (fast, no FFmpeg needed)
    python examples/gallery.py --animate  # also render short dynamic-color MP4s (needs FFmpeg)

Outputs go to examples/gallery_out/.
"""

import os
import sys
import argparse
import multiprocessing as mp
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gradient_sankey import (
    SankeyRaceMultiLayerParallel,
    ColorPalette,
    get_palette_colors,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "gallery_out")
os.makedirs(OUT, exist_ok=True)


# =============================================================================
# 1. INPUT OBJECTS
# =============================================================================
# The library needs TWO things:
#   (a) a tidy DataFrame, one row per flow, with columns for time/source/target/value
#   (b) `layers`: a list of lists, left -> right; each inner list is one column.
#
# RULE: node names must be UNIQUE across all layers. Here the three layers use
# naturally distinct names (sources / carriers / sectors), so we're fine.

LAYERS = [
    ["Coal", "Gas", "Nuclear", "Solar", "Wind"],   # layer 0: primary sources
    ["Electricity", "Heat", "Hydrogen"],            # layer 1: energy carriers
    ["Industry", "Buildings", "Transport"],         # layer 2: end-use sectors
]


def build_data() -> pd.DataFrame:
    """Return a tidy flow DataFrame for 3 snapshots (2020, 2025, 2030).

    Each dict below is exactly the kind of object you'd assemble from your own
    inputs (a CSV, an API, a query result ...). Values are illustrative and
    shift fossil -> renewable over time.
    """
    # (source -> target): value, per year.  source/target must match LAYERS names.
    per_year = {
        2020: {
            ("Coal", "Electricity"): 28, ("Gas", "Electricity"): 22, ("Gas", "Heat"): 10,
            ("Nuclear", "Electricity"): 18, ("Solar", "Electricity"): 6, ("Wind", "Electricity"): 8,
            ("Electricity", "Industry"): 30, ("Electricity", "Buildings"): 28, ("Electricity", "Transport"): 4,
            ("Heat", "Industry"): 6, ("Heat", "Buildings"): 4,
            ("Hydrogen", "Industry"): 2, ("Hydrogen", "Transport"): 1,
        },
        2025: {
            ("Coal", "Electricity"): 16, ("Gas", "Electricity"): 20, ("Gas", "Heat"): 9,
            ("Nuclear", "Electricity"): 18, ("Solar", "Electricity"): 16, ("Wind", "Electricity"): 18,
            ("Electricity", "Industry"): 34, ("Electricity", "Buildings"): 30, ("Electricity", "Transport"): 12,
            ("Heat", "Industry"): 6, ("Heat", "Buildings"): 3,
            ("Hydrogen", "Industry"): 4, ("Hydrogen", "Transport"): 4,
        },
        2030: {
            ("Coal", "Electricity"): 5, ("Gas", "Electricity"): 12, ("Gas", "Heat"): 6,
            ("Nuclear", "Electricity"): 18, ("Solar", "Electricity"): 30, ("Wind", "Electricity"): 32,
            ("Electricity", "Industry"): 38, ("Electricity", "Buildings"): 33, ("Electricity", "Transport"): 26,
            ("Heat", "Industry"): 5, ("Heat", "Buildings"): 2,
            ("Hydrogen", "Industry"): 7, ("Hydrogen", "Transport"): 9,
        },
    }
    rows = []
    for year, flows in per_year.items():
        for (src, tgt), val in flows.items():
            rows.append({"year": year, "source": src, "target": tgt, "value": val})
    return pd.DataFrame(rows)


def make_sankey(df, node_colors):
    return SankeyRaceMultiLayerParallel.from_dataframe(
        df=df, layers=LAYERS,
        time_col="year", source_col="source", target_col="target", value_col="value",
        node_colors=node_colors,
    )


# =============================================================================
# 2. PALETTE SELECTION
# =============================================================================
def colors_inter_layer() -> dict:
    """INTER-layer: give each LAYER its own palette.

    Each layer's palette is spread across that layer's nodes (that part is the
    'intra-layer' selection). This is the most common way to make layers read
    as distinct groups.
    """
    layer_palettes = [
        ColorPalette.EARTH,            # sources  (built-in enum)
        "Blues",                       # carriers (any matplotlib colormap name)
        ["#FF2E97", "#FFC400", "#7CFF6B"],  # sectors (custom hex list, interpolated)
    ]
    node_colors = {}
    for nodes, palette in zip(LAYERS, layer_palettes):
        # intra-layer: get exactly len(nodes) colors spread across the palette
        colors = get_palette_colors(palette, n_colors=len(nodes))
        node_colors.update(dict(zip(nodes, colors)))
    return node_colors


def colors_by_position() -> dict:
    """BY-POSITION: the i-th node of EVERY layer gets the same color.

    Makes each 'track' of the flow a uniform hue; the gradient then only appears
    where tracks split/merge. Great for a clean, calm look.
    """
    # one color per position; longest layer here has 5 nodes
    track = get_palette_colors(ColorPalette.RAINBOW, n_colors=max(len(l) for l in LAYERS))
    node_colors = {}
    for nodes in LAYERS:
        for i, node in enumerate(nodes):
            node_colors[node] = track[i]
    return node_colors


def colors_explicit() -> dict:
    """EXPLICIT: hand-pick a hex per node when you want full control / brand colors."""
    return {
        "Coal": "#4A4A4A", "Gas": "#FF7F0E", "Nuclear": "#9467BD", "Solar": "#FDB813", "Wind": "#17BECF",
        "Electricity": "#1F77B4", "Heat": "#D62728", "Hydrogen": "#2CA02C",
        "Industry": "#8C564B", "Buildings": "#E377C2", "Transport": "#7F7F7F",
    }


def colors_auto(df) -> dict:
    """AUTO: let from_dataframe pick per-layer palettes for you (pass node_colors=None).

    Returns the colors it generated so we can reuse/inspect them.
    """
    auto = SankeyRaceMultiLayerParallel.from_dataframe(
        df=df, layers=LAYERS,
        time_col="year", source_col="source", target_col="target", value_col="value",
        layer_palettes=[ColorPalette.SUNSET, ColorPalette.OCEAN, ColorPalette.NEON],  # inter-layer
        node_colors=None,   # -> auto-assigned from layer_palettes
    )
    return auto.node_colors


def demo_palettes(df):
    print("\n[2] PALETTE SELECTION -> static frames")
    variants = {
        "palette_inter_layer": colors_inter_layer(),
        "palette_by_position": colors_by_position(),
        "palette_explicit":    colors_explicit(),
        "palette_auto":        colors_auto(df),
    }
    for name, node_colors in variants.items():
        sankey = make_sankey(df, node_colors)
        sankey.save_frame(
            output_path=os.path.join(OUT, f"{name}.png"),
            frame_index=2, title=f"Energy mix 2030  ({name})",
            figsize=(14, 8), dpi=110,
            ranking_mode=False, stacked_mode=True,   # fixed order so palettes are easy to compare
        )


# =============================================================================
# 3. POSITIONING MODES
# =============================================================================
def demo_modes(df):
    print("\n[3] POSITIONING MODES -> static frames")
    node_colors = colors_inter_layer()
    sankey = make_sankey(df, node_colors)
    modes = {
        "mode_stacked_ranking": dict(ranking_mode=True,  stacked_mode=True),
        "mode_ranking":         dict(ranking_mode=True,  stacked_mode=False),
        "mode_stacked":         dict(ranking_mode=False, stacked_mode=True),
        "mode_fixed":           dict(ranking_mode=False, stacked_mode=False),
    }
    for name, flags in modes.items():
        sankey.save_frame(
            output_path=os.path.join(OUT, f"{name}.png"),
            frame_index=2, title=f"2030  ({name})",
            figsize=(14, 8), dpi=110, **flags,
        )


# =============================================================================
# 4. DYNAMIC NODE COLORS  (animation only)
# =============================================================================
def demo_dynamic_colors(df):
    print("\n[4] DYNAMIC NODE COLORS -> short MP4s (needs FFmpeg)")
    node_colors = colors_explicit()   # base colors; 'intensity' keeps these hues
    sankey = make_sankey(df, node_colors)

    common = dict(figsize=(14, 8), fps=24, duration_seconds=4.0, quality="low",
                  ranking_mode=False, stacked_mode=True, n_workers=4)

    # (a) ranking: 1st in each layer -> green, last -> red (neon RdYlGn)
    sankey.animate(output_path=os.path.join(OUT, "dyn_ranking.mp4"),
                   title="dynamic_color_mode='ranking'",
                   dynamic_color_mode="ranking",
                   dynamic_colormap=["#FF1E56", "#FFC400", "#7CFF6B"], **common)

    # (b) value: color by value, normalized within each layer
    sankey.animate(output_path=os.path.join(OUT, "dyn_value.mp4"),
                   title="dynamic_color_mode='value'",
                   dynamic_color_mode="value", dynamic_colormap="viridis", **common)

    # (c) intensity: keep each node's base hue, brighten with value
    sankey.animate(output_path=os.path.join(OUT, "dyn_intensity.mp4"),
                   title="dynamic_color_mode='intensity'  (dark theme)",
                   dynamic_color_mode="intensity", theme="dark", link_glow=1, **common)


def main(args):
    df = build_data()

    # [1] show the input objects
    print("[1] INPUT OBJECTS")
    print("    layers =", LAYERS)
    print("    df (tidy flows), first rows:")
    print(df.head(6).to_string(index=False))
    print(f"    {len(df)} rows | years: {sorted(df['year'].unique())}")

    demo_palettes(df)
    demo_modes(df)
    if args.animate:
        demo_dynamic_colors(df)

    print(f"\nDone. See {OUT}")


if __name__ == "__main__":
    mp.freeze_support()
    ap = argparse.ArgumentParser(description="Feature gallery / cookbook.")
    ap.add_argument("--animate", action="store_true",
                    help="Also render short dynamic-color MP4s (needs FFmpeg).")
    main(ap.parse_args())
