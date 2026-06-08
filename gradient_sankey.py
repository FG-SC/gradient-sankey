"""
gradient-sankey - High-quality animated, true-gradient Sankey diagrams
=======================================================================

Combines:
- High visual quality (FancyBboxPatch nodes, cubic-Bezier gradient links).
- Parallel video rendering with multiprocessing + FFmpeg piping.
- All color palettes and positioning modes (ranking, stacked, stacked+ranking,
  fixed), dynamic node colors, dark theme + neon glow, accounting-style value
  labels, an optional dynamic value axis and a bar-chart-race style overlay,
  plus background-music muxing (local MP3 or a YouTube URL).

Expected speedup: ~Nx (N = number of worker processes).
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import subprocess
import time
import warnings
import multiprocessing as mp
import tempfile
import shutil
import os

__version__ = "1.1.2"


# =============================================================================
# Color palettes
# =============================================================================

class ColorPalette(Enum):
    """Built-in color palettes."""
    RAINBOW = "rainbow"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    PASTEL = "pastel"
    DARK = "dark"
    EARTH = "earth"
    OCEAN = "ocean"
    SUNSET = "sunset"
    NEON = "neon"


# Mapping from ColorPalette to matplotlib colormap names.
# These are continuous colormaps that support smooth interpolation.
PALETTE_COLORMAPS = {
    ColorPalette.RAINBOW: "rainbow",
    ColorPalette.VIRIDIS: "viridis",
    ColorPalette.PLASMA: "plasma",
    ColorPalette.INFERNO: "inferno",
    ColorPalette.PASTEL: "Pastel1",
    ColorPalette.DARK: "Dark2",
    ColorPalette.EARTH: "YlOrBr",
    ColorPalette.OCEAN: "Blues",
    ColorPalette.SUNSET: "Oranges",
    ColorPalette.NEON: "Set1",
}


def get_colormap(palette: Union[ColorPalette, str, List[str]]):
    """
    Get a matplotlib colormap from various inputs.

    Args:
        palette: Can be:
            - ColorPalette enum (e.g., ColorPalette.VIRIDIS)
            - String name of a matplotlib colormap (e.g., "viridis", "RdYlGn")
            - List of hex colors for a custom palette (interpolated)

    Returns:
        matplotlib colormap object
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Custom discrete palette (list of hex colors)
    if isinstance(palette, list):
        rgb_colors = [to_rgb(c) for c in palette]
        return LinearSegmentedColormap.from_list("custom", rgb_colors, N=256)

    # ColorPalette enum
    if isinstance(palette, ColorPalette):
        cmap_name = PALETTE_COLORMAPS.get(palette, "rainbow")
        return plt.colormaps.get_cmap(cmap_name)

    # String - could be a ColorPalette name or a matplotlib colormap name
    if isinstance(palette, str):
        try:
            palette_enum = ColorPalette(palette.lower())
            cmap_name = PALETTE_COLORMAPS.get(palette_enum, "rainbow")
            return plt.colormaps.get_cmap(cmap_name)
        except ValueError:
            try:
                return plt.colormaps.get_cmap(palette)
            except (ValueError, KeyError):
                return plt.colormaps.get_cmap("rainbow")

    return plt.colormaps.get_cmap("rainbow")


def get_palette_colors(palette: Union[ColorPalette, str, List[str]], n_colors: int, reverse: bool = False) -> List[str]:
    """
    Get a list of colors from a palette with continuous interpolation.

    Args:
        palette: A ColorPalette enum, a matplotlib colormap name, or a list of
            hex colors to interpolate between.
        n_colors: Number of colors to generate.
        reverse: If True, reverse the color order.

    Returns:
        List of hex colors (#RRGGBB).

    Examples:
        get_palette_colors(ColorPalette.VIRIDIS, 5)
        get_palette_colors("RdYlGn", 10)
        get_palette_colors(["#FF0000", "#FFFF00", "#00FF00"], 10)
    """
    cmap = get_colormap(palette)

    if n_colors <= 1:
        positions = [0.5]
    else:
        positions = [i / (n_colors - 1) for i in range(n_colors)]

    if reverse:
        positions = positions[::-1]

    return [plt.matplotlib.colors.rgb2hex(cmap(pos)[:3]) for pos in positions]


@lru_cache(maxsize=256)
def get_rgb_cached(color: str) -> Tuple[float, float, float]:
    """Convert a hex color to an RGB tuple (cached)."""
    return to_rgb(color)


def get_text_color_for_background(bg_color: str) -> str:
    """Return the ideal text color (black/white) for a given background."""
    try:
        rgb = get_rgb_cached(bg_color)
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return '#000000' if luminance > 0.5 else '#FFFFFF'
    except (ValueError, TypeError):
        return '#000000'


# =============================================================================
# Dynamic Color Modes
# =============================================================================

class DynamicColorMode(Enum):
    """Modes for dynamic node coloring."""
    STATIC = "static"           # Fixed colors (original behavior)
    RANKING = "ranking"         # Color by rank within layer (1st=green, last=red)
    VALUE = "value"             # Color by value (normalized within layer)
    GLOBAL_VALUE = "global_value"  # Color by value (normalized globally)
    PERCENTILE = "percentile"   # Color by percentile within layer
    INTENSITY = "intensity"     # Keep each node's base HUE, scale brightness by value (globally)


def scale_brightness(hex_color: str, factor: float, floor: float = 0.30) -> str:
    """Keep a color's hue but scale its brightness/saturation by factor in [0,1].

    Dim (low value) -> dark & desaturated; bright (high value) -> full neon.
    `floor` keeps low values visible instead of pitch black.
    """
    import colorsys
    factor = max(0.0, min(1.0, factor))
    r, g, b = to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    lift = floor + (1.0 - floor) * factor
    v2 = v * lift
    s2 = s * (0.45 + 0.55 * factor)
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s2, v2)
    return plt.matplotlib.colors.rgb2hex((r2, g2, b2))


def get_dynamic_color(value: float, min_val: float, max_val: float,
                      colormap: Union[ColorPalette, str, List[str]] = 'RdYlGn') -> str:
    """Return a color based on a normalized value."""
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (value - min_val) / (max_val - min_val)
    cmap = get_colormap(colormap)
    return plt.matplotlib.colors.rgb2hex(cmap(normalized)[:3])


def get_ranking_color(rank: int, total: int, colormap: Union[ColorPalette, str, List[str]] = 'RdYlGn') -> str:
    """Return a color based on a ranking (1st = high/green, last = low/red)."""
    if total <= 1:
        normalized = 1.0
    else:
        normalized = 1.0 - (rank - 1) / (total - 1)
    cmap = get_colormap(colormap)
    return plt.matplotlib.colors.rgb2hex(cmap(normalized)[:3])


def interpolate_color(color1: str, color2: str, t: float) -> str:
    """Interpolate between two hex colors."""
    c1 = to_rgb(color1)
    c2 = to_rgb(color2)
    r = c1[0] + (c2[0] - c1[0]) * t
    g = c1[1] + (c2[1] - c1[1]) * t
    b = c1[2] + (c2[2] - c1[2]) * t
    return plt.matplotlib.colors.rgb2hex((r, g, b))


def youtube_to_mp3(url: str, out_dir: str = None, filename: str = None) -> str:
    """
    Download a YouTube video's audio as an MP3 (optional feature).

    Requires `yt-dlp` (pip install yt-dlp) and `ffmpeg` on PATH. Any timestamp or
    playlist parameters in the URL are IGNORED -- only the 11-char video id is used,
    so the full track is always downloaded.

    Args:
        url: A YouTube URL (watch?v=, youtu.be/, shorts/, embed/ ... with or without &t=).
        out_dir: Where to save the mp3 (default: a temp dir).
        filename: Output filename without extension (default: the video title).

    Returns:
        Absolute path to the resulting .mp3 file.
    """
    import re
    import tempfile as _tempfile

    try:
        import yt_dlp
    except ImportError as e:
        raise ImportError(
            "YouTube audio support requires 'yt-dlp'. Install it with:\n"
            "    pip install yt-dlp\n"
            "and make sure ffmpeg is available on your PATH."
        ) from e

    # Ignore timestamps / playlist junk: keep only the video id
    m = re.search(r'(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})', url)
    clean_url = f"https://www.youtube.com/watch?v={m.group(1)}" if m else url

    out_dir = out_dir or _tempfile.mkdtemp(prefix="sankey_audio_")
    os.makedirs(out_dir, exist_ok=True)
    out_tmpl = os.path.join(out_dir, (filename or "%(title)s") + ".%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_tmpl,
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3', 'preferredquality': '0'}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(clean_url, download=True)
    base = ydl.prepare_filename(info)
    mp3_path = os.path.splitext(base)[0] + ".mp3"
    return mp3_path


# =============================================================================
# Data structures (pickleable for multiprocessing)
# =============================================================================

@dataclass
class FrameData:
    """Data for one (interpolated) frame."""
    time_label: str
    links: List[Tuple[str, str, float]]
    node_positions: Dict[str, float]
    node_values: Dict[str, float]
    node_colors: Optional[Dict[str, str]] = None  # Dynamic colors per frame
    node_value_labels: Optional[Dict[str, str]] = None  # Override inside-node text (e.g. "(5)" for negatives)
    progress: float = 0.0  # Continuous data-frame index (0..n_data_frames-1) for timeline overlays


# Quality defaults
DEFAULT_GRADIENT_SEGMENTS = 50


# =============================================================================
# Shared low-level drawing helpers
# =============================================================================

def _bezier_factor(t_arr):
    """Ease-in-out cubic factor for the link curves (frame-invariant in t)."""
    return np.where(
        t_arr < 0.5,
        4 * t_arr ** 3,
        1 - (-2 * t_arr + 2) ** 3 / 2
    )


def _bezier_vec(t_arr, y_start, y_end):
    """Vectorized smooth (ease-in-out cubic) interpolation used for link curves."""
    return y_start + (y_end - y_start) * _bezier_factor(t_arr)


def _rmtree_robust(path, tries=4, delay=0.15):
    """Remove a temp dir, retrying briefly. On Windows a lingering FFmpeg/handle
    can leave the (now-empty) directory behind; ``ignore_errors=True`` alone then
    silently abandons it, so these accumulate. Retry, then give up quietly."""
    for _ in range(tries):
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return
        time.sleep(delay)
    shutil.rmtree(path, ignore_errors=True)


def _draw_frame(ax, frame_data, cfg):
    """Render one fully-composed Sankey frame onto ``ax``.

    This is the SINGLE source of truth for frame drawing, shared by both the
    static renderer (``render_frame_high_quality`` / ``save_frame``) and the
    parallel video worker (``_render_chunk``), so the two paths can never drift.

    It draws: gradient links (with crossing reduction + optional neon glow),
    rounded nodes with their name/value labels (accounting parentheses and a
    "value-next-to-name" fallback for tiny bars), the optional dynamic value
    Y-axis, and the optional bar-chart-race overlay. The TITLE is intentionally
    left to the caller (it differs slightly between the static and video paths).

    ``cfg`` carries layout, theme and feature parameters plus the frame-invariant
    precomputed values ``_t0``/``_t1``/``_t_mid`` and ``_layer_x_positions``.
    """
    layers = cfg['layers']
    n_layers = len(layers)
    node_colors = cfg['node_colors']
    rgb_cache = cfg.get('rgb_cache') or {}
    n_segments = cfg['n_segments']
    plot_width = cfg['plot_width']
    plot_height = cfg['plot_height']
    node_width = cfg['node_width']
    font_size = cfg['font_size']
    bar_height_ratio = cfg['bar_height_ratio']
    margin_top_ratio = cfg['margin_top_ratio']
    margin_bottom_ratio = cfg['margin_bottom_ratio']
    stacked_mode = cfg['stacked_mode']
    label_color = cfg.get('label_color', '#000000')
    node_edge_color = cfg.get('node_edge_color', '#333333')
    link_alpha = cfg.get('link_alpha', 0.7)
    link_glow = cfg.get('link_glow', 0)
    fixed_layer_max = cfg.get('fixed_layer_max')
    yaxis_node = cfg.get('yaxis_node')

    t0 = cfg['_t0']
    t1 = cfg['_t1']
    t_mid = cfg['_t_mid']
    layer_x_positions = cfg['_layer_x_positions']
    # Frame-invariant cubic-ease factors, hoisted out of the per-link loop.
    f0 = cfg.get('_f0')
    f1 = cfg.get('_f1')
    if f0 is None:
        f0 = np.where(t0 < 0.5, 4 * t0 ** 3, 1 - (-2 * t0 + 2) ** 3 / 2)
        f1 = np.where(t1 < 0.5, 4 * t1 ** 3, 1 - (-2 * t1 + 2) ** 3 / 2)

    def _node_hex(n):
        if frame_data.node_colors:
            return frame_data.node_colors.get(n, '#888888')
        return node_colors.get(n, '#888888')

    def _node_rgb(n):
        if frame_data.node_colors:
            return np.array(to_rgb(frame_data.node_colors.get(n, '#888888')))
        if n in rgb_cache:
            return np.array(rgb_cache[n])
        return np.array(to_rgb(node_colors.get(n, '#888888')))

    # Margins
    margin_top = plot_height * margin_top_ratio
    margin_bottom = plot_height * margin_bottom_ratio
    usable_height = plot_height - margin_top - margin_bottom

    # Scale
    if stacked_mode:
        if fixed_layer_max:
            max_layer_total = fixed_layer_max   # fixed GLOBAL scale (absolute growth)
        else:
            layer_totals = [sum(frame_data.node_values.get(node, 0) for node in layer_nodes)
                            for layer_nodes in layers]
            max_layer_total = max(layer_totals) if layer_totals else 1
        stacked_height = usable_height * bar_height_ratio
        scale = stacked_height / max_layer_total if max_layer_total > 0 else 1
    else:
        # Fixed mode: uniform node heights (no scaling by value)
        max_nodes_in_layer = max((len(layer) for layer in layers), default=1)
        available_height_per_slot = usable_height / max(1, max_nodes_in_layer)
        fixed_node_height = available_height_per_slot * bar_height_ratio * 0.8

    # Node rectangles
    node_positions_rect = {}
    for layer_idx, layer_nodes in enumerate(layers):
        layer_x = layer_x_positions[layer_idx]
        for node in layer_nodes:
            val = frame_data.node_values.get(node, 0)
            if stacked_mode:
                h = max(val * scale, 0.1)
            else:
                h = fixed_node_height
            y_center = frame_data.node_positions.get(node, margin_bottom + usable_height * 0.5)
            node_positions_rect[node] = (layer_x, y_center - h / 2, h)

    # Node in/out sums
    out_sums = {}
    in_sums = {}
    for src, tgt, val in frame_data.links:
        out_sums[src] = out_sums.get(src, 0) + val
        in_sums[tgt] = in_sums.get(tgt, 0) + val

    node_layer_idx = cfg.get('node_layer')
    if not node_layer_idx:
        node_layer_idx = {}
        for layer_idx, layer_nodes in enumerate(layers):
            for node in layer_nodes:
                node_layer_idx[node] = layer_idx

    # Link scales. Intermediate nodes use the SAME scale for in & out (based on
    # max(in, out)) so the spine stays visually consistent across a node.
    node_out_scale = {}
    node_in_scale = {}
    for node in node_positions_rect:
        total_out = out_sums.get(node, 0)
        total_in = in_sums.get(node, 0)
        h = node_positions_rect[node][2]
        layer_idx = node_layer_idx.get(node, 0)
        if 0 < layer_idx < n_layers - 1:
            total_max = max(total_out, total_in, 0.001)
            node_out_scale[node] = h / total_max if total_out > 0 else 1
            node_in_scale[node] = h / total_max if total_in > 0 else 1
        else:
            node_out_scale[node] = h / total_out if total_out > 0 else 1
            node_in_scale[node] = h / total_in if total_in > 0 else 1

    # Build gradient quads with crossing reduction: each node's outgoing flows are
    # stacked by the target's Y-center, incoming flows by the source's Y-center.
    # One (n_segments, 4, 2) vertex block + (n_segments, 4) RGBA block per link,
    # concatenated once into a single PolyCollection (numpy-vectorized, no per-
    # segment Python loop). Draw order is preserved exactly (alpha-blend safe).
    vert_blocks = []
    color_blocks = []

    def _yc(n):
        _nx, nyb, nh = node_positions_rect[n]
        return nyb + nh / 2

    valid_links = [(s, t, v) for (s, t, v) in frame_data.links
                   if v > 0 and s in node_positions_rect and t in node_positions_rect]

    # Bucket once (avoids O(nodes x links) rescans of valid_links)
    outs_by_src = {}
    ins_by_tgt = {}
    for link in valid_links:
        outs_by_src.setdefault(link[0], []).append(link)
        ins_by_tgt.setdefault(link[1], []).append(link)

    link_y0 = {}
    node_out_pos = {node: node_positions_rect[node][1] for node in node_positions_rect}
    for src, outs in outs_by_src.items():
        for s, t, v in sorted(outs, key=lambda l: _yc(l[1])):
            link_y0[(s, t)] = node_out_pos[s]
            node_out_pos[s] += v * node_out_scale.get(s, 1)

    link_y1 = {}
    node_in_pos = {node: node_positions_rect[node][1] for node in node_positions_rect}
    for tgt, ins in ins_by_tgt.items():
        for s, t, v in sorted(ins, key=lambda l: _yc(l[0])):
            link_y1[(s, t)] = node_in_pos[t]
            node_in_pos[t] += v * node_in_scale.get(t, 1)

    for src, tgt, val in sorted(valid_links, key=lambda x: (x[0], x[1])):
        link_height_src = val * node_out_scale.get(src, 1)
        link_height_tgt = val * node_in_scale.get(tgt, 1)

        src_x, _src_y, _src_h = node_positions_rect[src]
        x0 = src_x + node_width
        y0_bot = link_y0[(src, tgt)]
        y0_top = y0_bot + link_height_src

        tgt_x, _tgt_y, _tgt_h = node_positions_rect[tgt]
        x1 = tgt_x
        y1_bot = link_y1[(src, tgt)]
        y1_top = y1_bot + link_height_tgt

        rgb_start = _node_rgb(src)
        rgb_end = _node_rgb(tgt)

        seg_x0 = x0 + (x1 - x0) * t0
        seg_x1 = x0 + (x1 - x0) * t1
        # Bézier via precomputed ease factors (same result as _bezier_vec)
        seg_y0_top = y0_top + (y1_top - y0_top) * f0
        seg_y0_bot = y0_bot + (y1_bot - y0_bot) * f0
        seg_y1_top = y0_top + (y1_top - y0_top) * f1
        seg_y1_bot = y0_bot + (y1_bot - y0_bot) * f1

        colors = rgb_start + np.outer(t_mid, rgb_end - rgb_start)

        # (n_segments, 4, 2): the 4 corners per segment, same order as before
        # (bottom-left, top-left, top-right, bottom-right).
        verts = np.stack([
            np.column_stack((seg_x0, seg_y0_bot)),
            np.column_stack((seg_x0, seg_y0_top)),
            np.column_stack((seg_x1, seg_y1_top)),
            np.column_stack((seg_x1, seg_y1_bot)),
        ], axis=1)
        rgba = np.empty((n_segments, 4))
        rgba[:, :3] = colors
        rgba[:, 3] = link_alpha
        vert_blocks.append(verts)
        color_blocks.append(rgba)

    if vert_blocks:
        all_vertices = np.concatenate(vert_blocks, axis=0)   # (N, 4, 2)
        all_colors = np.concatenate(color_blocks, axis=0)    # (N, 4) RGBA
        # Glow: wide translucent layers behind the links (neon effect on dark bg).
        # The glow color is value-invariant across layers -> compute once.
        if link_glow:
            glow_colors = all_colors.copy()
            glow_colors[:, 3] = link_alpha * 0.12
            for g in range(link_glow, 0, -1):
                ax.add_collection(PolyCollection(all_vertices, facecolors=glow_colors,
                                                 edgecolors=glow_colors, linewidths=2.5 * g))
        ax.add_collection(PolyCollection(all_vertices, facecolors=all_colors, edgecolors='none'))

    # Nodes + labels
    min_height_for_inside_text = 0.5
    for layer_idx, layer_nodes in enumerate(layers):
        for node in layer_nodes:
            if node not in node_positions_rect:
                continue

            x, y, h = node_positions_rect[node]
            color = _node_hex(node)

            rect = mpatches.FancyBboxPatch(
                (x, y), node_width, h,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=color, edgecolor=node_edge_color, linewidth=1.5
            )
            ax.add_patch(rect)

            val = frame_data.node_values.get(node, 0)
            text_color = get_text_color_for_background(color)

            if frame_data.node_value_labels and node in frame_data.node_value_labels:
                val_text = frame_data.node_value_labels[node]
            else:
                val_text = f"{val:.0f}"

            fits = h >= min_height_for_inside_text
            # If the bar is too small, append the value to the NAME (outside the
            # bar) so negatives like "(5)" are never hidden.
            name_txt = f"{node}" if fits else f"{node}  {val_text}"

            if node == yaxis_node:
                # Node with the value axis: name goes on top to free the left side
                ax.text(x + node_width / 2, y + h + 0.15, name_txt,
                        ha='center', va='bottom', fontsize=font_size, fontweight='bold',
                        color=label_color)
            elif layer_idx == 0:
                ax.text(x - 0.15, y + h / 2, name_txt,
                        ha='right', va='center', fontsize=font_size, fontweight='bold',
                        color=label_color)
            elif layer_idx == n_layers - 1:
                ax.text(x + node_width + 0.15, y + h / 2, name_txt,
                        ha='left', va='center', fontsize=font_size, fontweight='bold',
                        color=label_color)
            else:
                ax.text(x + node_width / 2, y + h + 0.15, name_txt,
                        ha='center', va='bottom', fontsize=font_size - 1, fontweight='bold',
                        color=label_color)

            if fits:
                ax.text(x + node_width / 2, y + h / 2, val_text,
                        ha='center', va='center', fontsize=font_size - 1,
                        color=text_color, fontweight='bold')

    # Dynamic value Y-axis on a chosen node (discreet ruler with "nice" ticks)
    if yaxis_node and yaxis_node in node_positions_rect and frame_data.node_values.get(yaxis_node, 0) > 0:
        yx, yyb, yh = node_positions_rect[yaxis_node]
        yval = frame_data.node_values[yaxis_node]
        ysuf = cfg.get('yaxis_suffix', '')
        raw = yval / 4.0
        mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1.0
        step = next((m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag), 10 * mag)
        axis_x = yx - 0.14
        ax.plot([axis_x, axis_x], [yyb, yyb + yh],
                color=label_color, lw=1.0, alpha=0.40, zorder=3)
        tv = 0.0
        while tv <= yval + 1e-9:
            ty = yyb + yh * (tv / yval)
            ax.plot([axis_x - 0.07, axis_x], [ty, ty],
                    color=label_color, lw=1.0, alpha=0.40, zorder=3)
            if abs(tv) < 1e-9:
                lab = "$0"
            elif tv >= 10:
                lab = f"${tv:.0f}{ysuf}"
            else:
                lab = f"${tv:.1f}{ysuf}"
            ax.text(axis_x - 0.11, ty, lab, color=label_color, fontsize=font_size - 3,
                    alpha=0.55, ha='right', va='center', zorder=3)
            tv += step

    # Bar-chart-race style overlay (footer): mini area chart + discreet big number
    overlay_series = cfg.get('overlay_series')
    if overlay_series:
        ov_color = cfg.get('overlay_color', '#33E08A')
        ov_label = cfg.get('overlay_label', '')
        ov_suffix = cfg.get('overlay_value_suffix', '')
        ov_badge = cfg.get('overlay_badge', '')
        n_ov = len(overlay_series)
        footer_top = plot_height * margin_bottom_ratio

        # Aligned horizontally with the Sankey (from first layer to the last node)
        bx0 = layer_x_positions[0]
        bx1 = layer_x_positions[n_layers - 1]
        by0, by1 = footer_top * 0.34, footer_top * 0.80

        p = max(0.0, min(frame_data.progress, n_ov - 1))
        k = int(np.floor(p))
        if k < n_ov - 1:
            frac = p - k
            cur_val = overlay_series[k] + (overlay_series[k + 1] - overlay_series[k]) * frac
        else:
            cur_val = overlay_series[-1]

        # "Bar chart race" style: the curve ALWAYS fills the width (fixed size), the
        # X (time) axis evolves, the Y axis is a running max. "Now" sits at the right.
        def _sx(idx):
            return bx0 + (bx1 - bx0) * (idx / p if p > 0 else 1.0)

        seen = overlay_series[:k + 1] + [cur_val]
        smax = max(seen) or 1

        def _sy(v):
            return by0 + (by1 - by0) * (max(v, 0) / smax)

        xs = [_sx(j) for j in range(k + 1)] + [bx1]
        ys = [_sy(overlay_series[j]) for j in range(k + 1)] + [_sy(cur_val)]
        cx, cy = bx1, _sy(cur_val)

        ax.plot([bx0, bx1], [by0, by0], color=label_color, lw=0.8, alpha=0.22, zorder=5)
        ax.fill_between(xs, by0, ys, color=ov_color, alpha=0.16, zorder=6)
        ax.plot(xs, ys, color=ov_color, lw=2.2, alpha=0.95, zorder=7)
        ax.scatter([cx], [cy], s=46, color='#FFFFFF', zorder=8)
        ax.scatter([cx], [cy], s=20, color=ov_color, zorder=9)
        if ov_label:
            ax.text(bx0, by1 + footer_top * 0.16, ov_label, color=label_color,
                    fontsize=font_size - 2, alpha=0.6, va='bottom', ha='left')
        ytop = (f"${smax:.0f}{ov_suffix}" if smax >= 10 else f"${smax:.1f}{ov_suffix}")
        ax.text(bx0 - plot_width * 0.006, by1, ytop, color=ov_color,
                fontsize=font_size - 3, alpha=0.5, va='center', ha='right')

        # Time (X) axis: mark the first quarter of each year seen so far
        xlabels = cfg.get('overlay_x_labels')
        if xlabels:
            prev_year = None
            for j in range(k + 1):
                yr = str(xlabels[j]).split()[0]
                if yr != prev_year:
                    prev_year = yr
                    tx = _sx(j)
                    ax.plot([tx, tx], [by0, by0 - footer_top * 0.06],
                            color=label_color, lw=0.8, alpha=0.30, zorder=5)
                    ax.text(tx, by0 - footer_top * 0.10, yr, color=label_color,
                            fontsize=font_size - 4, alpha=0.5, ha='center', va='top', zorder=5)

        # Big number anchoring the bottom-right corner + optional badge
        title_fontsize = cfg.get('title_fontsize', 18)
        big = (f"${cur_val:.0f}{ov_suffix}" if cur_val >= 10 else f"${cur_val:.1f}{ov_suffix}")
        ax.text(plot_width * 0.985, footer_top * 0.42, big,
                color=ov_color, fontsize=title_fontsize * 2.4, fontweight='bold',
                alpha=0.26, ha='right', va='center', zorder=4)
        if ov_badge:
            ax.text(plot_width * 0.985, footer_top * 0.88, ov_badge,
                    color=label_color, fontsize=font_size - 1, fontweight='bold',
                    alpha=0.55, ha='right', va='center', zorder=4)


def render_frame_high_quality(ax, frame_data, layers, node_colors,
                              plot_width, plot_height, node_width, font_size,
                              n_segments, bar_height_ratio, stacked_mode,
                              label_color='#000000', node_edge_color='#333333',
                              link_alpha=0.7, link_glow=0,
                              padding=1.2, yaxis_node=None, yaxis_suffix=""):
    """
    Render a single high-quality Sankey frame onto ``ax`` (no title).

    Thin wrapper over the shared :func:`_draw_frame` so a static frame is drawn
    by exactly the same code path as a video frame.

    Args:
        ax: Matplotlib axes.
        frame_data: A :class:`FrameData` with links, positions, values.
        layers: List of node layers (left -> right).
        node_colors: Mapping node -> hex color.
        plot_width, plot_height: Plot extents in data coordinates.
        node_width: Node rectangle width.
        font_size: Base font size.
        n_segments: Gradient segments per link.
        bar_height_ratio: Fraction of usable height used by stacked bars.
        stacked_mode: If True, node heights scale with value.
        label_color: Node name label color.
        node_edge_color: Node border color.
        link_alpha: Base link opacity (0-1).
        link_glow: Number of neon glow layers behind the links (0 = none).
        padding: Horizontal padding around the layers.
        yaxis_node: Optional node to draw a dynamic value axis next to.
        yaxis_suffix: Suffix for the value-axis tick labels (e.g. "B").
    """
    n_layers = len(layers)
    t = np.linspace(0, 1, n_segments + 1)
    t0 = t[:-1]
    t1 = t[1:]
    t_mid = (t0 + t1) / 2
    f0 = _bezier_factor(t0)
    f1 = _bezier_factor(t1)

    layer_spacing = (plot_width - 2 * padding - node_width) / max(1, n_layers - 1)
    layer_x_positions = {}
    for layer_idx in range(n_layers):
        if n_layers == 1:
            layer_x_positions[layer_idx] = plot_width / 2 - node_width / 2
        else:
            layer_x_positions[layer_idx] = padding + layer_idx * layer_spacing

    cfg = {
        'layers': layers,
        'node_colors': node_colors,
        'n_segments': n_segments,
        'plot_width': plot_width,
        'plot_height': plot_height,
        'node_width': node_width,
        'font_size': font_size,
        'bar_height_ratio': bar_height_ratio,
        'margin_top_ratio': 0.12,
        'margin_bottom_ratio': 0.05,
        'stacked_mode': stacked_mode,
        'label_color': label_color,
        'node_edge_color': node_edge_color,
        'link_alpha': link_alpha,
        'link_glow': link_glow,
        'yaxis_node': yaxis_node,
        'yaxis_suffix': yaxis_suffix,
        '_t0': t0, '_t1': t1, '_t_mid': t_mid, '_f0': f0, '_f1': f1,
        '_layer_x_positions': layer_x_positions,
    }
    _draw_frame(ax, frame_data, cfg)


# =============================================================================
# Worker function - renders a chunk of frames in a separate process
# =============================================================================

def _render_chunk(args):
    """Render a chunk of frames in a separate process and pipe them to FFmpeg.

    Uses the shared :func:`_draw_frame` so every video frame matches a static one.
    Always closes the figure and the FFmpeg process, and raises a descriptive
    error (instead of silently emitting a truncated file) if FFmpeg fails.
    """
    (chunk_id, frames_data, config) = args

    # Import matplotlib inside the worker (isolated per process)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figsize = config['figsize']
    fps = config['fps']
    bitrate = config['bitrate']
    title = config['title']
    layers = config['layers']
    temp_dir = config['temp_dir']
    n_segments = config['n_segments']

    plot_width = config['plot_width']
    plot_height = config['plot_height']
    node_width = config['node_width']
    padding = config['padding']
    title_fontsize = config['title_fontsize']
    title_bg_color = config['title_bg_color']
    title_bg_alpha = config['title_bg_alpha']

    bg_color = config.get('bg_color', 'white')
    title_text_color = config.get('title_text_color', '#000000')

    # Precompute frame-invariant gradient params and layer X positions once
    n_layers = len(layers)
    t = np.linspace(0, 1, n_segments + 1)
    t0 = t[:-1]
    t1 = t[1:]
    t_mid = (t0 + t1) / 2

    layer_spacing = (plot_width - 2 * padding - node_width) / max(1, n_layers - 1)
    layer_x_positions = {}
    for layer_idx in range(n_layers):
        if n_layers == 1:
            layer_x_positions[layer_idx] = plot_width / 2 - node_width / 2
        else:
            layer_x_positions[layer_idx] = padding + layer_idx * layer_spacing

    draw_cfg = dict(config)
    draw_cfg['_t0'] = t0
    draw_cfg['_t1'] = t1
    draw_cfg['_t_mid'] = t_mid
    draw_cfg['_f0'] = _bezier_factor(t0)
    draw_cfg['_f1'] = _bezier_factor(t1)
    draw_cfg['_layer_x_positions'] = layer_x_positions

    chunk_path = os.path.join(temp_dir, f"chunk_{chunk_id:04d}.mp4")

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    canvas_width, canvas_height = fig.canvas.get_width_height()

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{canvas_width}x{canvas_height}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-b:v', bitrate,
        '-preset', 'fast',
        chunk_path,
    ]

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        for frame_data in frames_data:
            ax.clear()
            ax.set_xlim(0, plot_width)
            ax.set_ylim(0, plot_height)
            ax.set_aspect('auto')
            ax.axis('off')
            ax.set_facecolor(bg_color)

            _draw_frame(ax, frame_data, draw_cfg)

            # Title (drawn by the caller path so the static/video titles can differ)
            display_title = title if title else ""
            ax.text(plot_width / 2, plot_height * 0.95,
                    f"{display_title}\n{frame_data.time_label}" if display_title else frame_data.time_label,
                    ha='center', va='top', fontsize=title_fontsize, fontweight='bold',
                    color=title_text_color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=title_bg_color, alpha=title_bg_alpha))

            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            img_array = np.asarray(buf)
            rgb_array = np.ascontiguousarray(img_array[:, :, :3])
            process.stdin.write(rgb_array.tobytes())

        process.stdin.close()
        _stdout, stderr = process.communicate(timeout=300)
        if process.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed for chunk {chunk_id} (exit {process.returncode}):\n"
                f"{stderr.decode(errors='ignore')[-800:]}"
            )
    except BaseException:
        # Never leak the subprocess or the figure on error
        try:
            process.kill()
        except Exception:
            pass
        try:
            process.wait(timeout=10)
        except Exception:
            pass
        raise
    finally:
        plt.close(fig)

    return chunk_id, chunk_path, len(frames_data)


# =============================================================================
# Main class
# =============================================================================

class SankeyRaceMultiLayerParallel:
    """
    Multi-layer Sankey "race" with PARALLEL rendering.

    Combines high visual quality, parallel multiprocessing video rendering, and
    all palettes/modes (ranking, stacked, stacked+ranking, fixed).
    """

    def __init__(self,
                 layers: List[List[str]],
                 node_colors: Dict[str, str]):
        if not layers or not any(layers):
            raise ValueError("`layers` must be a non-empty list of non-empty node lists.")
        for i, layer in enumerate(layers):
            if not layer:
                raise ValueError(f"Layer {i} is empty; every layer must have at least one node.")

        # Node names must be unique across ALL layers (they key a single position dict)
        all_nodes = [node for layer in layers for node in layer]
        if len(all_nodes) != len(set(all_nodes)):
            dups = sorted({n for n in all_nodes if all_nodes.count(n) > 1})
            raise ValueError(
                "Node names must be unique across all layers; duplicates found: "
                f"{dups}. Rename them (e.g. prefix by layer)."
            )

        self.layers = layers
        self.node_colors = node_colors
        self.frames = []

        self.node_layer = {}
        for layer_idx, layer_nodes in enumerate(layers):
            for node in layer_nodes:
                self.node_layer[node] = layer_idx

        self._rgb_cache = {}
        for node, color in node_colors.items():
            self._rgb_cache[node] = get_rgb_cached(color)

        self._global_max_cache = None

    @classmethod
    def from_dataframe(cls,
                       df: pd.DataFrame,
                       layers: List[List[str]],
                       time_col: str = "tempo",
                       source_col: str = "origem",
                       target_col: str = "destino",
                       value_col: str = "valor",
                       layer_palettes: Optional[List[Union[ColorPalette, str]]] = None,
                       node_colors: Optional[Dict[str, str]] = None) -> 'SankeyRaceMultiLayerParallel':
        """Create an instance from a tidy DataFrame (one row per flow).

        The DataFrame must contain ``time_col``, ``source_col``, ``target_col`` and
        ``value_col``. Node names in ``layers`` must be unique across all layers.
        Link values should be magnitudes (>= 0); negatives are dropped at render
        time, so pass ``abs(value)`` and use ``node_value_labels`` for accounting
        parentheses if you need to show signed figures.
        """
        if df is None or len(df) == 0:
            raise ValueError("`df` is empty; nothing to render.")

        missing = [c for c in (time_col, source_col, target_col, value_col) if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing required column(s) {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        if df[value_col].isna().any():
            raise ValueError(f"`{value_col}` contains NaN values; clean or fill them first.")
        if (pd.to_numeric(df[value_col], errors='coerce') < 0).any():
            warnings.warn(
                f"`{value_col}` contains negative values; links with value <= 0 are not "
                "drawn. Pass magnitudes (abs) and use node_value_labels for signs.",
                stacklevel=2,
            )

        if node_colors is None:
            node_colors = {}

            if layer_palettes is None:
                default_palettes = [
                    ColorPalette.RAINBOW, ColorPalette.OCEAN, ColorPalette.SUNSET,
                    ColorPalette.VIRIDIS, ColorPalette.PLASMA, ColorPalette.EARTH,
                    ColorPalette.NEON, ColorPalette.PASTEL,
                ]
                layer_palettes = [default_palettes[i % len(default_palettes)] for i in range(len(layers))]

            for layer_idx, layer_nodes in enumerate(layers):
                palette = layer_palettes[layer_idx] if layer_idx < len(layer_palettes) else ColorPalette.RAINBOW
                colors = get_palette_colors(palette, len(layer_nodes))
                for i, node in enumerate(layer_nodes):
                    node_colors[node] = colors[i % len(colors)]

        instance = cls(layers=layers, node_colors=node_colors)

        # Build frames (vectorized via .values)
        for time_val in sorted(df[time_col].unique()):
            df_t = df[df[time_col] == time_val]
            src_arr = df_t[source_col].values
            tgt_arr = df_t[target_col].values
            val_arr = df_t[value_col].values
            links = list(zip(src_arr, tgt_arr, val_arr))
            instance.frames.append({'time_label': str(time_val), 'links': links})

        instance._global_max_cache = instance._compute_global_max()

        return instance

    def _compute_node_sums(self, links):
        out_sums = {}
        in_sums = {}
        for src, tgt, val in links:
            out_sums[src] = out_sums.get(src, 0) + val
            in_sums[tgt] = in_sums.get(tgt, 0) + val
        return out_sums, in_sums

    def _compute_node_values(self, links):
        out_sums, in_sums = self._compute_node_sums(links)
        node_values = {}
        for layer_nodes in self.layers:
            for node in layer_nodes:
                node_values[node] = max(out_sums.get(node, 0), in_sums.get(node, 0))
        return node_values

    def _compute_global_max(self):
        """Compute the global maximum node throughput across all frames."""
        if not self.frames:
            return 1

        max_val = 0
        for frame in self.frames:
            node_totals = {}
            for s, t, v in frame['links']:
                node_totals[s] = node_totals.get(s, 0) + v
                node_totals[t] = node_totals.get(t, 0) + v
            if node_totals:
                max_val = max(max_val, max(node_totals.values()))

        return max_val if max_val > 0 else 1

    def _get_global_max(self):
        if self._global_max_cache is None:
            self._global_max_cache = self._compute_global_max()
        return self._global_max_cache

    def _compute_ranking_positions(self, node_values, plot_height, margin_top_ratio,
                                    margin_bottom_ratio, ascending):
        """Compute Y positions by per-layer ranking (no stacking)."""
        margin_top = plot_height * margin_top_ratio
        margin_bottom = plot_height * margin_bottom_ratio
        usable_height = plot_height - margin_top - margin_bottom

        positions = {}
        for layer_nodes in self.layers:
            n_nodes = len(layer_nodes)
            if n_nodes == 0:
                continue

            node_spacing = usable_height / n_nodes
            sorted_nodes = sorted(layer_nodes, key=lambda n: node_values.get(n, 0), reverse=not ascending)

            for i, node in enumerate(sorted_nodes):
                y_center = margin_bottom + node_spacing * (n_nodes - 1 - i + 0.5)
                positions[node] = y_center

        return positions

    def _compute_stacked_positions(self, node_values, plot_height, bar_height_ratio,
                                   margin_top_ratio, margin_bottom_ratio, gap,
                                   fixed_max=None):
        """Compute Y positions in stacked mode (no ranking).

        fixed_max: if given, use this fixed GLOBAL scale instead of the frame max,
        so bars grow in absolute value between frames (e.g. revenue "explosion").
        """
        margin_top = plot_height * margin_top_ratio
        margin_bottom = plot_height * margin_bottom_ratio
        usable_height = plot_height - margin_top - margin_bottom

        if fixed_max is not None:
            max_layer_total = fixed_max
        else:
            layer_totals = [sum(node_values.get(node, 0) for node in layer_nodes)
                            for layer_nodes in self.layers]
            max_layer_total = max(layer_totals) if layer_totals else 1
        stacked_height = usable_height * bar_height_ratio
        scale = stacked_height / max_layer_total if max_layer_total > 0 else 1

        positions = {}
        for layer_nodes in self.layers:
            n_nodes = len(layer_nodes)
            if n_nodes == 0:
                continue

            # Keep original order (no sorting)
            heights = [max(node_values.get(node, 0) * scale, 0.1) for node in layer_nodes]
            total_height = sum(heights) + gap * (n_nodes - 1) if n_nodes > 1 else sum(heights)
            start_y = margin_bottom + (usable_height - total_height) / 2 + total_height

            current_y = start_y
            for i, node in enumerate(layer_nodes):
                h = heights[i]
                current_y -= h
                positions[node] = current_y + h / 2
                current_y -= gap

        return positions

    def _compute_fixed_positions(self, node_values, plot_height,
                                 margin_top_ratio, margin_bottom_ratio):
        """Compute fixed Y positions (no ranking, no stacking)."""
        margin_top = plot_height * margin_top_ratio
        margin_bottom = plot_height * margin_bottom_ratio
        usable_height = plot_height - margin_top - margin_bottom

        max_nodes_in_layer = max(len(layer) for layer in self.layers)
        available_height_per_slot = usable_height / max(1, max_nodes_in_layer)

        positions = {}
        for layer_nodes in self.layers:
            n_nodes = len(layer_nodes)
            if n_nodes == 0:
                continue

            for i, node in enumerate(layer_nodes):
                y_center = margin_bottom + available_height_per_slot * (n_nodes - 1 - i + 0.5)
                positions[node] = y_center

        return positions

    def _compute_dynamic_colors(self, node_values: Dict[str, float],
                                 mode: Union[DynamicColorMode, str] = DynamicColorMode.RANKING,
                                 colormap: Union[ColorPalette, str, List[str]] = 'RdYlGn',
                                 global_min: float = None,
                                 global_max: float = None) -> Dict[str, str]:
        """
        Compute dynamic colors for nodes based on their values.

        Args:
            node_values: Dict mapping node names to their values.
            mode: Coloring mode (RANKING, VALUE, GLOBAL_VALUE, PERCENTILE, INTENSITY).
            colormap: ColorPalette enum, matplotlib colormap name, or list of hex colors.
            global_min: Min value for global normalization.
            global_max: Max value for global normalization.

        Returns:
            Dict mapping node names to hex color strings.
        """
        if isinstance(mode, str):
            mode = DynamicColorMode(mode.lower())

        colors = {}

        if mode == DynamicColorMode.STATIC:
            return self.node_colors.copy()

        elif mode == DynamicColorMode.RANKING:
            for layer in self.layers:
                layer_values = [(node, node_values.get(node, 0)) for node in layer]
                sorted_nodes = sorted(layer_values, key=lambda x: -x[1])
                for rank, (node, _) in enumerate(sorted_nodes, 1):
                    colors[node] = get_ranking_color(rank, len(layer), colormap)

        elif mode == DynamicColorMode.VALUE:
            for layer in self.layers:
                layer_vals = [node_values.get(node, 0) for node in layer]
                min_val = min(layer_vals) if layer_vals else 0
                max_val = max(layer_vals) if layer_vals else 1
                for node in layer:
                    val = node_values.get(node, 0)
                    colors[node] = get_dynamic_color(val, min_val, max_val, colormap)

        elif mode == DynamicColorMode.GLOBAL_VALUE:
            if global_min is None:
                global_min = min(node_values.values()) if node_values else 0
            if global_max is None:
                global_max = max(node_values.values()) if node_values else 1
            for node in node_values:
                colors[node] = get_dynamic_color(node_values[node], global_min, global_max, colormap)

        elif mode == DynamicColorMode.PERCENTILE:
            for layer in self.layers:
                layer_vals = sorted([node_values.get(node, 0) for node in layer])
                n = len(layer_vals)
                for node in layer:
                    val = node_values.get(node, 0)
                    rank = sum(1 for v in layer_vals if v <= val)
                    percentile = rank / n if n > 0 else 0.5
                    colors[node] = get_dynamic_color(percentile, 0, 1, colormap)

        elif mode == DynamicColorMode.INTENSITY:
            # Keep each node's base hue (self.node_colors), scale brightness by value
            # normalized GLOBALLY (sqrt-compressed so small early values stay visible).
            if global_max is None:
                global_max = max(node_values.values()) if node_values else 1
            gmax = global_max if global_max > 0 else 1
            for node in node_values:
                base = self.node_colors.get(node, '#888888')
                factor = (max(0.0, node_values[node]) / gmax) ** 0.5
                colors[node] = scale_brightness(base, factor)

        return colors

    def _compute_stacked_ranking_positions(self, node_values, plot_height, bar_height_ratio,
                                           margin_top_ratio, margin_bottom_ratio, gap, ascending):
        """Compute Y positions in stacked + ranking mode."""
        margin_top = plot_height * margin_top_ratio
        margin_bottom = plot_height * margin_bottom_ratio
        usable_height = plot_height - margin_top - margin_bottom

        layer_totals = [sum(node_values.get(node, 0) for node in layer_nodes)
                        for layer_nodes in self.layers]
        max_layer_total = max(layer_totals) if layer_totals else 1
        stacked_height = usable_height * bar_height_ratio
        scale = stacked_height / max_layer_total if max_layer_total > 0 else 1

        positions = {}
        for layer_nodes in self.layers:
            n_nodes = len(layer_nodes)
            if n_nodes == 0:
                continue

            sorted_nodes = sorted(layer_nodes, key=lambda n: node_values.get(n, 0), reverse=not ascending)
            heights = [max(node_values.get(node, 0) * scale, 0.1) for node in sorted_nodes]
            total_height = sum(heights) + gap * (n_nodes - 1) if n_nodes > 1 else sum(heights)
            start_y = margin_bottom + (usable_height - total_height) / 2 + total_height

            current_y = start_y
            for i, node in enumerate(sorted_nodes):
                h = heights[i]
                current_y -= h
                positions[node] = current_y + h / 2
                current_y -= gap

        return positions

    def _precompute_frames(self, total_frames, plot_height, bar_height_ratio,
                           margin_top_ratio, margin_bottom_ratio, stacked_gap, ascending,
                           mode='stacked_ranking',
                           dynamic_color_mode: Union[DynamicColorMode, str] = DynamicColorMode.STATIC,
                           dynamic_colormap: Union[ColorPalette, str, List[str]] = 'RdYlGn',
                           node_value_labels_per_frame: Optional[List[Dict[str, str]]] = None,
                           fixed_layer_max: Optional[float] = None):
        """Pre-compute every interpolated frame.

        Args:
            mode: 'stacked_ranking', 'ranking', 'stacked', or 'fixed'.
            dynamic_color_mode: Dynamic node coloring mode.
            dynamic_colormap: Colormap for dynamic colors.
        """
        n_data_frames = len(self.frames)
        frames_per_period = max(1, total_frames // max(1, n_data_frames))

        if isinstance(dynamic_color_mode, str):
            dynamic_color_mode = DynamicColorMode(dynamic_color_mode.lower())

        use_dynamic_colors = dynamic_color_mode != DynamicColorMode.STATIC

        # Pre-compute global min/max for GLOBAL_VALUE / INTENSITY modes
        global_min_val = None
        global_max_val = None
        if dynamic_color_mode in (DynamicColorMode.GLOBAL_VALUE, DynamicColorMode.INTENSITY):
            all_values = []
            for frame in self.frames:
                values = self._compute_node_values(frame['links'])
                all_values.extend(values.values())
            if all_values:
                global_min_val = min(all_values)
                global_max_val = max(all_values)

        # Per data-frame values / positions / colors
        data_info = []
        for frame in self.frames:
            values = self._compute_node_values(frame['links'])

            if mode == 'stacked_ranking':
                positions = self._compute_stacked_ranking_positions(
                    values, plot_height, bar_height_ratio,
                    margin_top_ratio, margin_bottom_ratio, stacked_gap, ascending
                )
            elif mode == 'ranking':
                positions = self._compute_ranking_positions(
                    values, plot_height, margin_top_ratio, margin_bottom_ratio, ascending
                )
            elif mode == 'stacked':
                positions = self._compute_stacked_positions(
                    values, plot_height, bar_height_ratio,
                    margin_top_ratio, margin_bottom_ratio, stacked_gap,
                    fixed_max=fixed_layer_max
                )
            else:  # fixed
                positions = self._compute_fixed_positions(
                    values, plot_height, margin_top_ratio, margin_bottom_ratio
                )

            if use_dynamic_colors:
                colors = self._compute_dynamic_colors(
                    values, dynamic_color_mode, dynamic_colormap,
                    global_min_val, global_max_val
                )
            else:
                colors = None

            data_info.append((frame, values, positions, colors))

        interpolated = []

        for i in range(n_data_frames - 1):
            frame_a, values_a, pos_a, colors_a = data_info[i]
            frame_b, values_b, pos_b, colors_b = data_info[i + 1]

            links_a = {(s, d): v for s, d, v in frame_a['links']}
            links_b = {(s, d): v for s, d, v in frame_b['links']}
            all_link_keys = set(links_a.keys()) | set(links_b.keys())
            all_nodes = set(pos_a.keys()) | set(pos_b.keys())

            for j in range(frames_per_period):
                t = j / frames_per_period

                interp_links = []
                for key in all_link_keys:
                    va = links_a.get(key, 0)
                    vb = links_b.get(key, 0)
                    v = va + (vb - va) * t
                    # Keep every real link. The old absolute 0.1 cutoff silently
                    # dropped small-magnitude flows (e.g. $B financial data where a
                    # 0.085 flow is real), leaving nodes with missing in/out arrows;
                    # a link fading to 0 just shrinks smoothly, so only skip ~0.
                    if v > 1e-9:
                        interp_links.append((key[0], key[1], v))

                interp_pos = {}
                for node in all_nodes:
                    pa = pos_a.get(node, 0.5)
                    pb = pos_b.get(node, pa)
                    interp_pos[node] = pa + (pb - pa) * t

                interp_values = {}
                for node in all_nodes:
                    va = values_a.get(node, 0)
                    vb = values_b.get(node, va)
                    interp_values[node] = va + (vb - va) * t

                interp_colors = None
                if use_dynamic_colors and colors_a and colors_b:
                    interp_colors = {}
                    for node in all_nodes:
                        ca = colors_a.get(node, '#888888')
                        cb = colors_b.get(node, '#888888')
                        interp_colors[node] = interpolate_color(ca, cb, t)

                interp_labels = None
                if node_value_labels_per_frame:
                    interp_labels = (node_value_labels_per_frame[i] if t < 0.5
                                     else node_value_labels_per_frame[i + 1])

                interpolated.append(FrameData(
                    time_label=frame_a['time_label'] if t < 0.5 else frame_b['time_label'],
                    links=interp_links,
                    node_positions=interp_pos,
                    node_values=interp_values,
                    node_colors=interp_colors,
                    node_value_labels=interp_labels,
                    progress=i + t
                ))

        # Hold frames at the end
        last_frame, last_values, last_pos, last_colors = data_info[-1]
        last_labels = node_value_labels_per_frame[-1] if node_value_labels_per_frame else None
        remaining = max(0, total_frames - len(interpolated))
        for _ in range(remaining):
            interpolated.append(FrameData(
                time_label=last_frame['time_label'],
                links=last_frame['links'],
                node_positions=last_pos,
                node_values=last_values,
                node_colors=last_colors,
                node_value_labels=last_labels,
                progress=float(n_data_frames - 1)
            ))

        return interpolated

    def animate(self,
                output_path: str = "sankey_parallel.mp4",
                title: str = None,
                figsize: Tuple[float, float] = (18, 10),
                fps: int = 30,
                duration_seconds: float = 10.0,
                quality: str = "medium",
                node_width: float = 0.5,
                padding: float = 1.2,
                font_size: int = 10,
                bar_height_ratio: float = 0.85,
                margin_top: float = 0.12,
                margin_bottom: float = 0.05,
                title_fontsize: int = 18,
                title_bg_color: str = "wheat",
                title_bg_alpha: float = 0.9,
                ranking_mode: bool = True,
                stacked_mode: bool = True,
                stacked_gap: float = 0.1,
                ascending: bool = False,
                n_workers: int = None,
                n_segments: int = DEFAULT_GRADIENT_SEGMENTS,
                dynamic_color_mode: Union[DynamicColorMode, str] = "static",
                dynamic_colormap: Union[ColorPalette, str, List[str]] = "RdYlGn",
                theme: str = "light",
                bg_color: str = None,
                label_color: str = None,
                node_edge_color: str = None,
                title_text_color: str = None,
                link_alpha: float = 0.7,
                link_glow: int = 0,
                node_value_labels_per_frame: Optional[List[Dict[str, str]]] = None,
                absolute_scale: bool = False,
                overlay_series: Optional[List[float]] = None,
                overlay_label: str = None,
                overlay_color: str = "#33E08A",
                overlay_value_suffix: str = "",
                overlay_x_labels: Optional[List[str]] = None,
                overlay_badge: str = "",
                audio_path: str = None,
                audio_url: str = None,
                audio_start: float = 0.0,
                audio_fade: float = 1.5,
                yaxis_node: str = None,
                yaxis_suffix: str = ""):
        """
        Render the animation using multiple parallel worker processes.

        Positioning modes:
            - ranking_mode=True,  stacked_mode=True  -> Stacked + Ranking (default):
              nodes reorder by value AND resize proportionally.
            - ranking_mode=True,  stacked_mode=False -> Ranking only:
              nodes reorder by value but have UNIFORM heights.
            - ranking_mode=False, stacked_mode=True  -> Stacked only:
              fixed node order, heights vary by value.
            - ranking_mode=False, stacked_mode=False -> Fixed:
              fixed node order AND uniform heights (only links animate).

        Args:
            dynamic_color_mode: "static", "ranking", "value", "global_value",
                "percentile", or "intensity".
            dynamic_colormap: ColorPalette enum, matplotlib colormap name, or list of hex.
            theme: "light" (default) or "dark" preset.
            link_glow: Neon glow layers behind links (0 = none; 1-2 on dark bg).
            absolute_scale: If True (stacked-only mode), bars use a fixed global scale
                so absolute growth is visible across frames.
            node_value_labels_per_frame: Per-data-frame {node: text} overrides (e.g.
                accounting parentheses for negatives).
            overlay_series: One value per data frame, drawn as a bar-chart-race style
                footer chart with an evolving "big number".
            overlay_label / overlay_color / overlay_value_suffix / overlay_x_labels /
            overlay_badge: Overlay styling. ``overlay_badge`` is the corner tag (e.g. a
                ticker like "NVDA"); empty by default.
            yaxis_node / yaxis_suffix: Draw an evolving "$" value axis next to a node.
            audio_path / audio_url: Background music (local MP3 or YouTube URL).
            audio_start / audio_fade: Music start offset (s) and fade in/out (s).
            n_workers: Number of processes (default: CPU count, capped at frame count).
            n_segments: Gradient segments per link (default: 50).
        """
        if not self.frames:
            raise ValueError("No frames added.")

        if absolute_scale and not (stacked_mode and not ranking_mode):
            warnings.warn(
                "absolute_scale only applies to stacked-only mode "
                "(ranking_mode=False, stacked_mode=True); ignoring it.",
                stacklevel=2,
            )

        # Audio via YouTube: resolve the URL to a local mp3 BEFORE rendering, so it
        # fails fast (without wasting the render) if something goes wrong.
        if audio_url and not audio_path:
            print(f"Downloading audio from YouTube: {audio_url}")
            audio_path = youtube_to_mp3(audio_url)
            print(f"  -> {audio_path}")

        # Theme presets (overridable by explicit kwargs)
        if theme == "dark":
            _preset = dict(bg_color="#0a0a12", label_color="#EAEAF2",
                           node_edge_color="#1a1a28", title_text_color="#FFFFFF",
                           title_bg_color="#15151f")
        else:  # light
            _preset = dict(bg_color="white", label_color="#000000",
                           node_edge_color="#333333", title_text_color="#000000",
                           title_bg_color=title_bg_color)
        bg_color = bg_color or _preset["bg_color"]
        label_color = label_color or _preset["label_color"]
        node_edge_color = node_edge_color or _preset["node_edge_color"]
        title_text_color = title_text_color or _preset["title_text_color"]
        title_bg_color = _preset["title_bg_color"] if theme == "dark" else title_bg_color

        # Determine mode
        if stacked_mode and ranking_mode:
            mode = 'stacked_ranking'
            mode_name = 'Stacked + Ranking'
        elif ranking_mode:
            mode = 'ranking'
            mode_name = 'Ranking'
        elif stacked_mode:
            mode = 'stacked'
            mode_name = 'Stacked'
        else:
            mode = 'fixed'
            mode_name = 'Fixed'

        # Quality settings
        quality_settings = {
            "low": {"dpi": 72, "bitrate": "1500k"},
            "medium": {"dpi": 120, "bitrate": "3000k"},
            "high": {"dpi": 200, "bitrate": "8000k"},
        }
        settings = quality_settings.get(quality, quality_settings["medium"])

        plot_width, plot_height = figsize
        total_frames = int(fps * duration_seconds)
        if total_frames < 1:
            raise ValueError("fps * duration_seconds must be >= 1 frame.")

        # Fixed global scale (absolute growth between frames), stacked-only mode
        fixed_layer_max = None
        if absolute_scale and mode == 'stacked':
            flm = 0.0
            for frame in self.frames:
                vals = self._compute_node_values(frame['links'])
                for layer_nodes in self.layers:
                    flm = max(flm, sum(vals.get(n, 0) for n in layer_nodes))
            fixed_layer_max = flm if flm > 0 else None

        if isinstance(dynamic_color_mode, str):
            dynamic_color_mode_enum = DynamicColorMode(dynamic_color_mode.lower())
        else:
            dynamic_color_mode_enum = dynamic_color_mode

        color_mode_name = dynamic_color_mode_enum.value.replace('_', ' ').title()

        print(f"Settings (MULTI-LAYER PARALLEL):")
        print(f"  - Layers: {len(self.layers)}")
        print(f"  - Nodes per layer: {[len(l) for l in self.layers]}")
        print(f"  - FPS: {fps}, Duration: {duration_seconds}s")
        print(f"  - Quality: {quality}")
        print(f"  - Total frames: {total_frames}")
        print(f"  - Gradient segments: {n_segments}")
        print(f"  - Positioning mode: {mode_name}")
        print(f"  - Dynamic color mode: {color_mode_name}")
        if dynamic_color_mode_enum != DynamicColorMode.STATIC:
            print(f"  - Colormap: {dynamic_colormap}")

        # Pre-compute frames
        print(f"\nPre-computing {total_frames} frames...")
        t0 = time.time()
        all_frames = self._precompute_frames(
            total_frames, plot_height, bar_height_ratio,
            margin_top, margin_bottom, stacked_gap, ascending,
            mode=mode,
            dynamic_color_mode=dynamic_color_mode_enum,
            dynamic_colormap=dynamic_colormap,
            node_value_labels_per_frame=node_value_labels_per_frame,
            fixed_layer_max=fixed_layer_max
        )
        print(f"  Pre-computation: {time.time() - t0:.2f}s")

        # Cap workers at the number of frames so no worker gets an empty chunk
        # (empty chunks produce zero-frame mp4s that corrupt the concat).
        if n_workers is None:
            n_workers = mp.cpu_count()
        n_workers = max(1, min(n_workers, len(all_frames)))

        # Split into balanced chunks (distribute the remainder)
        chunks = [list(c) for c in np.array_split(all_frames, n_workers)]
        chunks = [c for c in chunks if c]
        n_workers = len(chunks)
        print(f"  Workers: {n_workers}")

        temp_dir = tempfile.mkdtemp(prefix="sankey_multi_parallel_")

        # Shared config
        config = {
            'width': int(figsize[0] * settings['dpi']),
            'height': int(figsize[1] * settings['dpi']),
            'figsize': figsize,
            'fps': fps,
            'bitrate': settings['bitrate'],
            'title': title,
            'layers': self.layers,
            'node_colors': self.node_colors,
            'rgb_cache': self._rgb_cache,
            'node_layer': self.node_layer,
            'temp_dir': temp_dir,
            'n_segments': n_segments,
            'plot_width': plot_width,
            'plot_height': plot_height,
            'node_width': node_width,
            'padding': padding,
            'font_size': font_size,
            'bar_height_ratio': bar_height_ratio,
            'margin_top_ratio': margin_top,
            'margin_bottom_ratio': margin_bottom,
            'title_fontsize': title_fontsize,
            'title_bg_color': title_bg_color,
            'title_bg_alpha': title_bg_alpha,
            'stacked_mode': stacked_mode,
            'bg_color': bg_color,
            'label_color': label_color,
            'node_edge_color': node_edge_color,
            'title_text_color': title_text_color,
            'link_alpha': link_alpha,
            'link_glow': link_glow,
            'fixed_layer_max': fixed_layer_max,
            'overlay_series': overlay_series,
            'overlay_label': overlay_label,
            'overlay_color': overlay_color,
            'overlay_value_suffix': overlay_value_suffix,
            'overlay_x_labels': overlay_x_labels,
            'overlay_badge': overlay_badge,
            'yaxis_node': yaxis_node,
            'yaxis_suffix': yaxis_suffix,
        }

        worker_args = [(i, chunks[i], config) for i in range(n_workers)]

        print(f"\nRendering on {n_workers} parallel processes...")
        t0 = time.time()

        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_render_chunk, worker_args)

        render_time = time.time() - t0
        print(f"  Parallel rendering: {render_time:.2f}s ({total_frames / render_time:.1f} fps)")

        # Sort and concatenate
        results.sort(key=lambda x: x[0])
        chunk_paths = [r[1] for r in results]

        print(f"\nConcatenating {n_workers} chunks...")
        t0_concat = time.time()

        list_path = os.path.join(temp_dir, "chunks.txt")
        with open(list_path, 'w') as f:
            for path in chunk_paths:
                # FFmpeg concat demuxer: single quotes, escaped for safety
                safe = path.replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        # Concatenate the chunks into a (silent) video. If audio is requested, do it
        # in a temp file and then mux the track; otherwise write output_path directly.
        silent_path = os.path.join(temp_dir, "video_silent.mp4") if audio_path else output_path
        concat_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_path,
            '-c', 'copy',
            silent_path,
        ]

        result = subprocess.run(concat_cmd, capture_output=True)
        if result.returncode != 0 or not os.path.exists(silent_path):
            _rmtree_robust(temp_dir)
            raise RuntimeError(
                "FFmpeg concat failed:\n" + result.stderr.decode(errors='ignore')[-1000:]
            )

        print(f"  Concatenation: {time.time() - t0_concat:.2f}s")

        # Mux the audio track (MP3) if provided
        if audio_path:
            if not os.path.exists(audio_path):
                print(f"WARNING: audio not found ({audio_path}); saving video without sound.")
                shutil.copyfile(silent_path, output_path)
            else:
                video_dur = total_frames / fps
                fade_out_st = max(0.0, video_dur - audio_fade)
                afade = f"afade=t=in:st=0:d={audio_fade},afade=t=out:st={fade_out_st}:d={audio_fade}"
                mux_cmd = [
                    'ffmpeg', '-y',
                    '-i', silent_path,
                    '-ss', str(audio_start), '-i', audio_path,
                    '-map', '0:v', '-map', '1:a',
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                    '-af', afade,
                    '-t', str(video_dur), '-shortest',
                    output_path,
                ]
                mux = subprocess.run(mux_cmd, capture_output=True)
                if mux.returncode != 0:
                    print(f"Error muxing audio: {mux.stderr.decode(errors='ignore')[-800:]}")
                    shutil.copyfile(silent_path, output_path)
                else:
                    print(f"  Audio muxed: {os.path.basename(audio_path)} (start at {audio_start}s)")

        _rmtree_robust(temp_dir)

        if not os.path.exists(output_path):
            raise RuntimeError(f"Render finished but output was not produced: {output_path}")

        total_time = render_time + (time.time() - t0_concat)
        print(f"\nAnimation saved to: {output_path}")
        print(f"Total time: {total_time:.2f}s ({total_frames / total_time:.1f} effective fps)")

        return output_path

    def save_frame(self,
                   output_path: str = "sankey_frame.png",
                   frame_index: int = 0,
                   title: str = None,
                   figsize: Tuple[float, float] = (16, 10),
                   dpi: int = 150,
                   node_width: float = 0.5,
                   padding: float = 1.2,
                   font_size: int = 10,
                   bar_height_ratio: float = 0.85,
                   margin_top: float = 0.12,
                   margin_bottom: float = 0.05,
                   title_fontsize: int = 18,
                   title_bg_color: str = "wheat",
                   title_bg_alpha: float = 0.9,
                   ranking_mode: bool = True,
                   stacked_mode: bool = True,
                   stacked_gap: float = 0.1,
                   ascending: bool = False,
                   n_segments: int = DEFAULT_GRADIENT_SEGMENTS,
                   theme: str = "light",
                   bg_color: str = None,
                   label_color: str = None,
                   node_edge_color: str = None,
                   title_text_color: str = None,
                   link_alpha: float = 0.7,
                   link_glow: int = 0,
                   node_value_labels: Dict[str, str] = None,
                   yaxis_node: str = None,
                   yaxis_suffix: str = ""):
        """
        Save a single frame as a static image (PNG, PDF, SVG).

        Uses the same drawing code as :meth:`animate` (via the shared renderer), so a
        static frame matches a video frame exactly. Dynamic node colors and the
        time-series overlay are animation-only; the dynamic value axis (``yaxis_node``)
        and ``node_value_labels`` are supported here too.

        Args:
            output_path: Output path (.png, .pdf, .svg).
            frame_index: Index of the frame to save (0 = first).
            theme: "light" (default) or "dark" preset (overridable per-color kwarg).

        Returns:
            Path to the saved file.
        """
        if not self.frames:
            raise ValueError("No frames added.")

        if not 0 <= frame_index < len(self.frames):
            raise ValueError(
                f"frame_index {frame_index} is out of range. Total frames: {len(self.frames)}"
            )

        # Theme presets (overridable by explicit kwargs)
        if theme == "dark":
            preset = dict(bg_color="#0a0a12", label_color="#EAEAF2",
                          node_edge_color="#1a1a28", title_text_color="#FFFFFF",
                          title_bg_color="#15151f")
        else:  # light
            preset = dict(bg_color="white", label_color="#000000",
                          node_edge_color="#333333", title_text_color="#000000",
                          title_bg_color=title_bg_color)
        bg_color = bg_color or preset["bg_color"]
        label_color = label_color or preset["label_color"]
        node_edge_color = node_edge_color or preset["node_edge_color"]
        title_text_color = title_text_color or preset["title_text_color"]
        title_bg_color = preset["title_bg_color"] if theme == "dark" else title_bg_color

        # Determine mode
        if stacked_mode and ranking_mode:
            mode = 'stacked_ranking'
        elif ranking_mode:
            mode = 'ranking'
        elif stacked_mode:
            mode = 'stacked'
        else:
            mode = 'fixed'

        plot_width, plot_height = figsize

        frame = self.frames[frame_index]
        values = self._compute_node_values(frame['links'])

        if mode == 'stacked_ranking':
            positions = self._compute_stacked_ranking_positions(
                values, plot_height, bar_height_ratio,
                margin_top, margin_bottom, stacked_gap, ascending
            )
        elif mode == 'ranking':
            positions = self._compute_ranking_positions(
                values, plot_height, margin_top, margin_bottom, ascending
            )
        elif mode == 'stacked':
            positions = self._compute_stacked_positions(
                values, plot_height, bar_height_ratio,
                margin_top, margin_bottom, stacked_gap
            )
        else:
            positions = self._compute_fixed_positions(
                values, plot_height, margin_top, margin_bottom
            )

        frame_data = FrameData(
            time_label=frame['time_label'],
            links=frame['links'],
            node_positions=positions,
            node_values=values,
            node_value_labels=node_value_labels
        )

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(-padding, plot_width + padding)
        ax.set_ylim(0, plot_height)
        ax.axis('off')
        ax.set_facecolor(bg_color)
        fig.patch.set_facecolor(bg_color)

        render_frame_high_quality(
            ax, frame_data, self.layers, self.node_colors,
            plot_width, plot_height, node_width, font_size,
            n_segments, bar_height_ratio, mode in ['stacked_ranking', 'stacked'],
            label_color=label_color, node_edge_color=node_edge_color,
            link_alpha=link_alpha, link_glow=link_glow,
            padding=padding, yaxis_node=yaxis_node, yaxis_suffix=yaxis_suffix,
        )

        if title:
            time_label = frame_data.time_label
            full_title = f"{title}\n{time_label}" if time_label else title
            ax.text(
                plot_width / 2, plot_height * 0.98, full_title,
                fontsize=title_fontsize, ha='center', va='top',
                fontweight='bold', color=title_text_color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor=title_bg_color,
                          alpha=title_bg_alpha, edgecolor='none')
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor=bg_color, edgecolor='none')
        plt.close(fig)

        print(f"Frame saved to: {output_path}")
        return output_path


# Backward-compatible alias
SankeyRaceMultiLayer = SankeyRaceMultiLayerParallel


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    mp.freeze_support()

    print("=" * 70)
    print("TEST: Sankey Multi-Layer PARALLEL")
    print("=" * 70)

    exporters = ["Brazil", "USA", "China", "India"]
    importers = ["EU", "Japan", "Mexico", "Canada"]

    flows = {
        ("Brazil", "EU"): 10, ("Brazil", "Japan"): 8, ("Brazil", "Mexico"): 5,
        ("USA", "EU"): 15, ("USA", "Mexico"): 12, ("USA", "Canada"): 8,
        ("China", "EU"): 20, ("China", "Japan"): 18, ("China", "Mexico"): 10,
        ("India", "EU"): 6, ("India", "Japan"): 4, ("India", "Canada"): 3,
    }

    data = []
    for year in range(2020, 2024):
        mult = 1 + (year - 2020) * 0.1
        for (src, tgt), val in flows.items():
            data.append({"year": year, "source": src, "target": tgt, "value": val * mult})

    df = pd.DataFrame(data)
    layers = [exporters, importers]

    sankey = SankeyRaceMultiLayerParallel.from_dataframe(
        df=df, layers=layers,
        time_col="year", source_col="source", target_col="target", value_col="value",
    )

    sankey.animate(
        output_path="test_multi_parallel.mp4",
        title="Multi-Layer Parallel Test",
        figsize=(16, 10), fps=24, duration_seconds=5.0, quality="medium", n_workers=4,
    )

    try:
        os.remove("test_multi_parallel.mp4")
    except OSError:
        pass

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
