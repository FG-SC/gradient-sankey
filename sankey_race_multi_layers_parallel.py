"""
Sankey Race Multi-Layers PARALLEL - Máxima Qualidade + Paralelização
=====================================================================

Combina:
- Qualidade visual do sankey_race_multi_layers_otimizado (FancyBboxPatch, 50 segmentos)
- Renderização paralela com multiprocessing
- Todas as paletas de cores e modos (ranking, stacked, stacked+ranking)

Speedup esperado: ~Nx (onde N = número de workers)
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
import multiprocessing as mp
import tempfile
import shutil
import os


# =============================================================================
# Paletas de cores (mesmas do sankey_race_multi_layers_otimizado)
# =============================================================================

class ColorPalette(Enum):
    """Paletas de cores disponíveis."""
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


# Mapping from ColorPalette to matplotlib colormap names
# These are continuous colormaps that support smooth interpolation
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
            - String name of matplotlib colormap (e.g., "viridis", "RdYlGn")
            - List of hex colors for custom discrete palette (e.g., ["#FF0000", "#00FF00", "#0000FF"])

    Returns:
        matplotlib colormap object
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Custom discrete palette (list of hex colors)
    if isinstance(palette, list):
        # Convert hex colors to RGB tuples
        rgb_colors = [to_rgb(c) for c in palette]
        # Create a continuous colormap from discrete colors
        cmap = LinearSegmentedColormap.from_list("custom", rgb_colors, N=256)
        return cmap

    # ColorPalette enum
    if isinstance(palette, ColorPalette):
        cmap_name = PALETTE_COLORMAPS.get(palette, "rainbow")
        return plt.colormaps.get_cmap(cmap_name)

    # String - could be ColorPalette name or matplotlib colormap name
    if isinstance(palette, str):
        # Try as ColorPalette first
        try:
            palette_enum = ColorPalette(palette.lower())
            cmap_name = PALETTE_COLORMAPS.get(palette_enum, "rainbow")
            return plt.colormaps.get_cmap(cmap_name)
        except ValueError:
            # Not a ColorPalette, try as matplotlib colormap name
            try:
                return plt.colormaps.get_cmap(palette)
            except ValueError:
                # Fallback to rainbow
                return plt.colormaps.get_cmap("rainbow")

    # Default fallback
    return plt.colormaps.get_cmap("rainbow")


def get_palette_colors(palette: Union[ColorPalette, str, List[str]], n_colors: int, reverse: bool = False) -> List[str]:
    """
    Get a list of colors from a palette with continuous interpolation.

    Args:
        palette: Can be:
            - ColorPalette enum (e.g., ColorPalette.VIRIDIS)
            - String name of matplotlib colormap (e.g., "viridis", "RdYlGn", "coolwarm")
            - List of hex colors for custom discrete palette (e.g., ["#FF0000", "#00FF00", "#0000FF"])
        n_colors: Number of colors to generate
        reverse: If True, reverse the color order

    Returns:
        List of hex colors (#RRGGBB)

    Examples:
        # Using built-in palette
        colors = get_palette_colors(ColorPalette.VIRIDIS, 5)

        # Using matplotlib colormap name
        colors = get_palette_colors("RdYlGn", 10)

        # Using custom discrete colors (will interpolate between them)
        colors = get_palette_colors(["#FF0000", "#FFFF00", "#00FF00"], 10)
    """
    cmap = get_colormap(palette)

    # Sample n_colors evenly spaced points from [0, 1]
    if n_colors == 1:
        positions = [0.5]
    else:
        positions = [i / (n_colors - 1) for i in range(n_colors)]

    if reverse:
        positions = positions[::-1]

    # Convert to hex colors
    colors = [plt.matplotlib.colors.rgb2hex(cmap(pos)[:3]) for pos in positions]

    return colors


@lru_cache(maxsize=256)
def get_rgb_cached(color: str) -> Tuple[float, float, float]:
    """Converte cor hex para RGB com cache."""
    return to_rgb(color)


def get_text_color_for_background(bg_color: str) -> str:
    """Retorna cor de texto ideal baseado na luminosidade."""
    try:
        rgb = get_rgb_cached(bg_color)
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return '#000000' if luminance > 0.5 else '#FFFFFF'
    except:
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


def get_dynamic_color(value: float, min_val: float, max_val: float,
                      colormap: Union[ColorPalette, str, List[str]] = 'RdYlGn') -> str:
    """
    Return color based on normalized value.

    Args:
        value: The value to map to a color
        min_val: Minimum value in the range
        max_val: Maximum value in the range
        colormap: Can be ColorPalette, matplotlib colormap name, or list of hex colors
    """
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (value - min_val) / (max_val - min_val)
    cmap = get_colormap(colormap)
    rgba = cmap(normalized)
    return plt.matplotlib.colors.rgb2hex(rgba[:3])


def get_ranking_color(rank: int, total: int, colormap: Union[ColorPalette, str, List[str]] = 'RdYlGn') -> str:
    """
    Return color based on ranking (1st = high/green, last = low/red).

    Args:
        rank: The rank (1-indexed, 1 = best)
        total: Total number of items being ranked
        colormap: Can be ColorPalette, matplotlib colormap name, or list of hex colors
    """
    if total <= 1:
        normalized = 1.0
    else:
        normalized = 1.0 - (rank - 1) / (total - 1)
    cmap = get_colormap(colormap)
    rgba = cmap(normalized)
    return plt.matplotlib.colors.rgb2hex(rgba[:3])


def interpolate_color(color1: str, color2: str, t: float) -> str:
    """Interpolate between two hex colors."""
    c1 = to_rgb(color1)
    c2 = to_rgb(color2)
    r = c1[0] + (c2[0] - c1[0]) * t
    g = c1[1] + (c2[1] - c1[1]) * t
    b = c1[2] + (c2[2] - c1[2]) * t
    return plt.matplotlib.colors.rgb2hex((r, g, b))


# =============================================================================
# Estruturas de dados (pickleable para multiprocessing)
# =============================================================================

@dataclass
class FrameData:
    """Dados de um frame interpolado."""
    time_label: str
    links: List[Tuple[str, str, float]]
    node_positions: Dict[str, float]
    node_values: Dict[str, float]
    node_colors: Optional[Dict[str, str]] = None  # Dynamic colors per frame


# Configurações de qualidade (mesmas do original)
DEFAULT_GRADIENT_SEGMENTS = 50


# =============================================================================
# Função de renderização de alta qualidade para save_frame
# =============================================================================

def render_frame_high_quality(ax, frame_data, layers, node_colors,
                              plot_width, plot_height, node_width, font_size,
                              n_segments, bar_height_ratio, stacked_mode):
    """
    Renderiza um frame com alta qualidade (mesma lógica do _render_chunk).

    Args:
        ax: Matplotlib axes
        frame_data: FrameData com links, posições, valores
        layers: Lista de camadas de nós
        node_colors: Dicionário de cores dos nós
        plot_width: Largura do plot
        plot_height: Altura do plot
        node_width: Largura dos nós
        font_size: Tamanho da fonte
        n_segments: Segmentos do gradiente
        bar_height_ratio: Proporção da altura da barra
        stacked_mode: Se True, usa modo empilhado
    """
    n_layers = len(layers)
    padding = 1.2
    margin_top_ratio = 0.12
    margin_bottom_ratio = 0.05

    # Pré-calcular valores de t para gradiente
    t = np.linspace(0, 1, n_segments + 1)
    t0 = t[:-1]
    t1 = t[1:]
    t_mid = (t0 + t1) / 2

    def bezier_vec(t_arr, y_start, y_end):
        factor = np.where(
            t_arr < 0.5,
            4 * t_arr ** 3,
            1 - (-2 * t_arr + 2) ** 3 / 2
        )
        return y_start + (y_end - y_start) * factor

    # Pré-calcular posições X das camadas
    layer_spacing = (plot_width - 2 * padding - node_width) / max(1, n_layers - 1)
    layer_x_positions = {}
    for layer_idx in range(n_layers):
        if n_layers == 1:
            layer_x_positions[layer_idx] = plot_width / 2 - node_width / 2
        else:
            layer_x_positions[layer_idx] = padding + layer_idx * layer_spacing

    # Margens
    margin_top = plot_height * margin_top_ratio
    margin_bottom = plot_height * margin_bottom_ratio
    usable_height = plot_height - margin_top - margin_bottom

    # Calcular global_max para escala
    global_max = max(frame_data.node_values.values()) if frame_data.node_values else 1

    # Escala
    if stacked_mode:
        layer_totals = [sum(frame_data.node_values.get(node, 0) for node in layer_nodes)
                      for layer_nodes in layers]
        max_layer_total = max(layer_totals) if layer_totals else 1
        stacked_height = usable_height * bar_height_ratio
        scale = stacked_height / max_layer_total if max_layer_total > 0 else 1
    else:
        # Fixed mode: uniform node heights (no scaling by value)
        max_nodes_in_layer = max(len(layer) for layer in layers)
        available_height_per_slot = usable_height / max_nodes_in_layer
        fixed_node_height = available_height_per_slot * bar_height_ratio * 0.8
        scale = None  # Not used in fixed mode

    # Calcular posições dos nós
    node_positions_rect = {}
    for layer_idx, layer_nodes in enumerate(layers):
        layer_x = layer_x_positions[layer_idx]
        for node in layer_nodes:
            val = frame_data.node_values.get(node, 0)
            if stacked_mode:
                h = max(val * scale, 0.1)
            else:
                # Fixed mode: all nodes have the same height
                h = fixed_node_height
            y_center = frame_data.node_positions.get(node, margin_bottom + usable_height * 0.5)
            node_positions_rect[node] = (layer_x, y_center - h/2, h)

    # Calcular somas para links
    out_sums = {}
    in_sums = {}
    for src, tgt, val in frame_data.links:
        out_sums[src] = out_sums.get(src, 0) + val
        in_sums[tgt] = in_sums.get(tgt, 0) + val

    # Escalas de links
    node_out_scale = {}
    node_in_scale = {}
    for node in node_positions_rect:
        total_out = out_sums.get(node, 0)
        total_in = in_sums.get(node, 0)
        h = node_positions_rect[node][2]
        node_out_scale[node] = h / total_out if total_out > 0 else 1
        node_in_scale[node] = h / total_in if total_in > 0 else 1

    # Posições de saída/entrada
    node_out_pos = {node: node_positions_rect[node][1] for node in node_positions_rect}
    node_in_pos = {node: node_positions_rect[node][1] for node in node_positions_rect}

    # Construir gradientes
    all_vertices = []
    all_colors = []

    sorted_links = sorted(frame_data.links, key=lambda x: (x[0], x[1]))

    for src, tgt, val in sorted_links:
        if val <= 0 or src not in node_positions_rect or tgt not in node_positions_rect:
            continue

        link_height_src = val * node_out_scale.get(src, 1)
        link_height_tgt = val * node_in_scale.get(tgt, 1)

        src_x, src_y, src_h = node_positions_rect[src]
        x0 = src_x + node_width
        y0_bot = node_out_pos[src]
        y0_top = y0_bot + link_height_src

        tgt_x, tgt_y, tgt_h = node_positions_rect[tgt]
        x1 = tgt_x
        y1_bot = node_in_pos[tgt]
        y1_top = y1_bot + link_height_tgt

        # Cores - usar cores dinâmicas se disponíveis
        if frame_data.node_colors:
            color_src = frame_data.node_colors.get(src, '#888888')
            color_tgt = frame_data.node_colors.get(tgt, '#888888')
        else:
            color_src = node_colors.get(src, '#888888')
            color_tgt = node_colors.get(tgt, '#888888')

        rgb_start = np.array(to_rgb(color_src))
        rgb_end = np.array(to_rgb(color_tgt))

        # Construir segmentos do gradiente
        seg_x0 = x0 + (x1 - x0) * t0
        seg_x1 = x0 + (x1 - x0) * t1

        seg_y0_top = bezier_vec(t0, y0_top, y1_top)
        seg_y0_bot = bezier_vec(t0, y0_bot, y1_bot)
        seg_y1_top = bezier_vec(t1, y0_top, y1_top)
        seg_y1_bot = bezier_vec(t1, y0_bot, y1_bot)

        # Cores interpoladas
        colors = rgb_start + np.outer(t_mid, rgb_end - rgb_start)

        for i in range(n_segments):
            verts = [
                (seg_x0[i], seg_y0_bot[i]),
                (seg_x0[i], seg_y0_top[i]),
                (seg_x1[i], seg_y1_top[i]),
                (seg_x1[i], seg_y1_bot[i])
            ]
            all_vertices.append(verts)
            all_colors.append((*colors[i], 0.7))  # RGBA com alpha

        node_out_pos[src] += link_height_src
        node_in_pos[tgt] += link_height_tgt

    # Desenhar links
    if all_vertices:
        pc = PolyCollection(all_vertices, facecolors=all_colors, edgecolors='none')
        ax.add_collection(pc)

    # Desenhar nós com FancyBboxPatch
    min_height_for_inside_text = 0.5

    for layer_idx, layer_nodes in enumerate(layers):
        for node in layer_nodes:
            if node not in node_positions_rect:
                continue

            x, y, h = node_positions_rect[node]

            # Cor do nó - usar dinâmica se disponível
            if frame_data.node_colors:
                color = frame_data.node_colors.get(node, '#888888')
            else:
                color = node_colors.get(node, '#888888')

            # FancyBboxPatch com cantos arredondados
            rect = mpatches.FancyBboxPatch(
                (x, y), node_width, h,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=color,
                edgecolor='#333333',
                linewidth=1.5
            )
            ax.add_patch(rect)

            val = frame_data.node_values.get(node, 0)

            # Calcular cor do texto
            try:
                rgb = to_rgb(color)
                luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                text_color = '#000000' if luminance > 0.5 else '#FFFFFF'
            except:
                text_color = '#000000'

            # Nome do nó
            if layer_idx == 0:
                ax.text(x - 0.15, y + h / 2, f"{node}",
                       ha='right', va='center', fontsize=font_size, fontweight='bold')
            elif layer_idx == n_layers - 1:
                ax.text(x + node_width + 0.15, y + h / 2, f"{node}",
                       ha='left', va='center', fontsize=font_size, fontweight='bold')
            else:
                ax.text(x + node_width / 2, y + h + 0.15, f"{node}",
                       ha='center', va='bottom', fontsize=font_size - 1, fontweight='bold')

            # Valor dentro do nó
            if h >= min_height_for_inside_text:
                ax.text(x + node_width / 2, y + h / 2, f"{val:.0f}",
                       ha='center', va='center', fontsize=font_size - 1,
                       color=text_color, fontweight='bold')


# =============================================================================
# Worker function - Renderiza chunk com MESMA qualidade do original
# =============================================================================

def _render_chunk(args):
    """
    Renderiza um chunk de frames em um processo separado.
    USA MESMA LÓGICA DE RENDERIZAÇÃO do sankey_race_multi_layers_otimizado.
    """
    (chunk_id, frames_data, config) = args

    # Importar matplotlib dentro do worker (isolado)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import PolyCollection

    # Desempacotar config
    width = config['width']
    height = config['height']
    figsize = config['figsize']
    fps = config['fps']
    bitrate = config['bitrate']
    title = config['title']
    layers = config['layers']
    node_colors = config['node_colors']
    rgb_cache = config['rgb_cache']
    node_layer = config['node_layer']
    temp_dir = config['temp_dir']
    n_segments = config['n_segments']

    # Parâmetros de layout (mesmos do original)
    plot_width = config['plot_width']
    plot_height = config['plot_height']
    node_width = config['node_width']
    padding = config['padding']
    global_max = config['global_max']
    font_size = config['font_size']
    bar_height_ratio = config['bar_height_ratio']
    margin_top_ratio = config['margin_top_ratio']
    margin_bottom_ratio = config['margin_bottom_ratio']
    title_fontsize = config['title_fontsize']
    title_bg_color = config['title_bg_color']
    title_bg_alpha = config['title_bg_alpha']
    stacked_mode = config['stacked_mode']

    # Criar arquivo de saída para este chunk
    chunk_path = os.path.join(temp_dir, f"chunk_{chunk_id:04d}.mp4")

    # Criar figura com mesmas dimensões do original
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    fig.tight_layout(pad=0.5)

    # Obter dimensões reais do canvas
    fig.canvas.draw()
    canvas_width, canvas_height = fig.canvas.get_width_height()

    # Iniciar FFmpeg para este chunk
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
        chunk_path
    ]

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    n_layers = len(layers)

    # Pré-calcular valores de t para gradiente
    t = np.linspace(0, 1, n_segments + 1)
    t0 = t[:-1]
    t1 = t[1:]
    t_mid = (t0 + t1) / 2

    def bezier_vec(t_arr, y_start, y_end):
        factor = np.where(
            t_arr < 0.5,
            4 * t_arr ** 3,
            1 - (-2 * t_arr + 2) ** 3 / 2
        )
        return y_start + (y_end - y_start) * factor

    # Pré-calcular posições X das camadas
    layer_spacing = (plot_width - 2 * padding - node_width) / max(1, n_layers - 1)
    layer_x_positions = {}
    for layer_idx in range(n_layers):
        if n_layers == 1:
            layer_x_positions[layer_idx] = plot_width / 2 - node_width / 2
        else:
            layer_x_positions[layer_idx] = padding + layer_idx * layer_spacing

    # Renderizar cada frame
    for frame_data in frames_data:
        ax.clear()
        ax.set_xlim(0, plot_width)
        ax.set_ylim(0, plot_height)
        ax.set_aspect('auto')
        ax.axis('off')

        # Margens
        margin_top = plot_height * margin_top_ratio
        margin_bottom = plot_height * margin_bottom_ratio
        usable_height = plot_height - margin_top - margin_bottom

        # Escala (mesma lógica do original)
        if stacked_mode:
            layer_totals = [sum(frame_data.node_values.get(node, 0) for node in layer_nodes)
                          for layer_nodes in layers]
            max_layer_total = max(layer_totals) if layer_totals else 1
            stacked_height = usable_height * bar_height_ratio
            scale = stacked_height / max_layer_total if max_layer_total > 0 else 1
        else:
            # Fixed mode: uniform node heights (no scaling by value)
            max_nodes_in_layer = max(len(layer) for layer in layers)
            available_height_per_slot = usable_height / max_nodes_in_layer
            fixed_node_height = available_height_per_slot * bar_height_ratio * 0.8
            scale = None  # Not used in fixed mode

        # Calcular posições dos nós
        node_positions_rect = {}
        for layer_idx, layer_nodes in enumerate(layers):
            layer_x = layer_x_positions[layer_idx]
            for node in layer_nodes:
                val = frame_data.node_values.get(node, 0)
                if stacked_mode:
                    h = max(val * scale, 0.1)
                else:
                    # Fixed mode: all nodes have the same height
                    h = fixed_node_height
                y_center = frame_data.node_positions.get(node, margin_bottom + usable_height * 0.5)
                node_positions_rect[node] = (layer_x, y_center - h/2, h)

        # Calcular somas para links
        out_sums = {}
        in_sums = {}
        for src, tgt, val in frame_data.links:
            out_sums[src] = out_sums.get(src, 0) + val
            in_sums[tgt] = in_sums.get(tgt, 0) + val

        # Determinar em qual camada cada nó está
        node_layer_idx = {}
        for layer_idx, layer_nodes in enumerate(layers):
            for node in layer_nodes:
                node_layer_idx[node] = layer_idx
        n_layers = len(layers)

        # Escalas de links - CORRIGIDO para nós intermediários
        # Para nós intermediários, usar a MESMA escala para entrada e saída
        # baseada no max(in, out) para garantir consistência visual
        node_out_scale = {}
        node_in_scale = {}
        for node in node_positions_rect:
            total_out = out_sums.get(node, 0)
            total_in = in_sums.get(node, 0)
            h = node_positions_rect[node][2]
            layer_idx = node_layer_idx.get(node, 0)

            # Para nós intermediários (não primeira nem última camada),
            # usar o mesmo fator de escala para entrada e saída
            if 0 < layer_idx < n_layers - 1:
                # Nó intermediário: usar max para escala consistente
                total_max = max(total_out, total_in, 0.001)
                node_out_scale[node] = h / total_max if total_out > 0 else 1
                node_in_scale[node] = h / total_max if total_in > 0 else 1
            else:
                # Primeira ou última camada: escala normal
                node_out_scale[node] = h / total_out if total_out > 0 else 1
                node_in_scale[node] = h / total_in if total_in > 0 else 1

        # Posições de saída/entrada
        node_out_pos = {node: node_positions_rect[node][1] for node in node_positions_rect}
        node_in_pos = {node: node_positions_rect[node][1] for node in node_positions_rect}

        # Construir gradientes (MESMA lógica do original)
        all_vertices = []
        all_colors = []

        sorted_links = sorted(frame_data.links, key=lambda x: (x[0], x[1]))

        for src, tgt, val in sorted_links:
            if val <= 0 or src not in node_positions_rect or tgt not in node_positions_rect:
                continue

            link_height_src = val * node_out_scale.get(src, 1)
            link_height_tgt = val * node_in_scale.get(tgt, 1)

            src_x, src_y, src_h = node_positions_rect[src]
            x0 = src_x + node_width
            y0_bot = node_out_pos[src]
            y0_top = y0_bot + link_height_src

            tgt_x, tgt_y, tgt_h = node_positions_rect[tgt]
            x1 = tgt_x
            y1_bot = node_in_pos[tgt]
            y1_top = y1_bot + link_height_tgt

            # Cores (use dynamic colors if available, otherwise use static)
            if frame_data.node_colors:
                color_src = frame_data.node_colors.get(src, '#888888')
                color_tgt = frame_data.node_colors.get(tgt, '#888888')
                rgb_start = np.array(to_rgb(color_src))
                rgb_end = np.array(to_rgb(color_tgt))
            else:
                rgb_start = np.array(rgb_cache.get(src, (0.5, 0.5, 0.5)))
                rgb_end = np.array(rgb_cache.get(tgt, (0.5, 0.5, 0.5)))

            # Construir segmentos do gradiente
            seg_x0 = x0 + (x1 - x0) * t0
            seg_x1 = x0 + (x1 - x0) * t1

            seg_y0_top = bezier_vec(t0, y0_top, y1_top)
            seg_y0_bot = bezier_vec(t0, y0_bot, y1_bot)
            seg_y1_top = bezier_vec(t1, y0_top, y1_top)
            seg_y1_bot = bezier_vec(t1, y0_bot, y1_bot)

            # Cores interpoladas
            colors = rgb_start + np.outer(t_mid, rgb_end - rgb_start)

            for i in range(n_segments):
                verts = [
                    (seg_x0[i], seg_y0_bot[i]),
                    (seg_x0[i], seg_y0_top[i]),
                    (seg_x1[i], seg_y1_top[i]),
                    (seg_x1[i], seg_y1_bot[i])
                ]
                all_vertices.append(verts)
                all_colors.append((*colors[i], 0.7))  # RGBA com alpha

            node_out_pos[src] += link_height_src
            node_in_pos[tgt] += link_height_tgt

        # Desenhar links
        if all_vertices:
            pc = PolyCollection(all_vertices, facecolors=all_colors, edgecolors='none')
            ax.add_collection(pc)

        # Desenhar nós com FancyBboxPatch (MESMA qualidade do original)
        min_height_for_inside_text = 0.5

        for layer_idx, layer_nodes in enumerate(layers):
            for node in layer_nodes:
                if node not in node_positions_rect:
                    continue

                x, y, h = node_positions_rect[node]
                # Use dynamic colors if available, otherwise static
                if frame_data.node_colors:
                    color = frame_data.node_colors.get(node, '#888888')
                else:
                    color = node_colors.get(node, '#888888')

                # FancyBboxPatch com cantos arredondados
                rect = mpatches.FancyBboxPatch(
                    (x, y), node_width, h,
                    boxstyle="round,pad=0.02,rounding_size=0.05",
                    facecolor=color,
                    edgecolor='#333333',
                    linewidth=1.5
                )
                ax.add_patch(rect)

                val = frame_data.node_values.get(node, 0)

                # Calcular cor do texto
                try:
                    rgb = to_rgb(color)
                    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    text_color = '#000000' if luminance > 0.5 else '#FFFFFF'
                except:
                    text_color = '#000000'

                # Nome do nó
                if layer_idx == 0:
                    ax.text(x - 0.15, y + h / 2, f"{node}",
                           ha='right', va='center', fontsize=font_size, fontweight='bold')
                elif layer_idx == n_layers - 1:
                    ax.text(x + node_width + 0.15, y + h / 2, f"{node}",
                           ha='left', va='center', fontsize=font_size, fontweight='bold')
                else:
                    ax.text(x + node_width / 2, y + h + 0.15, f"{node}",
                           ha='center', va='bottom', fontsize=font_size - 1, fontweight='bold')

                # Valor dentro do nó
                if h >= min_height_for_inside_text:
                    ax.text(x + node_width / 2, y + h / 2, f"{val:.0f}",
                           ha='center', va='center', fontsize=font_size - 1,
                           color=text_color, fontweight='bold')

        # Título
        display_title = title if title else ""
        ax.text(plot_width / 2, plot_height * 0.95,
               f"{display_title}\n{frame_data.time_label}" if display_title else frame_data.time_label,
               ha='center', va='top', fontsize=title_fontsize, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=title_bg_color, alpha=title_bg_alpha))

        # Capturar frame
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img_array = np.asarray(buf)
        rgb_array = np.ascontiguousarray(img_array[:, :, :3])

        try:
            process.stdin.write(rgb_array.tobytes())
        except:
            break

    process.stdin.close()
    process.communicate(timeout=300)

    plt.close(fig)

    return chunk_id, chunk_path, len(frames_data)


# =============================================================================
# Classe principal
# =============================================================================

class SankeyRaceMultiLayerParallel:
    """
    Sankey Race Multi-Layer com renderização PARALELA.

    Combina:
    - MESMA qualidade visual do sankey_race_multi_layers_otimizado
    - Renderização paralela com multiprocessing
    - Todas as paletas e modos (ranking, stacked, stacked+ranking)
    """

    def __init__(self,
                 layers: List[List[str]],
                 node_colors: Dict[str, str]):
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

        # Cache de global_max
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
        """Cria instância a partir de um DataFrame."""

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

        # Processar frames com .values (vetorizado)
        for time_val in sorted(df[time_col].unique()):
            df_t = df[df[time_col] == time_val]
            src_arr = df_t[source_col].values
            tgt_arr = df_t[target_col].values
            val_arr = df_t[value_col].values
            links = list(zip(src_arr, tgt_arr, val_arr))
            instance.frames.append({'time_label': str(time_val), 'links': links})

        # Pré-calcular global_max
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
        """Calcula valor máximo global."""
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
        """Calcula posições Y baseadas no ranking por camada (sem stacking)."""
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
                                   margin_top_ratio, margin_bottom_ratio, gap):
        """Calcula posições Y no modo stacked (empilhado, sem ranking)."""
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

            # Mantém ordem original (sem sorting)
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

    def _compute_fixed_positions(self, node_values, plot_height, global_max, bar_height_ratio,
                                 margin_top_ratio, margin_bottom_ratio):
        """Calcula posições Y fixas (sem ranking, sem stacking)."""
        margin_top = plot_height * margin_top_ratio
        margin_bottom = plot_height * margin_bottom_ratio
        usable_height = plot_height - margin_top - margin_bottom

        positions = {}
        for layer_nodes in self.layers:
            n_nodes = len(layer_nodes)
            if n_nodes == 0:
                continue

            max_nodes_in_layer = max(len(layer) for layer in self.layers)
            available_height_per_slot = usable_height / max_nodes_in_layer

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
            node_values: Dict mapping node names to their values
            mode: Coloring mode (RANKING, VALUE, GLOBAL_VALUE, PERCENTILE)
            colormap: Can be:
                - ColorPalette enum (e.g., ColorPalette.VIRIDIS)
                - Matplotlib colormap name (e.g., 'RdYlGn', 'viridis')
                - List of hex colors for custom palette (e.g., ['#FF0000', '#00FF00'])
            global_min: Min value for global normalization
            global_max: Max value for global normalization

        Returns:
            Dict mapping node names to hex color strings
        """
        if isinstance(mode, str):
            mode = DynamicColorMode(mode.lower())

        colors = {}

        if mode == DynamicColorMode.STATIC:
            # Return original static colors
            return self.node_colors.copy()

        elif mode == DynamicColorMode.RANKING:
            # Color by rank within each layer (1st = green, last = red)
            for layer in self.layers:
                layer_values = [(node, node_values.get(node, 0)) for node in layer]
                sorted_nodes = sorted(layer_values, key=lambda x: -x[1])
                for rank, (node, _) in enumerate(sorted_nodes, 1):
                    colors[node] = get_ranking_color(rank, len(layer), colormap)

        elif mode == DynamicColorMode.VALUE:
            # Color by value (normalized within each layer)
            for layer in self.layers:
                layer_vals = [node_values.get(node, 0) for node in layer]
                min_val = min(layer_vals) if layer_vals else 0
                max_val = max(layer_vals) if layer_vals else 1
                for node in layer:
                    val = node_values.get(node, 0)
                    colors[node] = get_dynamic_color(val, min_val, max_val, colormap)

        elif mode == DynamicColorMode.GLOBAL_VALUE:
            # Color by value (normalized globally)
            if global_min is None:
                global_min = min(node_values.values()) if node_values else 0
            if global_max is None:
                global_max = max(node_values.values()) if node_values else 1
            for node in node_values:
                colors[node] = get_dynamic_color(node_values[node], global_min, global_max, colormap)

        elif mode == DynamicColorMode.PERCENTILE:
            # Color by percentile within each layer
            for layer in self.layers:
                layer_vals = sorted([node_values.get(node, 0) for node in layer])
                n = len(layer_vals)
                for node in layer:
                    val = node_values.get(node, 0)
                    # Find percentile
                    rank = sum(1 for v in layer_vals if v <= val)
                    percentile = rank / n if n > 0 else 0.5
                    colors[node] = get_dynamic_color(percentile, 0, 1, colormap)

        return colors

    def _compute_stacked_ranking_positions(self, node_values, plot_height, bar_height_ratio,
                                           margin_top_ratio, margin_bottom_ratio, gap, ascending):
        """Calcula posições Y no modo stacked + ranking."""
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
                           mode='stacked_ranking', global_max=None,
                           dynamic_color_mode: Union[DynamicColorMode, str] = DynamicColorMode.STATIC,
                           dynamic_colormap: Union[ColorPalette, str, List[str]] = 'RdYlGn'):
        """Pré-computa todos os frames interpolados.

        Args:
            mode: 'stacked_ranking', 'ranking', 'stacked', ou 'fixed'
            dynamic_color_mode: Mode for dynamic node coloring (STATIC, RANKING, VALUE, etc.)
            dynamic_colormap: Colormap for dynamic colors. Can be:
                - ColorPalette enum
                - Matplotlib colormap name (string)
                - List of hex colors for custom palette
        """
        n_data_frames = len(self.frames)
        frames_per_period = max(1, total_frames // n_data_frames)

        # Convert string to enum if needed
        if isinstance(dynamic_color_mode, str):
            dynamic_color_mode = DynamicColorMode(dynamic_color_mode.lower())

        use_dynamic_colors = dynamic_color_mode != DynamicColorMode.STATIC

        # Pre-compute global min/max for GLOBAL_VALUE mode
        global_min_val = None
        global_max_val = None
        if dynamic_color_mode == DynamicColorMode.GLOBAL_VALUE:
            all_values = []
            for frame in self.frames:
                values = self._compute_node_values(frame['links'])
                all_values.extend(values.values())
            if all_values:
                global_min_val = min(all_values)
                global_max_val = max(all_values)

        # Calcular valores/posições/cores para cada frame de dados
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
                    margin_top_ratio, margin_bottom_ratio, stacked_gap
                )
            else:  # fixed
                positions = self._compute_fixed_positions(
                    values, plot_height, global_max or self._get_global_max(),
                    bar_height_ratio, margin_top_ratio, margin_bottom_ratio
                )

            # Compute dynamic colors if enabled
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
                    if v > 0.1:
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

                # Interpolate colors if dynamic
                interp_colors = None
                if use_dynamic_colors and colors_a and colors_b:
                    interp_colors = {}
                    for node in all_nodes:
                        ca = colors_a.get(node, '#888888')
                        cb = colors_b.get(node, '#888888')
                        interp_colors[node] = interpolate_color(ca, cb, t)

                interpolated.append(FrameData(
                    time_label=frame_a['time_label'] if t < 0.5 else frame_b['time_label'],
                    links=interp_links,
                    node_positions=interp_pos,
                    node_values=interp_values,
                    node_colors=interp_colors
                ))

        # Frames de hold
        last_frame, last_values, last_pos, last_colors = data_info[-1]
        remaining = total_frames - len(interpolated)
        for _ in range(remaining):
            interpolated.append(FrameData(
                time_label=last_frame['time_label'],
                links=last_frame['links'],
                node_positions=last_pos,
                node_values=last_values,
                node_colors=last_colors
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
                dynamic_colormap: Union[ColorPalette, str, List[str]] = "RdYlGn"):
        """
        Gera animação usando múltiplos processos em paralelo.

        Args:
            ranking_mode: Se True, ordena nós por valor
            stacked_mode: Se True, empilha nós proporcionalmente ao valor
            n_workers: Número de processos (default: número de CPUs)
            n_segments: Segmentos de gradiente (default: 50, igual ao original)
            dynamic_color_mode: Modo de coloração dinâmica dos nós:
                - "static": Cores fixas (comportamento original)
                - "ranking": Cor por ranking na camada (1º=verde, último=vermelho)
                - "value": Cor por valor normalizado na camada
                - "global_value": Cor por valor normalizado globalmente
                - "percentile": Cor por percentil na camada
            dynamic_colormap: Colormap para cores dinâmicas. Pode ser:
                - ColorPalette enum (ex: ColorPalette.VIRIDIS)
                - Nome de colormap do matplotlib (ex: 'RdYlGn', 'viridis', 'coolwarm')
                - Lista de cores hex para paleta customizada (ex: ['#FF0000', '#FFFF00', '#00FF00'])

        Modos de posicionamento:
            - ranking_mode=True, stacked_mode=True: Stacked + Ranking (padrão)
              Nodes reorder by value AND resize proportionally
            - ranking_mode=True, stacked_mode=False: Apenas Ranking
              Nodes reorder by value but have UNIFORM heights
            - ranking_mode=False, stacked_mode=True: Apenas Stacked
              Fixed node order, heights vary by value
            - ranking_mode=False, stacked_mode=False: Fixo
              Fixed node order AND uniform heights (only links animate)
        """
        if n_workers is None:
            n_workers = mp.cpu_count()

        if not self.frames:
            raise ValueError("Nenhum frame adicionado.")

        # Determinar modo
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
            mode_name = 'Fixo'

        # Configurações de qualidade
        quality_settings = {
            "low": {"dpi": 72, "bitrate": "1500k"},
            "medium": {"dpi": 120, "bitrate": "3000k"},
            "high": {"dpi": 200, "bitrate": "8000k"},
        }
        settings = quality_settings.get(quality, quality_settings["medium"])

        plot_width, plot_height = figsize
        global_max = self._get_global_max()
        total_frames = int(fps * duration_seconds)

        # Parse dynamic color mode
        if isinstance(dynamic_color_mode, str):
            dynamic_color_mode_enum = DynamicColorMode(dynamic_color_mode.lower())
        else:
            dynamic_color_mode_enum = dynamic_color_mode

        color_mode_name = dynamic_color_mode_enum.value.replace('_', ' ').title()

        print(f"Configuracoes (MULTI-LAYER PARALLEL - {n_workers} workers):")
        print(f"  - Camadas: {len(self.layers)}")
        print(f"  - Nos por camada: {[len(l) for l in self.layers]}")
        print(f"  - FPS: {fps}, Duracao: {duration_seconds}s")
        print(f"  - Qualidade: {quality}")
        print(f"  - Total de frames: {total_frames}")
        print(f"  - Segmentos de gradiente: {n_segments}")
        print(f"  - Modo posicionamento: {mode_name}")
        print(f"  - Modo cor dinamica: {color_mode_name}")
        if dynamic_color_mode_enum != DynamicColorMode.STATIC:
            print(f"  - Colormap: {dynamic_colormap}")

        # Pré-computar frames
        print(f"\nPre-computando {total_frames} frames...")
        t0 = time.time()
        all_frames = self._precompute_frames(
            total_frames, plot_height, bar_height_ratio,
            margin_top, margin_bottom, stacked_gap, ascending,
            mode=mode, global_max=global_max,
            dynamic_color_mode=dynamic_color_mode_enum,
            dynamic_colormap=dynamic_colormap
        )
        print(f"  Pre-computacao: {time.time() - t0:.2f}s")

        # Dividir em chunks
        chunk_size = len(all_frames) // n_workers
        chunks = []
        for i in range(n_workers):
            start = i * chunk_size
            end = start + chunk_size if i < n_workers - 1 else len(all_frames)
            chunks.append(all_frames[start:end])

        # Diretório temporário
        temp_dir = tempfile.mkdtemp(prefix="sankey_multi_parallel_")

        # Config compartilhada (MESMOS parâmetros do original)
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
            'global_max': global_max,
            'font_size': font_size,
            'bar_height_ratio': bar_height_ratio,
            'margin_top_ratio': margin_top,
            'margin_bottom_ratio': margin_bottom,
            'title_fontsize': title_fontsize,
            'title_bg_color': title_bg_color,
            'title_bg_alpha': title_bg_alpha,
            'stacked_mode': stacked_mode,
        }

        # Preparar argumentos
        worker_args = [(i, chunks[i], config) for i in range(n_workers)]

        print(f"\nRenderizando em {n_workers} processos paralelos...")
        t0 = time.time()

        # Executar em paralelo
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_render_chunk, worker_args)

        tempo_render = time.time() - t0
        print(f"  Renderizacao paralela: {tempo_render:.2f}s ({total_frames / tempo_render:.1f} fps)")

        # Ordenar e concatenar
        results.sort(key=lambda x: x[0])
        chunk_paths = [r[1] for r in results]

        print(f"\nConcatenando {n_workers} chunks...")
        t0_concat = time.time()

        # Criar lista para ffmpeg concat
        list_path = os.path.join(temp_dir, "chunks.txt")
        with open(list_path, 'w') as f:
            for path in chunk_paths:
                f.write(f"file '{path}'\n")

        concat_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_path,
            '-c', 'copy',
            output_path
        ]

        result = subprocess.run(concat_cmd, capture_output=True)

        if result.returncode != 0:
            print(f"Erro na concatenacao: {result.stderr.decode()}")

        print(f"  Concatenacao: {time.time() - t0_concat:.2f}s")

        # Limpar
        shutil.rmtree(temp_dir, ignore_errors=True)

        tempo_total = tempo_render + (time.time() - t0_concat)
        print(f"\nAnimacao salva em: {output_path}")
        print(f"Tempo total: {tempo_total:.2f}s ({total_frames / tempo_total:.1f} fps efetivo)")

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
                   n_segments: int = DEFAULT_GRADIENT_SEGMENTS):
        """
        Salva um frame unico como imagem estatica (PNG, PDF, SVG).

        Args:
            output_path: Caminho do arquivo de saida (.png, .pdf, .svg)
            frame_index: Indice do frame a ser salvo (0 = primeiro)
            title: Titulo do grafico
            figsize: Tamanho da figura (largura, altura)
            dpi: Resolucao da imagem
            node_width: Largura dos nos
            ranking_mode: Se True, ordena nos por valor
            stacked_mode: Se True, empilha nos proporcionalmente
            n_segments: Segmentos de gradiente (default: 50)

        Returns:
            Caminho do arquivo salvo
        """
        if not self.frames:
            raise ValueError("Nenhum frame adicionado.")

        if frame_index >= len(self.frames):
            raise ValueError(f"frame_index {frame_index} invalido. Total de frames: {len(self.frames)}")

        # Determinar modo
        if stacked_mode and ranking_mode:
            mode = 'stacked_ranking'
        elif ranking_mode:
            mode = 'ranking'
        elif stacked_mode:
            mode = 'stacked'
        else:
            mode = 'fixed'

        plot_width, plot_height = figsize
        global_max = self._get_global_max()

        # Computar dados do frame
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
                values, plot_height, global_max,
                bar_height_ratio, margin_top, margin_bottom
            )

        frame_data = FrameData(
            time_label=frame['time_label'],
            links=frame['links'],
            node_positions=positions,
            node_values=values
        )

        # Criar figura
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(-padding, plot_width + padding)
        ax.set_ylim(0, plot_height)
        ax.axis('off')
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Renderizar frame
        render_frame_high_quality(
            ax, frame_data, self.layers, self.node_colors,
            plot_width, plot_height, node_width, font_size,
            n_segments, bar_height_ratio, mode in ['stacked_ranking', 'stacked']
        )

        # Titulo
        if title:
            time_label = frame_data.time_label
            full_title = f"{title}\n{time_label}" if time_label else title
            ax.text(
                plot_width / 2, plot_height * 0.98, full_title,
                fontsize=title_fontsize, ha='center', va='top',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=title_bg_color,
                          alpha=title_bg_alpha, edgecolor='none')
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)

        print(f"Frame salvo em: {output_path}")
        return output_path


# Alias
SankeyRaceMultiLayer = SankeyRaceMultiLayerParallel


# =============================================================================
# Teste
# =============================================================================

if __name__ == "__main__":
    mp.freeze_support()

    print("=" * 70)
    print("TESTE: Sankey Multi-Layer PARALLEL")
    print("=" * 70)

    # Dados de teste
    data = []
    exportadores = ["Brazil", "USA", "China", "India"]
    importadores = ["EU", "Japan", "Mexico", "Canada"]

    fluxos = {
        ("Brazil", "EU"): 10, ("Brazil", "Japan"): 8, ("Brazil", "Mexico"): 5,
        ("USA", "EU"): 15, ("USA", "Mexico"): 12, ("USA", "Canada"): 8,
        ("China", "EU"): 20, ("China", "Japan"): 18, ("China", "Mexico"): 10,
        ("India", "EU"): 6, ("India", "Japan"): 4, ("India", "Canada"): 3,
    }

    for ano in range(2020, 2024):
        mult = 1 + (ano - 2020) * 0.1
        for (src, tgt), val in fluxos.items():
            data.append({"ano": ano, "origem": src, "destino": tgt, "valor": val * mult})

    df = pd.DataFrame(data)
    layers = [exportadores, importadores]

    # Criar Sankey
    sankey = SankeyRaceMultiLayerParallel.from_dataframe(
        df=df,
        layers=layers,
        time_col="ano",
        source_col="origem",
        target_col="destino",
        value_col="valor",
    )

    # Gerar animação
    print("\n")
    sankey.animate(
        output_path="teste_multi_parallel.mp4",
        title="Multi-Layer Parallel Test",
        figsize=(16, 10),
        fps=24,
        duration_seconds=5.0,
        quality="medium",
        n_workers=4,
    )

    # Limpar
    try:
        os.remove("teste_multi_parallel.mp4")
    except:
        pass

    print("\n" + "=" * 70)
    print("Teste concluido!")
    print("=" * 70)
