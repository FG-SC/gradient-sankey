"""Pure-logic tests for palette / color helpers (no rendering)."""
import re

import pytest

from gradient_sankey import (
    get_palette_colors,
    get_colormap,
    interpolate_color,
    get_dynamic_color,
    get_ranking_color,
    ColorPalette,
)

HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def test_get_palette_colors_count_and_hex():
    colors = get_palette_colors(ColorPalette.VIRIDIS, 5)
    assert len(colors) == 5
    assert all(HEX.match(c) for c in colors)


def test_get_palette_colors_single_uses_midpoint():
    one = get_palette_colors(ColorPalette.RAINBOW, 1)
    assert len(one) == 1 and HEX.match(one[0])


def test_get_palette_colors_reverse():
    fwd = get_palette_colors("viridis", 4)
    rev = get_palette_colors("viridis", 4, reverse=True)
    assert fwd == rev[::-1]


def test_get_palette_colors_custom_hex_interpolates():
    colors = get_palette_colors(["#FF0000", "#00FF00"], 3)
    assert len(colors) == 3
    assert colors[0].lower() == "#ff0000"
    assert colors[-1].lower() == "#00ff00"
    assert colors[1].lower() not in ("#ff0000", "#00ff00")  # a true midpoint


def test_get_colormap_invalid_name_falls_back():
    # A bogus name must not raise; it falls back to rainbow.
    cmap = get_colormap("definitely-not-a-colormap")
    assert cmap is not None


def test_interpolate_color_endpoints():
    assert interpolate_color("#000000", "#ffffff", 0.0).lower() == "#000000"
    assert interpolate_color("#000000", "#ffffff", 1.0).lower() == "#ffffff"
    mid = interpolate_color("#000000", "#ffffff", 0.5).lower()
    assert mid not in ("#000000", "#ffffff")


def test_get_dynamic_color_equal_min_max_is_safe():
    # No division by zero when min == max.
    c = get_dynamic_color(5, 5, 5, "RdYlGn")
    assert c.startswith("#")


def test_get_ranking_color_single_item_is_safe():
    c = get_ranking_color(1, 1, "RdYlGn")
    assert c.startswith("#")
