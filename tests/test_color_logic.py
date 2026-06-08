"""Pure-logic tests for brightness scaling, text contrast and dynamic colors."""
from matplotlib.colors import to_rgb

from gradient_sankey import (
    scale_brightness,
    get_text_color_for_background,
    DynamicColorMode,
)


def _value(hex_color):
    import colorsys
    r, g, b = to_rgb(hex_color)
    return colorsys.rgb_to_hsv(r, g, b)[2]


def test_scale_brightness_clamps_factor():
    base = "#33E08A"
    assert scale_brightness(base, -1.0) == scale_brightness(base, 0.0)
    assert scale_brightness(base, 2.0) == scale_brightness(base, 1.0)


def test_scale_brightness_floor_keeps_visible():
    # factor 0 still returns a non-black color (floor keeps it visible)
    dim = scale_brightness("#33E08A", 0.0, floor=0.30)
    assert _value(dim) > 0.0


def test_scale_brightness_brighter_with_value():
    base = "#33E08A"
    assert _value(scale_brightness(base, 1.0)) > _value(scale_brightness(base, 0.2))


def test_text_color_contrast():
    assert get_text_color_for_background("#ffffff") == "#000000"
    assert get_text_color_for_background("#000000") == "#FFFFFF"


def test_text_color_bad_input_defaults_black():
    assert get_text_color_for_background("not-a-color") == "#000000"


def test_dynamic_colors_static_returns_copy(tiny_sankey):
    vals = tiny_sankey._compute_node_values(tiny_sankey.frames[0]["links"])
    out = tiny_sankey._compute_dynamic_colors(vals, DynamicColorMode.STATIC)
    assert out == tiny_sankey.node_colors
    assert out is not tiny_sankey.node_colors  # a copy, not the same object


def test_dynamic_colors_ranking_per_layer(tiny_sankey):
    vals = tiny_sankey._compute_node_values(tiny_sankey.frames[-1]["links"])
    out = tiny_sankey._compute_dynamic_colors(
        vals, DynamicColorMode.RANKING, "RdYlGn"
    )
    # every node gets a color
    assert set(out) >= {"A", "B", "X", "Y"}
    assert all(c.startswith("#") for c in out.values())
