"""
03 · Themes & looks  🎨
========================
The *look* of a chart is controlled by one cohesive object: `Theme`. Three are
built in, and you can tweak any field to make your own. This example saves the
same budget four ways so you can compare them side by side.

Run it:
    python 03_themes.py
It writes  theme_dark.png, theme_light.png, theme_editorial.png, theme_custom.png
"""
import pandas as pd
from gradient_sankey import SankeyRaceMultiLayerParallel as Sankey
from gradient_sankey import Theme

df = pd.DataFrame([
    {"step": 1, "source": "Budget", "target": "Rent",    "value": 1200},
    {"step": 1, "source": "Budget", "target": "Food",    "value": 600},
    {"step": 1, "source": "Budget", "target": "Savings",  "value": 400},
])
sankey = Sankey.from_dataframe(
    df, layers=[["Budget"], ["Rent", "Food", "Savings"]],
    time_col="step", source_col="source", target_col="target", value_col="value",
)

# 1) The three built-in presets. Pass a Theme object (or just the string "dark").
for name, theme in [("dark", Theme.dark()), ("light", Theme.light()),
                    ("editorial", Theme.editorial())]:
    glow = 1 if name == "dark" else 0          # a neon glow suits the dark theme
    sankey.save_frame(f"theme_{name}.png", theme=theme, link_glow=glow,
                      title=f"{name} theme", figsize=(10, 6), dpi=130)

# 2) Make your own. Start from a preset and change individual fields. A `Theme`
#    bundles three groups of style: .node, .link and .type.
look = Theme.dark()
look.node.corner_radius = 0.20       # pill-shaped nodes
look.node.label_plate_alpha = 0.0    # no dark plate behind the labels
look.link.glow = 3                   # a lush neon halo
look.link.alpha = 0.8                # slightly more opaque ribbons
look.type.base = 14                  # bigger base font
sankey.save_frame("theme_custom.png", theme=look, title="my own theme",
                  figsize=(10, 6), dpi=130)

print("Done! Compare theme_dark / light / editorial / custom .png")
print("(Tip: `theme=\"dark\"` as a string works too, and any colour keyword still overrides.)")
print("Next up -> 04_colors.py (colour with meaning).")
