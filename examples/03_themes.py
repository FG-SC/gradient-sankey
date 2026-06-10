"""
03 - Themes: controlling the whole look
=======================================
You can now make a diagram and animate it. This lesson is about making it
*beautiful* - and, just as importantly, making a look you can reuse so every
chart in a series matches.

The key idea: appearance is NOT a pile of loose arguments. It lives in one
object, `Theme`, that you pass once.


THE THREE STYLE GROUPS INSIDE A THEME
-------------------------------------
A `Theme` bundles a background/text palette plus three nested style groups. You
rarely need all of them, but it helps to know the map:

    Theme
      .background, .text, .title_text, .title_bg   <- the canvas & text colours
      .node   (a NodeStyle)   -> .width .corner_radius .edge_color
                                 .edge_width .label_plate_alpha
      .link   (a LinkStyle)   -> .alpha .glow .segments
      .type   (a TypeScale)   -> .base .title           (font sizes)

Three ready-made presets ship with the library:

    Theme.dark()       near-black background, light text - the "neon reel" look
    Theme.light()      classic white background (this is the default)
    Theme.editorial()  warm paper, charcoal ink, hairline borders, no glow

You can pass a preset, the string "dark"/"light"/"editorial", OR a Theme you've
tweaked. They're interchangeable.


TWO WAYS TO CUSTOMISE (both shown below)
----------------------------------------
A) Quick override: pass any individual styling keyword (theme="dark",
   link_glow=2, bg_color="#101020", ...). Good for one-off tweaks.

B) Build a Theme: start from a preset, change the fields you care about, and
   reuse it across every render. Good when you want a consistent house style.

Backwards-compatible by design: the old per-argument keywords still work and
simply override whatever the chosen theme set. So nothing you learn here ever
"locks you in".


Run it:
    python 03_themes.py
Writes  theme_dark.png, theme_light.png, theme_editorial.png, theme_custom.png
- open them side by side to feel the difference.
"""
import pandas as pd
from gradient_sankey import SankeyRaceMultiLayerParallel as Sankey
from gradient_sankey import Theme

# Same little budget as before - we're changing only the LOOK, never the data.
df = pd.DataFrame([
    {"step": 1, "source": "Budget", "target": "Rent",    "value": 1200},
    {"step": 1, "source": "Budget", "target": "Food",    "value": 600},
    {"step": 1, "source": "Budget", "target": "Savings", "value": 400},
])
sankey = Sankey.from_dataframe(
    df, layers=[["Budget"], ["Rent", "Food", "Savings"]],
    time_col="step", source_col="source", target_col="target", value_col="value",
)

# --- A) the three built-in presets -------------------------------------------
# We pass a Theme object here, but `theme="dark"` (a string) is identical.
# A neon glow suits the dark theme; on light/editorial we leave it off.
for name, theme in [("dark", Theme.dark()),
                    ("light", Theme.light()),
                    ("editorial", Theme.editorial())]:
    sankey.save_frame(
        f"theme_{name}.png",
        theme=theme,
        link_glow=1 if name == "dark" else 0,   # <- a per-call override (option A)
        title=f"{name} theme",
        figsize=(10, 6), dpi=130,
    )

# --- B) build your own house style -------------------------------------------
# Start from a preset and reach into the nested groups to change exactly what you
# want. This `look` is now a reusable object - pass it to every chart in a deck
# and they'll all match.
look = Theme.dark()
look.node.corner_radius = 0.20        # rounder, "pill"-shaped nodes
look.node.label_plate_alpha = 0.0     # 0 = no dark plate behind node labels
look.link.glow = 3                    # a lush neon halo (0 = none, 1-3 typical)
look.link.alpha = 0.8                 # slightly more opaque ribbons (0-1)
look.type.base = 14                   # bump the base font size up
sankey.save_frame("theme_custom.png", theme=look, title="my own house style",
                  figsize=(10, 6), dpi=130)

print("Done! Compare theme_dark / light / editorial / custom .png")
print("Takeaway: build a Theme once, pass it everywhere, and your series stays consistent.")
print("Next up: 04_colors.py (give the colours meaning).")
