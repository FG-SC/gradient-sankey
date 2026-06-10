"""
01 · Hello, Sankey  👋
=======================
Your very first gradient Sankey diagram — about five lines of real code.

A "Sankey" shows how a quantity splits and flows. Here: a monthly budget
flowing from one pot into three. The library's trick is that each link is a
*true colour gradient* from the source's colour to the target's.

Run it:
    python 01_hello_sankey.py
It writes  hello_sankey.png  in this folder. That's the whole tutorial. 🎉
"""
import pandas as pd
from gradient_sankey import SankeyRaceMultiLayerParallel as Sankey

# 1) Your data is just a table: who sends, who receives, how much.
#    (`step` is the time column — with one value here we get a single picture.)
df = pd.DataFrame([
    {"step": 1, "source": "Budget", "target": "Rent",    "value": 1200},
    {"step": 1, "source": "Budget", "target": "Food",    "value": 600},
    {"step": 1, "source": "Budget", "target": "Savings",  "value": 400},
])

# 2) `layers` are the columns of nodes, left → right. The first column is the
#    single "Budget" node; the second column holds the three things it splits into.
sankey = Sankey.from_dataframe(
    df,
    layers=[["Budget"], ["Rent", "Food", "Savings"]],
    time_col="step", source_col="source", target_col="target", value_col="value",
)

# 3) Save one frame as an image. `figsize`/`dpi` control size & sharpness.
sankey.save_frame("hello_sankey.png", title="My monthly budget", figsize=(10, 6), dpi=150)

print("Done! Open hello_sankey.png")
print("Next up -> 02_first_animation.py (make it move).")
