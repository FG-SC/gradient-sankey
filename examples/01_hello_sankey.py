"""
01 - Hello, Sankey
==================
Welcome! This is the most important lesson in the series, because it teaches the
*mental model* the whole library is built on. Get this and everything else is
just variations.

We'll turn a monthly budget into a gradient Sankey diagram and save it as a PNG.


THE BIG IDEA (read this once, slowly)
-------------------------------------
Three concepts and you understand 90% of the library:

1. YOUR DATA IS A TABLE OF FLOWS. Not a tree, not a nested dict - a plain,
   "long" table where every row is one arrow: WHO it leaves, WHO it arrives at,
   and HOW MUCH. Four columns:

       time | source | target | value
       -----+--------+--------+------
          1 | Budget | Rent   | 1200
          1 | Budget | Food   |  600
          1 | Budget | Savings|  400

   (We'll use the `time` column for animation in lesson 02. With a single value
   here, we just get one still picture.)

2. NODES LIVE IN LAYERS. A "layer" is a vertical column of nodes, and you list
   the layers left-to-right. Here we have two columns: the single "Budget" node
   on the left, and the three things it splits into on the right:

       layers = [["Budget"], ["Rent", "Food", "Savings"]]
                  ^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                  layer 0      layer 1

   You do NOT list the links - the library reads those from your table. You only
   declare which nodes sit in which column. A node mentioned in your data must
   appear in exactly one layer, and every node name must be UNIQUE across all
   layers (more on that pitfall below).

3. ONE PICTURE PER TIME VALUE. Each distinct value of the time column is one
   "frame". One value -> one image (this lesson). Several values -> an animation
   (lesson 02). Same data, same code; you just add more rows.

That's the model. Everything else - colours, themes, axes, music - is polish on
top of these three ideas.


A COMMON FIRST PITFALL: unique node names
-----------------------------------------
Because nodes are identified by their *name*, the same name can't appear in two
layers. If "China" is both an exporter and an importer, give them distinct names
like "China (export)" and "China (import)". Otherwise the library can't tell
which column you mean.


Run it:
    python 01_hello_sankey.py
It writes  hello_sankey.png  next to this file. Open it - that's the whole
tutorial. Then move on to 02_first_animation.py to make it move.
"""
import pandas as pd

# We import the main class under a short alias to keep the code readable.
from gradient_sankey import SankeyRaceMultiLayerParallel as Sankey

# ---------------------------------------------------------------------------
# STEP 1 - describe the flows as a long table (one row per arrow).
# ---------------------------------------------------------------------------
# Tip: building this with a list of dicts is the most readable way to start.
# Later you'll often load it from a CSV or an API instead (see lesson 05).
df = pd.DataFrame([
    {"step": 1, "source": "Budget", "target": "Rent",    "value": 1200},
    {"step": 1, "source": "Budget", "target": "Food",    "value": 600},
    {"step": 1, "source": "Budget", "target": "Savings", "value": 400},
])

# ---------------------------------------------------------------------------
# STEP 2 - declare the layers (the columns of nodes, left -> right).
# ---------------------------------------------------------------------------
# Layer 0 is the single source; layer 1 holds the three destinations. The names
# here MUST match the names used in the table above, exactly.
layers = [
    ["Budget"],                      # layer 0 (left)
    ["Rent", "Food", "Savings"],     # layer 1 (right)
]

# ---------------------------------------------------------------------------
# STEP 3 - hand the table + layers to the library, mapping your column names.
# ---------------------------------------------------------------------------
# `from_dataframe` is the one entry point you'll use 99% of the time. The four
# *_col arguments simply tell it which of YOUR columns play which role - so your
# table can use any column names you like.
sankey = Sankey.from_dataframe(
    df,
    layers=layers,
    time_col="step",        # which column separates frames
    source_col="source",    # where each arrow starts
    target_col="target",    # where it ends
    value_col="value",      # how thick it is
)

# ---------------------------------------------------------------------------
# STEP 4 - render one frame to an image.
# ---------------------------------------------------------------------------
# `figsize` is in inches (width, height); `dpi` is sharpness. title is optional.
sankey.save_frame("hello_sankey.png", title="My monthly budget", figsize=(10, 6), dpi=150)

print("Done! Open hello_sankey.png")
print("You just configured nodes, layers and a flow table - the core skill.")
print("Next up: 02_first_animation.py (turn time into motion).")
