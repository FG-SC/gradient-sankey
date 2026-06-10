"""
02 - Your first animation (the dynamic part)
============================================
In lesson 01 the time column had a single value, so we got one still frame. The
*whole point* of this library is what happens when you give it MORE than one:
it interpolates smoothly between the periods and renders a video.

This is the part people find fiddly, so let's slow down and get it right.


HOW TIME BECOMES MOTION
-----------------------
Remember rule #3 from lesson 01: one frame per distinct value of the time column.
So to animate, you add rows for more time values. Conceptually:

    month 1:  Budget -> Rent 1200,  Food 600,  Savings 400
    month 2:  Budget -> Rent 1200,  Food 700,  Savings 300
    month 3:  Budget -> Rent 1300,  Food 650,  Savings 550

Each month is a "keyframe" - a snapshot of the flows. The library does NOT just
cut between snapshots; it MORPHS between them (a link that goes 600 -> 700 grows
smoothly, a node slides if it changes size). You provide the keyframes; the
in-between motion is automatic.

The data shape is the same long table as before, just with several time values.
The rule of thumb: ONE ROW PER (period, flow). If you have 3 months and 3 flows,
that's 3 x 3 = 9 rows. Below we build those rows with a small comprehension so
you can see the pattern - in real life you'd usually get this straight from a
DataFrame already in this shape (lesson 05).


WHY THE `if __name__ == "__main__"` GUARD?
------------------------------------------
`animate()` renders frames across several CPU processes for speed. On Windows
(and anywhere using the "spawn" start method) each worker process re-imports your
script. Without the guard below, that re-import would re-run your top-level code
- including animate() itself - and the workers would spawn their own workers, and
so on. The guard makes "the script that kicks off rendering" run only in the main
process. ALWAYS wrap animation scripts this way. (Static save_frame() in lesson
01 doesn't need it - no extra processes are involved.)


Run it (needs FFmpeg installed - see the README):
    python 02_first_animation.py
It writes  first_animation.mp4. Then on to 03_themes.py to make it pretty.
"""
import multiprocessing as mp

import pandas as pd
from gradient_sankey import SankeyRaceMultiLayerParallel as Sankey

# Our three keyframes, as plain tuples: (month, rent, food, savings).
keyframes = [
    (1, 1200, 600, 400),
    (2, 1200, 700, 300),
    (3, 1300, 650, 550),
]

# Expand the keyframes into the long "one row per (period, flow)" table. Read
# this as: "for each month, emit one row for each of the three categories."
df = pd.DataFrame([
    {"month": month, "source": "Budget", "target": category, "value": amount}
    for (month, rent, food, savings) in keyframes
    for category, amount in [("Rent", rent), ("Food", food), ("Savings", savings)]
])
# (Peek at `df` in a console if you like - it's 9 rows: 3 months x 3 categories.)


def main():
    sankey = Sankey.from_dataframe(
        df,
        layers=[["Budget"], ["Rent", "Food", "Savings"]],
        time_col="month",       # <- the column with several values is what animates
        source_col="source", target_col="target", value_col="value",
    )

    # fps x duration_seconds = total frames the library renders. The library
    # spreads your 3 keyframes across those frames and fills in the motion.
    #   - more fps  -> smoother (and slower to render)
    #   - longer duration -> the morph between months takes longer / feels calmer
    sankey.animate(
        "first_animation.mp4",
        title="Budget over 3 months",
        fps=24,
        duration_seconds=6,
    )
    print("Done! Play first_animation.mp4")
    print("Notice how month 1 -> 2 -> 3 morphs smoothly - you only gave 3 snapshots.")
    print("Next up: 03_themes.py (make it beautiful).")


if __name__ == "__main__":
    mp.freeze_support()   # required on Windows for the parallel renderer (see docstring)
    main()
