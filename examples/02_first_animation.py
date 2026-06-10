"""
02 · Your first animation  🎬
==============================
Same idea as #01 — but now the numbers change over *time*. Give the library
several time periods and it smoothly interpolates between them and renders an
MP4. (This step needs FFmpeg installed; see the README.)

Run it:
    python 02_first_animation.py
It writes  first_animation.mp4  in this folder.

⚠️ Note the `if __name__ == "__main__"` guard at the bottom: the video renderer
uses several processes in parallel, and on Windows that guard is required so the
worker processes don't re-import and re-run this script. Keep it in your own
animation scripts.
"""
import multiprocessing as mp

import pandas as pd
from gradient_sankey import SankeyRaceMultiLayerParallel as Sankey

# Three months of the same budget. Each month is one "frame"; the library fills
# in the smooth motion between them for you.
monthly = [
    # (month, rent, food, savings)
    (1, 1200, 600, 400),
    (2, 1200, 700, 300),
    (3, 1300, 650, 550),
]
df = pd.DataFrame([
    {"month": m, "source": "Budget", "target": cat, "value": val}
    for (m, rent, food, sav) in monthly
    for cat, val in [("Rent", rent), ("Food", food), ("Savings", sav)]
])


def main():
    sankey = Sankey.from_dataframe(
        df,
        layers=[["Budget"], ["Rent", "Food", "Savings"]],
        time_col="month", source_col="source", target_col="target", value_col="value",
    )
    sankey.animate(
        "first_animation.mp4",
        title="Budget over 3 months",
        fps=24,                 # frames per second
        duration_seconds=6,     # how long the whole clip lasts
    )
    print("Done! Play first_animation.mp4")
    print("Next up: 03_themes.py (make it beautiful).")


if __name__ == "__main__":
    mp.freeze_support()   # required on Windows for the parallel renderer
    main()
