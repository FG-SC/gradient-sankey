"""
06 · Add a soundtrack  🎵
==========================
A reel feels finished with music. `animate()` can mux audio straight into the
MP4 — from a local file, or downloaded from a YouTube URL for you.

Needs FFmpeg, plus (for the YouTube option) `pip install yt-dlp`.

Run it:
    python 06_background_music.py                       # uses a YouTube URL
    python 06_background_music.py path/to/song.mp3      # or your own file

Writes  with_music.mp4
"""
import multiprocessing as mp
import sys

import pandas as pd
from gradient_sankey import SankeyRaceMultiLayerParallel as Sankey

# A little growth story to score.
rows = []
for t, profit in enumerate([20, 35, 55, 80, 95]):
    rows += [{"t": t, "s": "Revenue", "d": "Profit", "v": profit},
             {"t": t, "s": "Revenue", "d": "Costs",  "v": 100 - profit}]
df = pd.DataFrame(rows)


def main():
    sankey = Sankey.from_dataframe(
        df, layers=[["Revenue"], ["Profit", "Costs"]],
        time_col="t", source_col="s", target_col="d", value_col="v",
    )

    # An MP3 path? use audio_path. A URL? use audio_url (downloaded via yt-dlp;
    # any &t= timestamp in the link is ignored — use audio_start to pick the spot).
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    audio = {}
    if arg and arg.lower().endswith(".mp3"):
        audio = dict(audio_path=arg, audio_start=0)
    else:
        audio = dict(
            audio_url="https://www.youtube.com/watch?v=-DfHaOYeaqk",  # a no-copyright lo-fi mix
            audio_start=30,        # start 30s in
        )

    sankey.animate(
        "with_music.mp4", title="now with sound",
        theme="dark", link_glow=1, fps=24, duration_seconds=8,
        audio_fade=1.5,            # gentle fade in/out
        **audio,
    )
    print("Done! Play with_music.mp4")
    print("Note: use tracks you have the rights to - downloaded audio may be copyrighted.")
    print("\nThat's the tutorial path! For the real thing, explore advanced/")


if __name__ == "__main__":
    mp.freeze_support()
    main()
