"""
06 - Add a soundtrack
=====================
The finishing touch. A reel feels alive with music, and `animate()` can mux a
soundtrack straight into the MP4 - no video editor needed. This short lesson
covers the few options that matter.

Needs FFmpeg, plus (for the YouTube route) `pip install yt-dlp`.


TWO WAYS TO PROVIDE AUDIO
-------------------------
    audio_path="song.mp3"     a local file on your disk
    audio_url="https://..."   a link the library downloads for you (via yt-dlp)

Use one or the other. With a URL, any "&t=90s" timestamp in the link is ignored
- the full track is fetched and you choose the start point yourself.


TIMING CONTROLS
---------------
    audio_start=30    begin 30 seconds into the track (great for skipping an
                      intro and landing on the "drop")
    audio_fade=1.5    fade in at the start and out at the end, in seconds

The audio is automatically trimmed to the length of your video, so you never
have to match them by hand. A 3-minute song on an 8-second clip just uses the
first 8 seconds (from `audio_start`), faded.


A WORD ON RIGHTS (please read)
------------------------------
Downloaded tracks are often copyrighted. For social platforms with a licensed
in-app music library, prefer that. Muxed audio is handy for platforms without
one - but only use tracks you actually have the right to use. The default URL
below points at a no-copyright lo-fi mix to keep this lesson safe.


Run it:
    python 06_background_music.py                       # uses the no-copyright URL
    python 06_background_music.py path/to/song.mp3      # or your own file
Writes  with_music.mp4
"""
import multiprocessing as mp
import sys

import pandas as pd
from gradient_sankey import SankeyRaceMultiLayerParallel as Sankey

# A little growth story worth scoring: profit climbing from 20 to 95.
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

    # Decide which audio source to use based on the argument you passed.
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg and arg.lower().endswith(".mp3"):
        audio = dict(audio_path=arg, audio_start=0)
    else:
        audio = dict(
            audio_url="https://www.youtube.com/watch?v=-DfHaOYeaqk",   # a no-copyright lo-fi mix
            audio_start=30,        # skip the intro, start 30s in
        )

    sankey.animate(
        "with_music.mp4", title="now with sound",
        theme="dark", link_glow=1, fps=24, duration_seconds=8,
        audio_fade=1.5,            # gentle fade in and out
        **audio,                   # <- audio_path/audio_url + audio_start
    )
    print("Done! Play with_music.mp4")
    print("Reminder: only use tracks you have the rights to.")
    print("\nThat's the whole tutorial path - you can now build real reels.")
    print("Graduate to examples/advanced/ for the complete, production builds.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
