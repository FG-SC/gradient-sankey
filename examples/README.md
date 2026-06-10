# Examples — a gentle, hands-on tour

New here? You're in the right place. These examples are a **learning path**: each
one is short, runs on its own, and teaches exactly one idea. Start at the top and
go down — by the end you'll be making animated, broadcast-quality flow videos.

First, install the library (and FFmpeg if you want video):

```bash
pip install gradient-sankey
# FFmpeg (video/audio export only — PNGs work without it):
#   Windows: choco install ffmpeg   ·   macOS: brew install ffmpeg   ·   Linux: sudo apt install ffmpeg
```

Then run any example with plain `python`, e.g. `python 01_hello_sankey.py`.

## Start here — the learning path

| # | File | You'll learn | Needs |
|---|------|--------------|-------|
| 01 | [`01_hello_sankey.py`](01_hello_sankey.py) | Turn a table into a gradient Sankey **PNG** — 5 lines. | — |
| 02 | [`02_first_animation.py`](02_first_animation.py) | Add time periods → a smooth **MP4** animation. | FFmpeg |
| 03 | [`03_themes.py`](03_themes.py) | The `Theme` system: `dark` / `light` / `editorial`, or your own. | — |
| 04 | [`04_colors.py`](04_colors.py) | Palettes, color-by-position, and value-driven **dynamic colors**. | FFmpeg |
| 05 | [`05_real_company_data.py`](05_real_company_data.py) | Pull a **real company's income statement** and draw it. | `yfinance` |
| 06 | [`06_background_music.py`](06_background_music.py) | Add a **soundtrack** to your reel. | FFmpeg, `yt-dlp` |

Every example is heavily commented and ends by telling you which file it wrote.
Outputs (PNGs/MP4s) are git-ignored, so run them freely — nothing gets committed.

## When you're ready for more

The [`advanced/`](advanced/) folder holds the **complete, real-world builds** the
tutorials lead up to — including the flagship **NVIDIA income-statement reel**
(18 years of real SEC data, dynamic stock overlay, neon theme, soundtrack) and a
full **cookbook** of every feature. They're the "graduation projects"; the
tutorials above teach everything you need to read them comfortably.

## A 30-second mental model

Three ideas and you understand the whole library:

1. **Your data is a table** of flows: `time, source, target, value`.
2. **Layers** are the columns of nodes, left → right (`[["Revenue"], ["Profit", "Costs"]]`).
3. **One frame per time value** — give it several and `animate()` does the rest.

That's it. Have fun. 🎉
