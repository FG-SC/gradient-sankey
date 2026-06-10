# Advanced — the complete builds

These are the real, production-grade examples the [tutorials](../README.md) lead
up to. Each combines many features into one finished piece. Read the tutorials
first; then everything here will feel familiar.

> They assume the library is installed (`pip install gradient-sankey`, or
> `pip install -e .` from the repo root for development).

| File | What it is | Extras |
|------|------------|--------|
| [`nvidia_reel.py`](nvidia_reel.py) | **The flagship.** 18 years of NVIDIA's real income statement as a 90-second neon reel — dynamic `$` axis, accounting labels, a split-adjusted stock overlay, and a soundtrack. | FFmpeg, `yfinance`, `yt-dlp` |
| [`nvidia_sec_edgar.py`](nvidia_sec_edgar.py) | **The data layer** behind the reel: scrapes the income statement from the SEC EDGAR XBRL API, derives the missing fiscal Q4, and balances the waterfall. Cached to [`../data/`](../data). | `requests` |
| [`gallery.py`](gallery.py) | **The cookbook** — a frame for every feature: inputs, palettes, positioning modes, dynamic colours. | FFmpeg |
| [`company_financials.py`](company_financials.py) | A P&L-style flow with negative values shown in accounting parentheses. | FFmpeg |
| [`energy_flow.py`](energy_flow.py) | A three-layer energy system (sources → carriers → end use). | FFmpeg |

## Run the flagship

```bash
# Uses the cached SEC data in ../data (offline-friendly):
python nvidia_reel.py --start-year 2009 --duration 90

# Re-scrape the latest filings from SEC EDGAR first:
python nvidia_sec_edgar.py --refresh && python nvidia_reel.py --start-year 2009 --duration 90

# Add a soundtrack from a local file or a YouTube URL:
python nvidia_reel.py --duration 45 --audio-url "https://youtu.be/..." --audio-start 30
```

`python nvidia_reel.py --help` lists every flag (start year, duration, workers, audio).
