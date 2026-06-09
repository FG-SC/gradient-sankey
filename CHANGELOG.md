# Changelog

All notable changes to **gradient-sankey** are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [1.2.0] — 2026-06-09 — A first-class `Theme` design system

The library now treats *design* as a first-class concern alongside animation:
how a chart looks is described by one cohesive, named object instead of a dozen
scattered keyword arguments and duplicated presets.

### Added
- **`Theme` design system** — `Theme`, `NodeStyle`, `LinkStyle`, `TypeScale`
  dataclasses. Pass `theme=Theme.dark()` (or `Theme.light()` / `Theme.editorial()`)
  once to `animate()` / `save_frame()`, or build your own:
  ```python
  from gradient_sankey import Theme, NodeStyle, LinkStyle
  look = Theme.dark()
  look.node.corner_radius = 0.18      # pill-shaped nodes
  look.link.glow = 3                  # heavier neon glow
  look.node.label_plate_alpha = 0     # turn the label backing plate off
  sk.animate("reel.mp4", theme=look)
  ```
- **Themeable node geometry & label plate** — node corner radius, padding, edge
  width and the name-label backing plate are now controlled by the theme
  (previously hard-coded in the renderer).
- **`Theme.editorial()`** preset — a clean print look (warm paper, charcoal ink,
  hairline borders, no glow, no label plate).

### Changed
- The `dark` / `light` theme presets are now defined in **one place**
  (`Theme.dark()` / `Theme.light()`); the duplicated preset dicts in `animate()`
  and `save_frame()` were removed.
- `theme=` accepts a `Theme` object as well as a preset name. Every existing
  styling keyword (`bg_color`, `label_color`, `link_glow`, `font_size`, …) still
  works and now acts as an **override** on top of the resolved theme.
- `title_bg_color` now defaults to `None` (resolves from the theme) instead of
  `"wheat"`, so the dark preset's title background is no longer overridden.

### Fixed
- **Node name labels stayed unreadable over bright ribbons** — names now sit on a
  contrast-derived backing plate (dark plate under light text and vice-versa),
  legible on any background. Controlled by `Theme.node.label_plate_alpha`.

> **Backward compatible & pixel-verified:** the existing dark/light output is
> byte-for-byte identical (`max ‖new − old‖∞ = 0/255` on the NVIDIA frame); all
> 32 unit tests pass.

## [1.1.3] — 2026-06-08 — Resilient parallel rendering under load

### Changed
- **Render workers retry transient failures** — under heavy system load an FFmpeg
  pipe can break mid-chunk (`OSError [Errno 22]`). The worker now re-spawns FFmpeg
  and re-renders the chunk (up to 3 attempts with backoff) instead of failing the
  whole render. (Before v1.1.1 such a hiccup was silently swallowed into a
  truncated/corrupt video; v1.1.1 surfaced it; v1.1.3 recovers from it.)
- The NVIDIA reel defaults to `--workers 4` (was 6) and exposes `--workers` so the
  render competes less with other jobs on a busy machine.

## [1.1.2] — 2026-06-08 — Fix dropped small-magnitude links in animations

### Fixed
- **Small flows vanished during animation** — `animate()`'s frame interpolation
  dropped any link whose value fell below an absolute `0.1` threshold. For
  small-magnitude data (e.g. an income statement in `$B`, where a `0.085` flow is
  real) this silently removed flows, leaving nodes with missing in/out "arrows"
  that looked misaligned (visible in the early years of the NVIDIA reel). The
  threshold is now `1e-9` (magnitude-independent): every real link is kept and a
  link fading toward zero simply shrinks smoothly. `save_frame()` was unaffected.

## [1.1.1] — 2026-06-08 — Hot-path vectorization (pixel-identical)

Performance pass on the render hot path. **Output is byte-for-byte unchanged** —
verified by a 13-scene old-vs-new pixel diff (all modes, themes, glow, labels,
$ y-axis, overlay, dynamic colors, n_segments 25/50/100): max Δ = 0/255.

### Changed
- **Vectorized gradient-quad assembly** — the per-segment Python `for` loop that
  built each link's quads is replaced by numpy (`np.stack`/`column_stack` into an
  `(n_segments, 4, 2)` block, concatenated once into a single `PolyCollection`).
  ~4× faster assembly at `n_segments=50`, ~8× at 100; draw order preserved so
  alpha-blended overlaps are identical.
- **Hoisted the Bézier easing factors** (`_bezier_factor`) out of the per-link
  loop — they're frame-invariant.
- **Glow color computed once** per frame instead of a list-comprehension per glow
  layer.

### Fixed
- **Temp-dir leak on Windows** — `_rmtree_robust` retries cleanup so the (empty)
  `sankey_multi_parallel_*` directories no longer accumulate in `%TEMP%`.

## [1.1.0] — 2026-06-08 — Hardening, packaging & repo cleanup

Quality pass over the whole repo. Backward‑compatible except for the module
rename (update imports: `sankey_race_multi_layers_parallel` → `gradient_sankey`).

### Added
- **Packaging** — `pyproject.toml` (PEP 621): `pip install -e .` with optional
  extras `[finance]` (requests + yfinance) and `[audio]` (yt‑dlp). `__version__`.
- **Test suite** — `tests/` (pytest): unit tests for palettes/colors/positions
  and validation, plus ffmpeg‑gated render smoke tests (`-m slow`).
- **Input validation** — clear errors for empty data, missing columns, NaN
  values, duplicate node names across layers, and empty layers.
- `overlay_badge` — configurable corner tag for the overlay (was a hardcoded
  "NVDA"); `save_frame` now also supports `yaxis_node` / `yaxis_suffix`.

### Changed
- **Single drawing path** — the static and parallel renderers now share one
  `_draw_frame`, so a `save_frame` image matches a video frame exactly.
- **Robust rendering** — `n_workers` is capped at the frame count (no more empty
  chunks corrupting short clips), the FFmpeg subprocess and figure are always
  cleaned up, and concat/mux failures raise instead of silently "succeeding".
- **Module renamed** `sankey_race_multi_layers_parallel.py` → `gradient_sankey.py`;
  English docstrings; dead parameters removed; specific (not bare) excepts.
- **NVIDIA example** — derives the missing fiscal **Q4** (FullYear − 9‑month YTD),
  uses fiscal‑quarter labels, retries SEC requests with a proper User‑Agent, and
  **caches** to `nvidia_dre_wide.csv` so renders are reproducible/offline
  (`--refresh` to re‑scrape). The stock overlay degrades gracefully when offline.
- `requirements.txt` is now core‑only; optional deps live in `pyproject.toml`.

### Fixed
- `.gitignore` negation for the shipped demo `.mp4` (an inline comment had
  silently broken the rule).
- Removed abandoned scratch files (`ai_model_boom*`, `poc_dark_frame*`, …).

## [1.0.0] — 2026-06-08 — Major update: animation toolkit, themes, overlays & audio

A big, backward‑compatible expansion. Everything from the first release still
works; the additions below turn the library into a full toolkit for storytelling
videos.

### Added

- **Dark theme & neon glow** — `theme="dark"` preset plus `link_glow`, `link_alpha`,
  and explicit `bg_color` / `label_color` / `node_edge_color` / `title_text_color`.
- **Dynamic node colors** — `dynamic_color_mode`: `static`, `ranking`, `value`,
  `global_value`, `percentile`, and the new **`intensity`** (keeps each node's hue,
  brightens with value). Works with any `ColorPalette`, matplotlib colormap name,
  or custom hex list via `dynamic_colormap`.
- **Absolute scale** — `absolute_scale=True` renders bars in true magnitude across
  frames (the "explosion" effect) instead of per‑frame normalization.
- **Custom value labels & accounting parentheses** — `node_value_labels` (static)
  and `node_value_labels_per_frame` (animation). Negative figures can be shown as
  `(5)`; values that don't fit inside a small bar are printed next to the node name
  so they're never hidden.
- **Dynamic value Y‑axis on a node** — `yaxis_node` / `yaxis_suffix` draw an evolving
  `$`-style ruler (nice round ticks) next to a chosen node.
- **Optional time‑series overlay (bar‑chart‑race style)** — `overlay_series`,
  `overlay_label`, `overlay_color`, `overlay_value_suffix`, `overlay_x_labels`.
  The curve fills a fixed footer band, the time (X) axis evolves, the Y axis is a
  running max, with an evolving "big number". Horizontally aligned to the Sankey.
- **Background music** — `audio_path` (local MP3) **or** `audio_url` (YouTube,
  downloaded via `yt-dlp`, timestamps ignored), plus `audio_start` and `audio_fade`.
  New helper `youtube_to_mp3(url)`.

### Changed

- **Automatic crossing reduction** — link stacking now orders each node's outgoing
  flows by the target's vertical position (and incoming by the source's), so flows
  no longer tangle. Applies to both the static and parallel renderers.
- Full **README** rewrite covering every feature with examples and parameter tables.
- `requirements.txt` documents optional extras (`yfinance`, `yt-dlp`).

### New examples

- `examples/nvidia_dre.py` + `examples/render_nvidia_reel.py` — animated NVIDIA
  income‑statement (P&L) reel from real **SEC EDGAR** data, combining most features
  (waterfall, dynamic Y‑axis, parentheses, stock overlay, dark theme, music). CLI:
  `--start-year`, `--duration`, `--audio`, `--audio-url`, `--audio-start`.
- `examples/gallery.py` — a feature cookbook: input shapes, inter‑/intra‑layer and
  by‑position palettes, the four positioning modes, and dynamic node colors.

## [0.1.0] — Initial release

- True gradient Sankey links (cubic‑Bézier, configurable segments).
- Multi‑layer support, parallel video rendering, static image export.
- Positioning modes (`ranking_mode` × `stacked_mode`) and 10 built‑in palettes.
