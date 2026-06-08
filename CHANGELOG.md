# Changelog

All notable changes to **gradient-sankey** are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

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
