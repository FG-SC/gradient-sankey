# -*- coding: utf-8 -*-
"""Builds the energy tutorial (03_tutorial_energy.ipynb) with nbformat.

SELF-CONTAINED: the generated notebook depends ONLY on `gradient_sankey` (the
committed module at the repo root, added via sys.path), the Python stdlib, and
pandas. It does NOT import or reference any local-only project folder -- the OWID
CSV download, the corrected (non-double-counted) per-source SOURCES dict, the
wide->tidy melt, the source->continent flow table, and the colours/layers/overlay
are all INLINED as teaching code (this is a reshape-a-wide-CSV lesson, so showing
the code is the point).

Our World in Data energy CSV (one clean download, disk-cached to notebooks/.nbcache/)
-> explore -> reshape wide per-source TWh columns into tidy
[year, source, continent, value] long form -> from_dataframe -> save_frame stills +
one short animate() with the dark/neon look, the dynamic TWh axis, and the
total-generation overlay.

This is the lesson on RESHAPING A WIDE CSV (melt) and on a FIXED brand-coloured
source->continent flow (the counterpoint to the refugee reel's dynamic colours).

    python build_tutorial_energy.py
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

# =====================================================================
# 1. Title / what we'll build
# =====================================================================
md(r"""
# The energy transition → a source‑to‑continent reel
### Your fourth end‑to‑end project with **gradient‑sankey** — built *com todo cuidado e carinho* 💙

Welcome back! 👋 This is the **fourth** lesson in our onboarding series. The previous ones pulled
data from JSON APIs; this one starts from a single, beautifully clean **CSV** and teaches the move
you'll reach for constantly in the real world: **reshaping a wide table into the tidy long form** the
library wants.

We'll reproduce, **end to end**, the **energy‑transition reel**: real **Our World in Data**
electricity figures flowing from **generation source → continent**, animated year by year from
**2000 → the latest complete year**. Ribbon width is **terawatt‑hours (TWh)**, so you literally watch
**coal stay gigantic in Asia** while **wind + solar erupt** out of almost nothing.

> The earlier tutorials (`tutorial_stablecoins.ipynb`, `tutorial_nvidia_income.ipynb`,
> `tutorial_refugees.ipynb`) taught the tidy `[time, source, target, value]` contract, the dynamic
> top‑N, the "Others" bucket, and dynamic rank colours. Here we reuse the contract and add two new
> ideas: starting from a **wide CSV** (one `melt` away from tidy) and a **fixed, brand‑coloured**
> source → continent flow — the calm counterpoint to the refugee reel's screaming rank colours.

---

### What we'll build

A **stacked ranking race** with two columns: on the **left**, the **generation sources** (Coal, Gas,
Oil, Nuclear, Hydro, Wind, Solar, Bioenergy, Other); on the **right**, the **continents** where it's
generated (Asia, North America, Europe, …). Ribbons flow source → continent, their width is
**TWh**, and **every frame is one year**. A footer chart tracks **total world generation**, and a
live **TWh axis** scales itself to the source column.

### Learning outcomes

1. Load the **free OWID energy dataset** (one CSV, no key) and find the right **grain** (continent
   aggregate rows vs. real countries vs. `World`).
2. **Reshape** a wide table (`coal_electricity`, `gas_electricity`, …) into the tidy long form
   `[year, source, continent, value]` with a single `pandas.melt`.
3. Give each source a **brand‑ish neon colour** and drive a **fixed‑colour** (not rank‑coloured)
   source → continent flow.
4. Use the **dynamic TWh value axis** (`yaxis_node` + `yaxis_suffix=" TWh"` + `value_prefix=""`),
   `layer0_label_side`, and the **total‑generation overlay** with `overlay_band`.
5. Read the **transition** in the picture: coal entrenched in Asia, the wind+solar explosion.

> **This notebook is fully self‑contained.** It clones‑and‑runs from the gradient‑sankey repo: it
> imports only `gradient_sankey` (the committed module at the repo root), the Python standard library,
> and `pandas`. The OWID download, the corrected source columns, the `melt`, the flow table and every
> styling knob are **inlined below as teaching steps**, so you can read the whole pipeline end to end. 💙
""")

# =====================================================================
# 2. Setup & install
# =====================================================================
md(r"""
## 1 · Setup & install

The library lives in this repo as a single module, `gradient_sankey.py`, at the **repo root**. This
notebook lives in `notebooks/`, so the repo root is one level up — we add it to `sys.path` so we
always get the **local** copy with the newest features.

```bash
pip install gradient-sankey        # the public package
# this notebook uses the LOCAL repo copy via sys.path (see below)
```

> ⚠️ **Heads‑up — local‑only features (newer than the pinned PyPI release).** This reel leans on a
> few capabilities that already exist in the **local** `gradient_sankey.py` but are **not yet on the
> published pip package**:
> - the **layer‑total value axis** (`yaxis_node`) with **`yaxis_suffix=" TWh"`** and
>   **`value_prefix=""`** — an energy axis, not dollars,
> - **`yaxis_gap`** and **`layer0_label_side`** to place the axis and the node labels,
> - the **`overlay_band`** control (push the footer line chart lower, clear of the Sankey),
> - the overlay styling kwargs (`overlay_value_suffix`, `overlay_badge`, `overlay_x_labels`).
>
> Importing the local module via `sys.path` (below) makes all of these work today. 🎁

We only need `pandas` for the data (plus the standard library to fetch the CSV), and the library for
the visuals. Rendering an **MP4** also needs **FFmpeg** on your PATH — but the still frames
(`save_frame`) need nothing extra, so the notebook stays runnable even without FFmpeg.
""")

code(r"""import os, sys, pathlib, urllib.request

# --- import the LOCAL gradient_sankey from the repo root (committed module) ---
# This notebook lives in notebooks/, so the repo root is one level up.
REPO_ROOT = os.path.abspath("..")          # notebooks/  ->  repo root
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd
from IPython.display import Image, display

import gradient_sankey as gs
from gradient_sankey import SankeyRaceMultiLayerParallel as Sankey

# where we'll drop the rendered stills / sample clip (kept out of the repo root)
ASSETS = os.path.abspath("gallery_assets"); os.makedirs(ASSETS, exist_ok=True)

# a LOCAL, gitignored cache for the one OWID download, so a re-run is instant.
CACHE = pathlib.Path(os.path.abspath(".nbcache")); CACHE.mkdir(parents=True, exist_ok=True)

def show(path, w=900):
    return Image(filename=path, width=w)

print("gradient-sankey version:", gs.__version__)
print("pandas:", pd.__version__)
print("cache ->", CACHE)
""")

# =====================================================================
# 3. OWID primer
# =====================================================================
md(r"""
## 2 · A 2‑minute OWID primer — and the *grain* trap

**[Our World in Data](https://ourworldindata.org/energy)** maintains a superb, openly licensed energy
dataset. It's a single CSV — no API key, no pagination, no auth:

```
https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv
```

Each row is a `(country, year)` pair with **~130 columns**. We want the **per‑source electricity
generation in TWh**:

| Column | Source label we'll show |
|---|---|
| `coal_electricity` | Coal |
| `gas_electricity` | Gas |
| `oil_electricity` | Oil |
| `nuclear_electricity` | Nuclear |
| `hydro_electricity` | Hydro |
| `wind_electricity` | Wind |
| `solar_electricity` | Solar |
| `biofuel_electricity` | Bioenergy |
| `other_renewable_exc_biofuel_electricity` | Other |

> 💡 **Why the `_exc_biofuel_` variant?** Per the OWID codebook, the plain
> `other_renewable_electricity` column **already includes bioenergy**, so pairing it with
> `biofuel_electricity` would **double‑count**. We use `other_renewable_exc_biofuel_electricity`
> (bioenergy excluded) so each TWh is counted exactly once.

> ### ⚠️ The grain trap
> The `country` column is **not** only countries! It also holds **aggregate rows**: `World`,
> continents (`Asia`, `Europe`, `North America`, `South America`, `Africa`, `Oceania`), income groups,
> and more. If you naively `groupby` everything you'll **double‑count** — a French wind farm would be
> counted in *France*, in *Europe*, **and** in *World*. We sidestep this by using **exactly one
> grain**: the **continent aggregate rows**. Their per‑source TWh sum to **approximately** the `World`
> row, so the right‑hand column is a tidy handful of nodes and the totals stay honest. (The small
> residual — rounding plus a few minor uncounted sources — we'll show openly against OWID's
> authoritative `electricity_generation` total in the next step.)
""")

# =====================================================================
# 4. Load + explore
# =====================================================================
md(r"""
## 3 · Load the CSV (cached) and explore

One download, cached to disk so a re‑run is instant. Then we peek at the grain so the next step is
concrete.
""")

code(r'''OWID_URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"

def load_owid():
    """Download (disk-cached) the OWID energy CSV and return it as a DataFrame."""
    cf = CACHE / "owid-energy-data.csv"
    if not cf.exists():
        print("downloading OWID energy CSV (~9 MB) ...")
        urllib.request.urlretrieve(OWID_URL, cf)
    return pd.read_csv(cf)

raw = load_owid()
print("rows x cols:", raw.shape)
elec_cols = [c for c in raw.columns if c.endswith("_electricity")]
print("\nper-source electricity columns available:")
print("  " + ", ".join(elec_cols))
''')

md(r"""
Let's prove the **grain trap** is real — and confirm our six continents are present for every year we
care about. Notice how `World` is roughly the **sum** of the continents, which is exactly why mixing
grains would double‑count.
""")

code(r"""SRC_COLS = ["coal_electricity", "gas_electricity", "oil_electricity",
            "nuclear_electricity", "hydro_electricity", "wind_electricity",
            "solar_electricity", "biofuel_electricity",
            "other_renewable_exc_biofuel_electricity"]
CONTINENTS = ["Asia", "North America", "Europe", "South America", "Africa", "Oceania"]

# the grain trap, made concrete: sum(continents) ~= the World row (so don't mix them).
# We also print OWID's authoritative `electricity_generation` World total alongside our
# summed sources, so the small honest residual (rounding + uncounted minor sources) shows.
for y in (2000, 2024):
    conts = raw[(raw.country.isin(CONTINENTS)) & (raw.year == y)][SRC_COLS].sum().sum()
    world = raw[(raw.country == "World") & (raw.year == y)][SRC_COLS].sum().sum()
    world_auth = raw[(raw.country == "World") & (raw.year == y)]["electricity_generation"].sum()
    print(f"  {y}: sum(continents) = {conts:,.0f} TWh   vs   World(sum sources) = {world:,.0f} TWh"
          f"   vs   World electricity_generation = {world_auth:,.0f} TWh")

# every continent reports every year in our window?
present = (raw[raw.country.isin(CONTINENTS)]
          .query("2000 <= year <= 2024")
          .groupby("year")["country"].nunique())
print("\ncontinents present per year (want 6):", present.min(), "->", present.max())
""")

# =====================================================================
# 5. Reshape (the lesson)
# =====================================================================
md(r"""
## 4 · Reshape wide → tidy long (the heart of this lesson) 🔑

The library wants a **tidy long** frame: one row per flow, columns
`[year, source, continent, value]`. Our CSV is **wide**: each source is its *own column*. The bridge
between them is a single **`pandas.melt`** — the move you'll use again and again.

We:
1. keep only the **continent** rows and our **year window**,
2. fill source gaps with `0.0` (a continent with no oil that year is a real zero, not missing),
3. `melt` the nine source columns into two columns: a `src_col` name and its `value`,
4. map `coal_electricity → "Coal"` etc., rename `country → continent`, and
5. group to one clean row per `(year, source, continent)`.
""")

code(r"""Y0, Y1 = 2000, 2024

sub = raw[raw.country.isin(CONTINENTS) & raw.year.between(Y0, Y1)].copy()
sub[SRC_COLS] = sub[SRC_COLS].fillna(0.0)

# the one move that matters: wide source columns -> tidy long rows
long = sub.melt(id_vars=["country", "year"], value_vars=SRC_COLS,
                var_name="src_col", value_name="value")

# pretty source labels + tidy column names
SRC_LABEL = {
    "coal_electricity": "Coal", "gas_electricity": "Gas", "oil_electricity": "Oil",
    "nuclear_electricity": "Nuclear", "hydro_electricity": "Hydro",
    "wind_electricity": "Wind", "solar_electricity": "Solar",
    "biofuel_electricity": "Bioenergy",
    "other_renewable_exc_biofuel_electricity": "Other",
}
long["source"] = long["src_col"].map(SRC_LABEL)
long = long.rename(columns={"country": "continent"})[["year", "source", "continent", "value"]]
long = long.groupby(["year", "source", "continent"], as_index=False)["value"].sum()
long = long[long["value"] > 0]                       # drop empty flows

print(f"{len(long)} tidy [year, source, continent, value] rows, {long.year.min()}-{long.year.max()}")
long.head(8)
""")

md(r"""
### A first read of the transition

Before we draw anything, let's confirm the **story** is in the numbers. Watch **coal in Asia** climb
while **wind + solar in Asia** explode from a rounding error to a juggernaut — the whole reel in two
print statements.
""")

code(r"""asia = long[long.continent == "Asia"]
print("Asia — coal vs (wind + solar), TWh:")
for y in (2000, 2010, 2015, 2020, 2024):
    coal = asia[(asia.year == y) & (asia.source == "Coal")]["value"].sum()
    ws   = asia[(asia.year == y) & (asia.source.isin(["Wind", "Solar"]))]["value"].sum()
    print(f"  {y}: coal {coal:7.0f}   wind+solar {ws:7.0f}")

# world solar+wind share of generation, then vs now
print("\nWorld solar+wind share of all generation:")
for y in (2000, 2024):
    tot = long[long.year == y]["value"].sum()
    ws  = long[(long.year == y) & (long.source.isin(["Wind", "Solar"]))]["value"].sum()
    print(f"  {y}: {100 * ws / tot:4.1f}%   (total {tot:,.0f} TWh)")
""")

# =====================================================================
# 6. Layers + colours
# =====================================================================
md(r"""
## 5 · Layers and brand colours

Two layers: **sources on the left**, **continents on the right**. Unlike the refugee reel — where we
threw away the base colours and let **rank** repaint everything — here the colours **carry meaning**,
so we give each source a fixed, brand‑ish neon: **coal = gray, gas = orange, solar = yellow, wind =
teal, hydro = blue, nuclear = violet**, and so on. The continents get their own accent colours.

We order the right column by each continent's **all‑time total**, so the biggest (Asia) leads.
""")

code(r"""# brand-ish neon colour per generation source (the corrected, non-double-counted SOURCES)
SRC_COLOR = {
    "Coal": "#9AA0A6", "Gas": "#F5871F", "Oil": "#8B5A2B", "Nuclear": "#B36BFF",
    "Hydro": "#3FA9F5", "Wind": "#2DD4BF", "Solar": "#FFD400", "Bioenergy": "#7CFF6B",
    "Other": "#6B7280",
}
CONT_COLOR = {
    "Asia": "#FF5D73", "North America": "#4EA8DE", "Europe": "#9B5DE5",
    "South America": "#00BBF9", "Africa": "#F4A261", "Oceania": "#80ED99",
}

# left layer in our declared (visual) source order; right layer by all-time total
sources = [s for s in SRC_COLOR if s in set(long.source)]
continents = (long.groupby("continent")["value"].sum()
              .reindex(CONTINENTS).dropna()
              .sort_values(ascending=False).index.tolist())
LAYERS = [sources, continents]

node_colors = {**{s: SRC_COLOR[s] for s in sources},
               **{c: CONT_COLOR[c] for c in continents}}

print("sources    (left) :", sources)
print("continents (right):", continents)
""")

# =====================================================================
# 7. Overlay
# =====================================================================
md(r"""
## 6 · The total‑generation overlay

The footer chart is a bar‑chart‑race style line of **total world electricity generation**, one point
per year — the global pie growing under the shifting mix. We also build the **year tick labels**.
""")

code(r"""years = sorted(long["year"].unique())
total = [float(long[long.year == y]["value"].sum()) for y in years]   # world TWh each year
xlabels = [str(y) for y in years]
print(f"world generation: {total[0]:,.0f} TWh ({years[0]}) -> {total[-1]:,.0f} TWh ({years[-1]})")
""")

# =====================================================================
# 8. Build the visual
# =====================================================================
md(r"""
## 7 · Building the visual — every knob explained

We hand the tidy `long` + `LAYERS` to `from_dataframe`, then drive the render with the reel's design.
Here's what each knob does in **this** reel:

| Knob | Value | Why |
|---|---|---|
| `ranking_mode` | `True` | reorder sources/continents by value each frame — *the race* |
| `stacked_mode` | `True` | bar heights scale with TWh (biggest = tallest) |
| `theme` | `"dark"` | the neon‑on‑black reel look |
| `link_glow` / `link_alpha` | `1` / `0.55` | soft neon glow behind the ribbons |
| `n_segments` | `100` | smooth gradients along each ribbon |
| `yaxis_node` | `sources[0]` ⚠️ | draw a value axis scaled to the **source‑layer total** |
| `yaxis_suffix` / `value_prefix` | `" TWh"` / `""` ⚠️ | ticks read like `5000 TWh` — **energy, not dollars** |
| `yaxis_gap` / `layer0_label_side` | `0.15` / `"right"` ⚠️ | axis hugs the source column; names on the right |
| `overlay_series` / `overlay_x_labels` | total / years | the footer race chart |
| `overlay_band` | `(0.18, 0.56)` ⚠️ | push the line chart **lower**, clear of the Sankey |

> 💡 **Fixed colours, on purpose.** We *don't* pass `dynamic_color_mode` here, so every source keeps
> its brand colour all the way through. (Pass `dynamic_color_mode="ranking"` if you ever want the
> refugee‑reel rank colouring instead — try it as an exercise.)

First, build the renderer — the single call that ingests everything.
""")

code(r"""sk = Sankey.from_dataframe(
    df=long, layers=LAYERS,
    time_col="year", source_col="source", target_col="continent", value_col="value",
    node_colors=node_colors,
)
anchor = sources[0]    # any source identifies the left layer for the TWh axis
print(f"{len(sk.frames)} frames (years):", sk.frames[0]['time_label'], "->", sk.frames[-1]['time_label'])
print("TWh axis anchored on the source layer of:", anchor)
""")

# =====================================================================
# 9. Render a short proof
# =====================================================================
md(r"""
## 8 · Render a short proof

Stills first — `save_frame()` needs **no FFmpeg**, so this always works. We render three years that
bookend the transition:

- **2000** — coal & hydro dominate, wind+solar are a rounding error,
- **2012** — gas surges, the renewables wedge starts to show,
- **2024** — wind + solar are unmistakable, coal still towering over Asia.

The brand colours, the **TWh axis** (`yaxis_node` + `value_prefix=""`) and `layer0_label_side` all
work in stills.
""")

code(r"""still_design = dict(
    figsize=(16, 9), dpi=95, font_size=13, title_fontsize=18, padding=2.8,
    margin_top=0.16, margin_bottom=0.20,
    ranking_mode=True, stacked_mode=True,            # the race: reorder AND resize
    theme="dark", link_glow=1, link_alpha=0.55, n_segments=100,
    yaxis_node=anchor, yaxis_suffix=" TWh",          # TWh axis, scaled to the source-layer TOTAL
    yaxis_gap=0.15, layer0_label_side="right",       # axis hugs the sources; names on the right
    value_prefix="",                                 # energy, not dollars
)

year_to_index = {f["time_label"]: i for i, f in enumerate(sk.frames)}
still_paths = []
for yr in ("2000", "2012", "2024"):
    if yr not in year_to_index:
        continue
    p = os.path.join(ASSETS, f"energy_{yr}.png")
    sk.save_frame(p, frame_index=year_to_index[yr],
                  title="World electricity generation — source to continent  (TWh / year)",
                  **still_design)
    still_paths.append(p)
    print("saved", p)

for p in still_paths:
    display(show(p, w=1000))
""")

md(r"""
### One short animation 🎬

A tiny clip to prove the motion, with the **full** reel design: dark theme, glow, the TWh axis, and
the total‑generation overlay. We keep it short for speed (a few seconds). The cell **skips
gracefully** if FFmpeg isn't installed — but if it is, you'll see the wind+solar wedge swell in real
time.
""")

code(r"""sample_mp4 = os.path.join(ASSETS, "energy_sample.mp4")
try:
    sk.animate(
        sample_mp4,
        figsize=(16, 9), fps=24, duration_seconds=8, quality="medium", n_workers=2,
        title="World electricity generation — source to continent  (TWh / year)",
        font_size=13, title_fontsize=20, padding=2.8,
        margin_top=0.16, margin_bottom=0.20,
        ranking_mode=True, stacked_mode=True,
        theme="dark", link_glow=1, link_alpha=0.55, n_segments=100,
        yaxis_node=anchor, yaxis_suffix=" TWh", yaxis_gap=0.15, layer0_label_side="right",
        value_prefix="",                                              # energy, not dollars
        overlay_series=total, overlay_x_labels=xlabels,
        overlay_label="Total world electricity generation  (TWh / year)",
        overlay_color="#7CFF6B", overlay_value_suffix=" TWh", overlay_badge="WORLD",
        overlay_band=(0.18, 0.56),                                    # push the line chart lower
    )
    from IPython.display import Video
    print("rendered:", sample_mp4)
    display(Video(sample_mp4, embed=True, width=900))
except Exception as e:
    print("Skipping the MP4 (FFmpeg likely not installed) — the stills above tell the story.")
    print("Reason:", repr(e))
""")

md(r"""
### The full reel — one call

Everything above (the inlined `load_owid`, the `melt`, the colours, the overlay and the `animate`
call) is the whole pipeline. To render the full‑length reel, just lengthen `duration_seconds` and pick
the year window you want — for example the cell below renders the complete **2000 → latest** story.
Stamp the output filename with a **big‑endian, sortable** date‑time prefix (`%Y-%m-%d-%Hh%M`, `:` → `h`
because it's illegal in paths) so your renders always sort chronologically. 🗂️

```python
from datetime import datetime
stamp = datetime.now().strftime("%Y-%m-%d-%Hh%M")     # sortable, path-safe
out = os.path.join(ASSETS, f"{stamp}_energy.mp4")
sk.animate(out, figsize=(16, 9), fps=30, duration_seconds=30, quality="high", **{...})
```

> Want the **rank‑coloured** look (like the refugee reel) instead of fixed brand colours? Pass
> `dynamic_color_mode="ranking", dynamic_colormap="plasma"` to `animate` and drop the brand
> `node_colors`. Want a wider window? Change `Y0, Y1` in §4 and re‑run from there. 🎛️
""")

# =====================================================================
# 10. Recap
# =====================================================================
md(r"""
## 9 · Recap & try it yourself

🎉 **You built the whole reel** — from a raw 130‑column CSV to an animated gradient Sankey that shows
the global electricity transition, source by source, continent by continent.

**The pipeline, in one breath:** download the OWID CSV (cached) → pick the **continent grain** (avoid
the double‑counting trap) → **`melt`** the wide source columns into tidy `[year, source, continent,
value]` → fixed **brand colours** per source → **total‑generation** overlay → `from_dataframe` →
`save_frame` stills + `animate` with the dark theme, the **TWh axis** (`value_prefix=""`,
`yaxis_suffix=" TWh"`) and `overlay_band`.

### 🧪 Try it yourself

1. **Rank colours.** Pass `dynamic_color_mode="ranking", dynamic_colormap="plasma"` to `animate`
   (and drop the brand `node_colors`) to recolour by live rank, like the refugee reel.
2. **Flip the flow.** Make it **continent → source** by swapping `source_col`/`target_col` and the
   layer order — now the left column is continents and you read *what each continent runs on*.
3. **One country.** Replace the `CONTINENTS` filter with a single country (e.g. `["China"]` or
   `["United States"]`) to tell that nation's transition story — same `melt`, same knobs.
4. **Primary energy.** OWID also has `*_consumption` columns (primary energy, not just electricity).
   Point the `melt` at those for a much bigger, oil‑heavy picture.
5. **A different window.** Set `Y0 = 1985` to include the nuclear build‑out of the late 20th century,
   or `Y0 = 2010` to zoom into the solar decade.

### 🔗 Links

- The library: `gradient_sankey.py` (repo root) · `pip install gradient-sankey`
- Our World in Data — Energy: https://ourworldindata.org/energy  (CSV is free, no API key)
- The raw CSV: https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv
- Companion tutorials: `notebooks/06_tutorial_refugees.ipynb`, `notebooks/02_tutorial_stablecoins.ipynb`,
  `notebooks/01_tutorial_nvidia_income.ipynb`

> 💡 Remember: the **TWh value axis** (`yaxis_node` + `value_prefix=""` + `yaxis_suffix=" TWh"`),
> `yaxis_gap`, `layer0_label_side` and `overlay_band` are **local‑only** today — they're **newer than
> the pinned PyPI release**. Until then, import the local module via `sys.path` (as we did in §1) and
> everything just works. 💙

*Built com todo cuidado e carinho. Data: Our World in Data — Energy (free, no API key). Happy hacking!*
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "03_tutorial_energy.ipynb")
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
