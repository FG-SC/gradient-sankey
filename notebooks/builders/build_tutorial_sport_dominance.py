# -*- coding: utf-8 -*-
"""Builds the "Who owns each Olympic sport" tutorial (10_tutorial_sport_dominance.ipynb).

SELF-CONTAINED: the generated notebook depends ONLY on `gradient_sankey` (the
committed module at the repo root, added via sys.path), the Python stdlib, and
pandas. It does NOT import or reference the local-only daily_posts/ folder at all
-- the public athlete_events.csv download, the GOLD-only de-dup, the curated
sports list, and the sport->nation rolling-window dominance flow builder are all
INLINED below as teaching code (this is an ETL lesson, so showing the code is
the point).

The pipeline, end to end: the "120 years of Olympic history" athlete_events.csv
(downloaded from the public GitHub raw URL, cached to notebooks/.nbcache/) ->
keep GOLD only -> de-dup to one gold per (Games, Event, NOC) -> curated iconic
sports -> per sport keep the top nations + a pinned neutral "Others" -> a ROLLING
3-Games window (so ownership changes hands) -> from_dataframe -> animate() with
dynamic_color_mode="ranking" -> save_frame stills + one short clip.

This is the lesson on GOLD-only "ownership", a ROLLING window that lets dominance
change hands, and the HONESTY of keeping dissolved nations (URS, GDR, EUN)
distinct from their modern successors (UNLIKE the medal-race notebook, which
stitches them into one continuous entity).

    python build_tutorial_sport_dominance.py
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
# Who **owns** each Olympic sport? → a gold‑medal ownership reel
### A hands‑on project with **gradient‑sankey** — built *com todo cuidado e carinho* 💙

Welcome! 👋 This lesson asks a fun, slightly contentious question: for the great Summer Olympic
disciplines, **which nation owns the GOLD medals** — and how does that ownership **change hands** across
eras? We build a 2‑column Sankey that flows **sport → nation**, where a ribbon's width is the **number
of that sport's golds** a nation has won. Every frame is one **Games**, and the nation column is
**recoloured by live rank**, so the owner of the moment **burns hot**.

> If you've done `08_tutorial_olympics.ipynb`, the **de‑duplication** and **dynamic rank colouring** will
> feel familiar. This lesson reuses both ideas, but reframes the story around two new ones that are the
> whole point: **(1)** a **rolling window** so dominance can *change hands*, and **(2)** an honesty rule —
> we keep **dissolved nations distinct**, because *"the USSR owned gymnastics"* is exactly what we want
> to show. *(That's the opposite choice from the medal‑race notebook, which **stitches** URS → Russia and
> the two Germanys into one continuous line.)*

---

### What we'll build

A **stacked ranking race** with two columns: on the **left**, a curated set of **iconic sports**
(Athletics, Swimming, Gymnastics, Cycling, Wrestling, Boxing, Weightlifting, Rowing, Fencing, Table
Tennis, Shooting, Canoeing); on the **right**, the **nations** — *the racers*, recoloured every frame by
their **live rank**. Each sport keeps its **top few owners** and folds the rest into a neutral, pinned
**"Others"**, and a footer chart tracks the **golds in play** in the current era.

### Learning outcomes

1. Reuse the **"120 years of Olympic history"** loader and the **de‑duplication** — but keep **GOLD
   only** (the cleanest signal of *ownership*), de‑duped to one gold per `(Games, Event, NOC)`.
2. Build a **sport → nation** flow and read it as *"who owns this discipline"*.
3. Use a **rolling window** of the last few Games so ownership visibly **changes hands** (gymnastics
   URS → USA/CHN; table tennis seized by CHN from 1988).
4. Per sport, keep the **top‑N owners** + a neutral pinned **"Others"**.
5. **The honesty rule:** keep **URS / GDR / EUN** as *distinct historical nations* — never merged into
   RUS/GER — and **verify a known monopoly** straight from the data (Table Tennis = CHN, Swimming = USA).

> **This notebook is fully self‑contained.** It clones‑and‑runs from the gradient‑sankey repo: it imports
> only `gradient_sankey` (the committed module at the repo root), the Python standard library, and
> `pandas`. The public dataset download and every helper are **inlined below as a teaching step**, so you
> can read the whole pipeline end to end. 💙
""")

# =====================================================================
# 2. Setup & install
# =====================================================================
md(r"""
## 1 · Setup & install

The library lives in this repo as a single module, `gradient_sankey.py`, at the **repo root**. We add the
repo root to `sys.path` so we always get the **local** copy with the newest features.

```bash
pip install gradient-sankey        # the public package
# this notebook uses the LOCAL repo copy via sys.path (see below)
```

> ⚠️ **Heads‑up — local‑only features.** This reel leans on capabilities present in this **local** copy
> of `gradient_sankey.py` that may not be on your **published pip package**:
> - **`dynamic_color_mode="ranking"`** + **`dynamic_colormap`** — recolour each nation by its live rank,
> - pinning a node named exactly **`"Others"`** to the **bottom** and keeping it neutral,
> - the **layer‑total value axis** (`yaxis_node`) with **`value_prefix=""`** (golds, not dollars), plus
>   **`yaxis_gap`** and **`layer0_label_side`**,
> - the **`overlay_band`** control (push the footer chart lower, clear of the Sankey).
>
> Importing the local module via `sys.path` (below) makes all of these work today. 🎁

We only need `pandas` for the data, and the library for the visuals. Rendering an **MP4** also needs
**FFmpeg** on your PATH — but the still frames (`save_frame`) need nothing extra.
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

# a LOCAL, gitignored cache for the one CSV download, so a re-run is instant
# (kept under notebooks/.nbcache/, self-contained)
CACHE = pathlib.Path(os.path.abspath(".nbcache")); CACHE.mkdir(parents=True, exist_ok=True)

def show(path, w=1000):
    return Image(filename=path, width=w)

print("gradient-sankey version:", gs.__version__)
print("pandas:", pd.__version__)
print("cache ->", CACHE)
""")

# =====================================================================
# 3. Dataset primer + GOLD-only framing
# =====================================================================
md(r"""
## 2 · The dataset, and why **GOLD only**

We use the classic
**["120 years of Olympic history"](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results)**
dataset (`athlete_events.csv`), every modern Games from **Athens 1896 → Rio 2016**, one CSV, no key:

```
https://raw.githubusercontent.com/Thomas-George-T/Olympic-History-Analytics-in-R/main/athlete_events.csv
```

The columns we care about: `Year`, `Season`, `Games`, `NOC` (3‑letter country code), `Sport`, `Event`,
and `Medal` (`Gold`/`Silver`/`Bronze`/`NaN`).

> ### Why golds only? 🥇
> "Ownership" reads cleanest as **outright wins**. A bronze means you showed up; a **gold** means you
> *won the event*. So we keep **`Medal == "Gold"`** and tell the story in golds. (Silver/bronze would
> blur the picture and triple the ribbons.)

> ### ⚠️ The grain trap (recap) — *one row per athlete per event*
> Each row is **one athlete in one event**, so a relay/team gold is **many rows**. Counting rows would
> wildly over‑count team sports. The fix is one **de‑duplication** to **one gold per
> `(Games, Event, NOC)`** — one *physical* gold medal per event. (We don't need `Medal` in the key since
> we already filtered to Gold.)
""")

# =====================================================================
# 4. Load + GOLD dedup
# =====================================================================
md(r"""
## 3 · Load (cached) and de‑duplicate to one **physical gold** per event

One download (~40 MB), cached to disk. We fetch the public CSV and apply the GOLD‑only de‑dup — one
physical gold per `(Games, Event, NOC)`.
""")

code(r'''OLYMPICS_URL = ("https://raw.githubusercontent.com/Thomas-George-T/"
                "Olympic-History-Analytics-in-R/main/athlete_events.csv")

def load_olympics():
    """Download (disk-cached) the 120-years athlete_events CSV from the public GitHub mirror."""
    cf = CACHE / "athlete_events.csv"
    if not cf.exists():
        print("downloading athlete_events.csv (~40 MB) ...")
        urllib.request.urlretrieve(OLYMPICS_URL, cf)
    return pd.read_csv(cf)

raw = load_olympics()
print("rows x cols:", raw.shape)

SEASON = "Summer"     # the Summer story

# keep GOLD only, this season; then THE de-dup: one physical gold per (Games, Event, NOC)
g_all = raw[(raw.Season == SEASON) & (raw.Medal == "Gold")].copy()
print(f"{SEASON} GOLD athlete-rows (pre-dedup): {len(g_all):,}")
g_all = g_all.drop_duplicates(subset=["Games", "Event", "NOC"])
print(f"{SEASON} golds after de-dup (one per event-place): {len(g_all):,}")
g_all[["Year", "NOC", "Sport", "Event"]].head(5)
''')

# =====================================================================
# 5. The honesty rule: dissolved nations stay distinct
# =====================================================================
md(r"""
## 4 · The honesty rule — dissolved nations stay **distinct** 🏛️

It's tempting to "tidy up" the codes by merging the **USSR (URS)** into **Russia (RUS)**, **East Germany
(GDR)** into **Germany (GER)**, or the 1992 **Unified Team (EUN)** into RUS. **We do not.** The medals
were *won* by those states, and the whole story — *"the USSR owned gymnastics; East Germany owned the
pool"* — would be **erased** by merging. So URS, GDR, EUN, TCH, etc. appear as their **own** nations,
exactly as the golds were earned.

> ### 🆚 The opposite choice from the medal‑race notebook
> The companion `08_tutorial_olympics.ipynb` deliberately **stitches** these codes (URS → Russia, the two
> Germanys → Germany) so a nation's line doesn't *vanish* mid‑race. Here we deliberately **keep them
> distinct** — because *"who owned this sport in this era"* is the whole point, and a dissolved‑nation
> dynasty is exactly what we want to spotlight, not blur into the modern state. Same data, opposite (and
> equally honest) modelling choice.

Let's *see* it: gymnastics gold by era. The USSR towers before 1992 and then **vanishes**; new owners
(China, USA, Romania) rise after.
""")

code(r"""gym = g_all[g_all.Sport == "Gymnastics"]
pre  = gym[gym.Year <  1992].groupby("NOC").size().sort_values(ascending=False).head(4)
post = gym[gym.Year >= 1992].groupby("NOC").size().sort_values(ascending=False).head(4)
print("Gymnastics golds — pre-1992 (the USSR dynasty):")
print(pre.to_string())
print("\nGymnastics golds — 1992-2016 (after the USSR dissolves):")
print(post.to_string())
print("\n-> URS (71 pre-1992) is its OWN nation; we never fold it into RUS. That IS the story. 🇸🇺")
""")

# =====================================================================
# 6. Verify a known monopoly
# =====================================================================
md(r"""
## 5 · Verify a known monopoly 🔎

Before we trust any of the numbers, we **prove** them against famous facts:

- **Table Tennis** entered the Games in **1988** and has been a near‑total **Chinese** monopoly ever
  since — CHN should hold the overwhelming majority of golds.
- **Swimming** is a permanent **USA** stronghold — far ahead of anyone else all‑time.

If our GOLD de‑dup were wrong, these would not line up.
""")

code(r"""tt = g_all[(g_all.Sport == "Table Tennis") & (g_all.Year >= 1988)]
tt_by = tt.groupby("NOC").size().sort_values(ascending=False)
sw = g_all[g_all.Sport == "Swimming"].groupby("NOC").size().sort_values(ascending=False)

print("SANITY CHECK (de-duplicated golds):")
print(f"  Table Tennis since 1988: {dict(tt_by.head(4))}")
print(f"    -> CHN owns {tt_by.get('CHN', 0)}/{int(tt_by.sum())} "
      f"({100 * tt_by.get('CHN', 0) / tt_by.sum():.0f}%) of all table-tennis golds 🏓🇨🇳")
print(f"  Swimming all-time      : {dict(sw.head(4))}")
print(f"    -> USA leads with {sw.get('USA', 0)} (next is {sw.index[1]} {sw.iloc[1]}) 🏊🇺🇸")

assert tt_by.idxmax() == "CHN" and tt_by["CHN"] / tt_by.sum() > 0.8, "table-tennis check failed!"
assert sw.idxmax() == "USA" and sw["USA"] > 2 * sw.iloc[1], "swimming check failed!"
print("\nboth monopolies confirmed straight from the data ✅")
""")

# =====================================================================
# 7. Curated sports + per-sport top-N + Others
# =====================================================================
md(r"""
## 6 · Curated sports, and per‑sport **top owners + "Others"**

We pick a curated set of **iconic, medal‑rich** disciplines for the left column (a readable dozen rather
than every sport). Then, **per sport**, we keep the **top `PER_SPORT` nations** (by all‑time golds *in
that sport*) and fold every other nation into a single neutral **"Others"** ribbon. We name it exactly
**`"Others"`** because the library **pins that name to the bottom** and keeps it neutral even under
dynamic rank colouring — so a contested tail never steals the spotlight from the owner.
""")

code(r"""CURATED_SPORTS = [
    "Athletics", "Swimming", "Gymnastics", "Cycling", "Wrestling", "Boxing",
    "Weightlifting", "Rowing", "Fencing", "Table Tennis", "Shooting", "Canoeing",
]
PER_SPORT = 3   # top owners kept per sport; the rest -> neutral "Others"
OTHERS = "Others"

g = g_all[g_all.Sport.isin(CURATED_SPORTS)].copy()
tab = g.groupby(["Year", "Sport", "NOC"]).size().rename("n").reset_index()

# per sport: top-N nations by ALL-TIME golds IN THAT SPORT; the rest -> Others
keep_pairs = set()
print(f"Top {PER_SPORT} all-time owners per sport:")
for sp, sub in tab.groupby("Sport"):
    top = sub.groupby("NOC")["n"].sum().sort_values(ascending=False).head(PER_SPORT)
    keep_pairs.update((sp, noc) for noc in top.index)
    print(f"  {sp:<14} " + ", ".join(f"{k} {int(v)}" for k, v in top.items()))

tab["nation"] = [noc if (sp, noc) in keep_pairs else OTHERS
                 for sp, noc in zip(tab["Sport"], tab["NOC"])]
""")

# =====================================================================
# 8. Rolling window (ownership changes hands)
# =====================================================================
md(r"""
## 7 · A **rolling window** so ownership changes hands 🔄

A cumulative all‑time tally would *freeze* ownership (whoever led early stays ahead forever). Instead we
use a **rolling window** of the last **`ROLL`** Games (default 3 ≈ a **12‑year era**), so the ribbons
reflect **current dominance** — and you literally watch sports change hands. We first make the table
**dense** over every `(year, sport, nation)` (a Games a pair skipped counts as 0, which the window must
see), then take a rolling sum.
""")

code(r"""ROLL = 3        # rolling window in Games (~12-yr "current era")
years = sorted(tab["Year"].unique())

# collapse to [year, sport, nation, value] (folding the tail into Others)
gg = (tab.groupby(["Year", "Sport", "nation"])["n"].sum()
      .rename("value").reset_index()
      .rename(columns={"Year": "year", "Sport": "sport"}))

# make dense over every (year, sport, nation), then rolling-sum over the last ROLL Games
sport_set  = sorted(gg["sport"].unique())
nation_set = sorted(gg["nation"].unique())
idx = pd.MultiIndex.from_product([years, sport_set, nation_set],
                                 names=["year", "sport", "nation"])
gg = (gg.set_index(["year", "sport", "nation"])["value"]
      .reindex(idx, fill_value=0).reset_index().sort_values("year"))
gg["value"] = (gg.groupby(["sport", "nation"])["value"]
               .transform(lambda s: s.rolling(ROLL, min_periods=1).sum()))
df = gg[gg["value"] > 0].copy()       # drop zero-width (undrawable) ribbons

print(f"{len(df)} tidy [year, sport, nation, value] rows, {years[0]}-{years[-1]}")

# who owns each sport in the LAST window? (window leader, excluding Others)
last = df[df.year == years[-1]]
print(f"\nOwner of each sport in the {years[-1]} window (last {ROLL} Games):")
for sp in CURATED_SPORTS:
    s2 = last[(last.sport == sp) & (last.nation != OTHERS)]
    if len(s2):
        own = s2.groupby("nation")["value"].sum().sort_values(ascending=False)
        print(f"  {sp:<14} {own.index[0]} ({own.iloc[0]:.0f})")
""")

# =====================================================================
# 9. Layers + colours
# =====================================================================
md(r"""
## 8 · Layers and colours

Two layers: **sports on the left**, **nations on the right** (the racers). Sports keep calm, fixed
colours; the nation base colours are **placeholders** — `dynamic_color_mode="ranking"` repaints each by
its live rank every frame. **"Others"** stays a fixed neutral gray, pinned last. We order the nation
column by all‑time golds (Others last).
""")

code(r"""SPORT_COLORS = {
    "Athletics": "#E5484D", "Swimming": "#4EA8DE", "Gymnastics": "#F5A623",
    "Cycling": "#80ED99", "Wrestling": "#9B5DE5", "Boxing": "#FF7A59",
    "Weightlifting": "#00BBF9", "Rowing": "#56C2A8", "Fencing": "#C77DFF",
    "Table Tennis": "#F15BB5", "Shooting": "#A0A7B0", "Canoeing": "#3FBF7F",
}

# left layer: curated sports (in our order, where present)
sports_order = [s for s in CURATED_SPORTS if s in set(df["sport"])]

# right layer: nations by all-time golds, 'Others' pinned LAST
nat_tot = df.groupby("nation")["value"].sum().sort_values(ascending=False)
nations = [n for n in nat_tot.index if n != OTHERS]
if OTHERS in set(df["nation"]):
    nations = nations + [OTHERS]
LAYERS = [sports_order, nations]

# nation colours are placeholders (rank decides at render); sports + Others are fixed
node_colors = {n: "#888888" for n in nations}
node_colors[OTHERS] = "#6B7280"
for sp in sports_order:
    node_colors[sp] = SPORT_COLORS.get(sp, "#888888")

print("sports  (left):", sports_order)
print("nations (right):", nations)
""")

# =====================================================================
# 10. Overlay
# =====================================================================
md(r"""
## 9 · The growth overlay

The footer chart tracks the **golds in play** across these sports in the current window — the field of
contested gold rising as more events are added over the century. We also build the year tick labels.
""")

code(r"""total = [float(df[df.year == y]["value"].sum()) for y in years]   # golds in the active window
xlabels = [str(y) for y in years]
print(f"golds in play: {total[0]:,.0f} ({years[0]}) -> {total[-1]:,.0f} ({years[-1]})")
""")

# =====================================================================
# 11. Build the visual
# =====================================================================
md(r"""
## 10 · Building the visual — every knob explained

We hand the tidy `df` + `LAYERS` to `from_dataframe`, then drive `animate()` with the reel's design:

| Knob | Value | Why |
|---|---|---|
| `ranking_mode` | `True` | reorder nations by value each frame — *the race* |
| `stacked_mode` | `True` | bar heights scale with gold count (biggest owner = tallest) |
| `dynamic_color_mode` | `"ranking"` ⚠️ | **repaint each nation by its live rank each frame** — the star |
| `dynamic_colormap` | `"autumn_r"` ⚠️ | the rank→colour map: **#1 = hot → last = cool** |
| `theme` | `"dark"` | the neon‑on‑black reel look |
| `link_glow` / `link_alpha` | `1` / `0.6` | soft neon glow behind the ribbons |
| `n_segments` | `100` | smooth gradients along each ribbon |
| `yaxis_node` | `nations[0]` ⚠️ | a value axis scaled to the **leading nation's total golds** |
| `yaxis_suffix` / `value_prefix` | `""` / `""` ⚠️ | ticks read like `60` — **golds, not dollars** |
| `yaxis_gap` / `layer0_label_side` | `0.15` / `"left"` ⚠️ | axis hugs the column; sport names on the left |
| `overlay_series` / `overlay_x_labels` | total / years | the footer growth chart |
| `overlay_band` | `(0.18, 0.56)` ⚠️ | push the line chart **lower**, clear of the Sankey |

### How rank → colour works 🌈

With `dynamic_color_mode="ranking"`, every frame the library ranks the visible nations by value, maps
each rank into `[0, 1]` (**#1 → 0.0**, last → 1.0) and samples `autumn_r` there. `autumn` runs
yellow→red; the **`_r`** reverses it, so **#1 burns hot** and trailing nations cool off. As China seizes
table tennis and weightlifting, it **heats up on screen**. **"Others" stays gray** regardless.

First, build the renderer.
""")

code(r"""sk = Sankey.from_dataframe(
    df=df, layers=LAYERS,
    time_col="year", source_col="sport", target_col="nation", value_col="value",
    node_colors=node_colors,
)
anchor = nations[0]    # the all-time-leading nation anchors the gold-count axis
print(f"{len(sk.frames)} frames (Games):", sk.frames[0]['time_label'], "->", sk.frames[-1]['time_label'])
print("gold-count axis anchored on the nation layer of:", anchor)
""")

# =====================================================================
# 12. Render a short proof
# =====================================================================
md(r"""
## 11 · Render a short proof

Stills first — `save_frame()` needs **no FFmpeg**. We render three Games that bookend the ownership
story:

- **1912** — the early Games (USA + a few European powers own almost everything),
- **1980** — the **Cold‑War peak** (the **USSR** towers across gymnastics, wrestling, weightlifting; **East
  Germany** owns the water),
- **2016** — China firmly atop several sports (table tennis, weightlifting, shooting); Britain owns
  cycling; swimming still USA.

> ⚠️ **Caveat on the stills.** `save_frame` is **static**, so it can't do the per‑frame **dynamic
> ranking colours** (those are computed *during* the animation). The stills use the placeholder gray base
> for nations — they prove the **layout, axis, labels and ranking order**; the **hot→cool rank
> colouring** only appears in the animated clip below.
""")

code(r"""still_design = dict(
    figsize=(16, 9), dpi=95, font_size=12, title_fontsize=17, padding=2.8,
    margin_top=0.16, margin_bottom=0.22,
    ranking_mode=True, stacked_mode=True,
    theme="dark", link_glow=1, link_alpha=0.6, n_segments=100,
    yaxis_node=anchor, yaxis_suffix="",
    yaxis_gap=0.15, layer0_label_side="left",
    value_prefix="",                                 # golds, not dollars
)

title = f"Who owns each Olympic sport — {SEASON} Games, sport to nation by GOLD  (rolling {ROLL}-Games window)"
year_to_index = {f["time_label"]: i for i, f in enumerate(sk.frames)}
still_paths = []
for yr in ("1912", "1980", "2016"):
    if yr not in year_to_index:
        continue
    p = os.path.join(ASSETS, f"sport_dominance_{yr}.png")
    sk.save_frame(p, frame_index=year_to_index[yr], title=title, **still_design)
    still_paths.append(p)
    print("saved", p)

for p in still_paths:
    display(show(p, w=1000))
""")

md(r"""
### One short animation — *now* the rank colours come alive 🎬

A tiny clip to prove the motion **and** the dynamic colouring, with the **full** reel design: dark theme,
glow, the gold‑count axis, the growth overlay, and — the headline — **`dynamic_color_mode="ranking"` +
`autumn_r`**. We keep it short for speed; the cell **skips gracefully** if FFmpeg isn't installed.
""")

code(r"""sample_mp4 = os.path.join(ASSETS, "sport_dominance_sample.mp4")
try:
    sk.animate(
        sample_mp4,
        figsize=(16, 9), fps=24, duration_seconds=8, quality="medium", n_workers=2,
        title=title,
        font_size=12, title_fontsize=18, padding=2.8,
        margin_top=0.16, margin_bottom=0.22,
        ranking_mode=True, stacked_mode=True,
        dynamic_color_mode="ranking", dynamic_colormap="autumn_r",   # <-- the star of the demo
        theme="dark", link_glow=1, link_alpha=0.6, n_segments=100,
        yaxis_node=anchor, yaxis_suffix="", yaxis_gap=0.15, layer0_label_side="left",
        value_prefix="",                                             # golds, not dollars
        overlay_series=total, overlay_x_labels=xlabels,
        overlay_label=f"Golds in play across these {len(sports_order)} sports  (rolling {ROLL}-Games window)",
        overlay_color="#FFD24A", overlay_value_suffix="", overlay_badge=f"{ROLL}-GAMES",
        overlay_band=(0.18, 0.56),
    )
    from IPython.display import Video
    print("rendered:", sample_mp4)
    display(Video(sample_mp4, embed=True, width=900))
except Exception as e:
    print("Skipping the MP4 (FFmpeg likely not installed) — the stills above tell the story.")
    print("Reason:", repr(e))
""")

md(r"""
### Scale it up to the full reel ⏱️

The short clip above is deliberately tiny (8 s, low fps) so the notebook runs fast. The same `sk` object
renders the **full, postable reel** — just raise the duration and quality:

```python
# the full ~38s reel: sport -> nation by GOLD, rolling 3-Games window, ranking colours
sk.animate(
    os.path.join(ASSETS, "sport_dominance_full.mp4"),
    figsize=(16, 9), fps=30, duration_seconds=38, quality="high", n_workers=2,
    title=title,
    ranking_mode=True, stacked_mode=True,
    dynamic_color_mode="ranking", dynamic_colormap="autumn_r",
    theme="dark", link_glow=1, link_alpha=0.6, n_segments=100,
    yaxis_node=anchor, yaxis_suffix="", yaxis_gap=0.15, layer0_label_side="left",
    value_prefix="",
    overlay_series=total, overlay_x_labels=xlabels,
    overlay_label=f"Golds in play across these {len(sports_order)} sports  (rolling {ROLL}-Games window)",
    overlay_color="#FFD24A", overlay_value_suffix="", overlay_badge=f"{ROLL}-GAMES",
    overlay_band=(0.18, 0.56),
)
```

> A nice convention for batch renders: stamp every output filename with a **big‑endian, sortable**
> date‑time prefix (`%Y-%m-%d-%Hh%M`, `:` → `h` because it's illegal in paths) so your renders always
> sort chronologically. 🗂️
""")

# =====================================================================
# 13. Recap
# =====================================================================
md(r"""
## 12 · Recap & try it yourself

🎉 **You built the whole reel** — from a raw, one‑row‑per‑athlete CSV to an animated gradient Sankey that
shows **who owns each Olympic sport**, recoloured by rank every frame.

**The pipeline, in one breath:** download `athlete_events.csv` (cached) → **keep GOLD only** →
**de‑duplicate to one gold per `(Games, Event, NOC)`** → keep **dissolved nations distinct** (URS, GDR,
EUN) → curated **iconic sports** → per sport keep the **top owners** + neutral pinned **"Others"** →
**rolling 3‑Games window** so ownership changes hands → growth overlay → `from_dataframe` →
`save_frame` stills + `animate` with **`dynamic_color_mode="ranking"` + `autumn_r`**.

### 🧪 Try it yourself

1. **A longer era.** Raise `ROLL` to 4–5 in §7 — a smoother, more "dynasty" view; lower it to 1 for a
   jumpy "who won *this* Games" cut.
2. **More contested.** Raise `PER_SPORT` to 4–5 in §6 to show more challengers per sport (the "Others"
   tail shrinks).
3. **Different sports.** Swap the `CURATED_SPORTS` list — try team sports (Basketball, Football, Hockey)
   for a very different ownership map.
4. **Winter Games.** Set `SEASON = "Winter"` in §3 and curate winter disciplines (Alpine Skiing, Speed
   Skating, Biathlon, Cross Country Skiing) — Norway/USSR/Germany/USA stories on snow and ice.
5. **A different colormap.** Swap `dynamic_colormap="autumn_r"` for `"plasma"`, `"turbo"`, `"inferno"`.

### 🔗 Links

- The library: `gradient_sankey.py` (repo root) · `pip install gradient-sankey`
- The dataset — *120 years of Olympic history* (free, no API key):
  https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results
- The CSV mirror used here:
  https://raw.githubusercontent.com/Thomas-George-T/Olympic-History-Analytics-in-R/main/athlete_events.csv
- Companion tutorials: `notebooks/08_tutorial_olympics.ipynb` (same data, **stitched** nations, cumulative
  medal race), `notebooks/04_tutorial_co2.ipynb`

> 💡 The two lessons that make this reel **honest**: **(1)** de‑duplicate to one gold per
> `(Games, Event, NOC)` before counting, and **(2)** keep **dissolved nations distinct** (URS, GDR, EUN)
> — *"the USSR owned gymnastics"* is the whole point (the opposite of the medal‑race notebook, which
> stitches them). And **`dynamic_color_mode="ranking"`**, pinning **"Others"**, the gold axis
> (`value_prefix=""`), `yaxis_gap`, `layer0_label_side` and `overlay_band` are available in **this local
> copy** — importing the local module via `sys.path` (as in §1) makes everything just work. 💙

*Built com todo cuidado e carinho. Data: 120 years of Olympic history (free, no API key). Happy hacking!*
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
# Write to notebooks/ (the parent of this builders/ dir), keeping the output name.
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "10_tutorial_sport_dominance.ipynb")
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
