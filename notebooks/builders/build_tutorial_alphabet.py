# -*- coding: utf-8 -*-
"""Builds the Alphabet flagship tutorial (11_tutorial_alphabet.ipynb) with nbformat.

SELF-CONTAINED: the generated notebook depends ONLY on `gradient_sankey` (the
committed module at the repo root, added via sys.path), the Python stdlib, and
pip packages (pandas, requests, yfinance). It does NOT import or reference the
local-only daily_posts/ folder at all -- the data tables, the EDGAR companyfacts
fetch, the yfinance helper and the whole flow-builder are INLINED as teaching code
(this is an ETL lesson, so showing the code is the point).

It rebuilds the channel's flagship "company financials" format end to end:

  * FLIPPED layout: Total revenue is ONE block on the LEFT that fans OUT into the
    six revenue STREAMS, which then carry into the income-statement waterfall:
       [Total revenue] -> [Search, YouTube, Network, Subs/Plat/Dev, Cloud, Other
       Bets] -> [Gross profit, Cost of revenue] -> [Operating income, Operating
       expenses] -> [Net income, Tax & other].
  * Window FY2004 -> FY2025 (22 years; Google IPO 2004), with era folding for the
    years before each stream was broken out (YouTube 2017, Cloud 2018, Other Bets
    2015) and a hardcoded historical P&L for the pre-companyfacts years (2004-2012).
  * Each stream -> Gross/Cost split at the COMPANY-WIDE margin (per-stream cost is
    NOT disclosed -- stated as an honesty caveat).
  * STABLE curated stream hues via STREAM_COLORS + dynamic_color_mode="static";
    Total-revenue hub = calm slate #5B6B7E; waterfall = emerald profit family +
    red/orange cost family, kept off the ramp via fixed_color_nodes.
  * Footer overlay = GOOGL split-adjusted year-end SHARE PRICE via yfinance,
    falling back to total revenue when offline.
  * Segment-OI captions (2016+), era-marker annotations, and a hold on FY2023
    (Cloud's first profitable year).

    python build_tutorial_alphabet.py
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
# Alphabet: total revenue → its streams → the income statement
### The flagship **company‑financials** reel, built with **gradient‑sankey** — *com todo cuidado e carinho* 💙

Welcome! 👋 This is the channel's **flagship format**: a single multi‑layer Sankey that tells **a whole
company's money story in one conserved diagram**, animated across fiscal years **FY2004 → FY2025**
(22 years, from Google's 2004 IPO to the latest 10‑K).

The chart reads **left → right**:

1. **Total revenue** — one block — fans **out** into
2. the **six revenue streams** — Search, YouTube ads, Network, Subscriptions/Platforms/Devices, Cloud,
   Other Bets — which carry forward into
3. the **income‑statement waterfall**: Gross profit / Cost of revenue → Operating income / Operating
   expenses → **Net income** / Tax & other.

So the **left half** shows *where each dollar comes from* (the revenue **mix**) and the **right half**
shows *where it goes* (the company‑wide **P&L**). The streams keep **stable identity colours** every
frame (Cloud is always amber) while `ranking_mode` re‑sorts them, so you literally *watch* **Google
Cloud** and **YouTube ads** climb. The waterfall is pinned to **emerald (profit) / red‑orange (cost)**
so the two halves never blur into one colour ramp.

---

### Two honest data sources

| Half | Source | Why |
|---|---|---|
| **Right** — the P&L waterfall | SEC **EDGAR `companyfacts`** (company‑wide totals) | the API exposes only the **default member** of each concept — the company‑wide total |
| **Left** — the six streams | Alphabet **10‑K "Disaggregation of revenue"** tables | `companyfacts` *strips* dimensional members, so the per‑stream split must come from the filing text |

We **reconcile** the streams back to the EDGAR total **to the dollar** (the only bridging item is a tiny
*hedging* line), and we never imply a *per‑stream* cost (Alphabet doesn't disclose one — the cost/profit
split is **company‑wide**). The footer overlays the **GOOGL split‑adjusted year‑end share price** — a
second story: *how the market valued all of this.*

> **This notebook is fully self‑contained.** It clones‑and‑runs from the gradient‑sankey repo: it imports
> only `gradient_sankey` (the committed module at the repo root), the Python standard library, and the pip
> packages `pandas`, `requests` and `yfinance`. Every data table and every helper is **inlined below as a
> teaching step**, so you can read the whole ETL pipeline end to end. 💙
""")

# =====================================================================
# 2. Setup & install
# =====================================================================
md(r"""
## 1 · Setup & install

The library lives in this repo as a single module, `gradient_sankey.py`, at the **repo root**. We add the
repo root to `sys.path` so we always get the **local** copy with the newest features:

```bash
pip install gradient-sankey pandas requests yfinance   # the public packages
# this notebook uses the LOCAL repo copy of gradient_sankey via sys.path (see below)
```

> ⚠️ **Heads‑up — newest features.** This reel leans on capabilities in `gradient_sankey.py` that may be
> newer than the pinned PyPI release: the **layer‑total \$ axis** (`yaxis_node`), **static curated colours**
> with a live **ranking race** (`dynamic_color_mode="static"` + `ranking_mode`), **`fixed_color_nodes`**
> (keep some nodes off the ramp), **`annotations`** (a per‑year caption), **`hold_periods`** (linger on a
> year), and the **overlay** controls. Importing the local module via `sys.path` makes all of these work
> today. 🎁

For an **MP4** you also need **FFmpeg** on PATH — but still frames (`save_frame`) need nothing extra, so
the notebook stays runnable even without FFmpeg or yfinance.
""")

code(r"""import os, sys, json, pathlib
from datetime import date

# --- import the LOCAL gradient_sankey from the repo root (committed module) ---
# This notebook lives in notebooks/, so the repo root is one level up.
REPO_ROOT = os.path.abspath("..")                      # notebooks/  ->  repo root
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd
from IPython.display import Image, display

import gradient_sankey as gs
from gradient_sankey import SankeyRaceMultiLayerParallel as Sankey

# where we'll drop the rendered stills / sample clip (kept out of the repo root)
ASSETS = os.path.abspath("gallery_assets"); os.makedirs(ASSETS, exist_ok=True)

# a LOCAL, gitignored cache for the one EDGAR download + the yfinance pull, so a
# re-run is instant. Re-fetching from EDGAR/yfinance on a clean run is fine.
CACHE = pathlib.Path(os.path.abspath(".nbcache")); CACHE.mkdir(parents=True, exist_ok=True)

def show(path, w=1000):
    return Image(filename=path, width=w)

print("gradient-sankey version:", gs.__version__)
print("pandas:", pd.__version__)
print("cache ->", CACHE)
""")

# =====================================================================
# 3. The constants: streams, colours, layers
# =====================================================================
md(r"""
## 2 · The vocabulary — streams, the waterfall, and a cohesive palette

Before any data, let's name the **nodes** and give them **meaning‑carrying colours**. The six revenue
**streams** each keep a **stable identity hue** every frame (a curated rainbow, *not* a turbo rank‑recolour),
so you can track "Cloud = amber" across 20 years. The income‑statement **waterfall** is pinned to an
**emerald (profit) / red‑orange (cost)** family, kept **off** the rank ramp via `FIXED_COLOR_NODES`, so the
two halves never blur into one gradient.
""")

code(r'''# --- Layer 0: the SIX revenue streams (fixed visual declaration order) ---------
SEARCH   = "Google Search & other"
YOUTUBE  = "YouTube ads"
NETWORK  = "Google Network"
SUBSPD   = "Subscriptions, platforms & devices"
CLOUD    = "Google Cloud"
OTHERB   = "Other Bets"
STREAMS  = [SEARCH, YOUTUBE, NETWORK, SUBSPD, CLOUD, OTHERB]

# A CURATED rainbow (cyan -> blue -> violet -> pink -> amber, + slate): each stream
# keeps a STABLE identity hue every frame, and the warm end is held to ONE amber so
# the streams stay visually distinct from the red/orange COST side of the waterfall.
STREAM_COLORS = {
    SEARCH:  "#22D3EE",  # cyan       - the search advertising engine (usually #1)
    NETWORK: "#3B82F6",  # blue       - the ad network (AdSense/AdMob)
    SUBSPD:  "#A78BFA",  # violet     - subscriptions, platforms & devices
    YOUTUBE: "#F472B6",  # pink       - YouTube ads (distinct from cost-red)
    CLOUD:   "#FBBF24",  # amber      - Google Cloud (the climber)
    OTHERB:  "#94A3B8",  # slate gray - Other Bets (+ hedging remainder)
}

# --- Layer 1+: the income-statement WATERFALL nodes (fixed meaning colours) -----
HUB          = "Total revenue"
COST_OF_REV  = "Cost of revenue"
GROSS_PROFIT = "Gross profit"
OPEX         = "Operating expenses"
OP_INCOME    = "Operating income"
TAX_OTHER    = "Tax & other"
NET_INCOME   = "Net income"

PNL_COLORS = {
    HUB:          "#5B6B7E",   # calm slate    - the total-revenue trunk (NOT glaring white)
    GROSS_PROFIT: "#2DD4A7",   # emerald       - profit kept after cost of revenue
    OP_INCOME:    "#26C485",   # emerald       - profit kept after opex
    NET_INCOME:   "#6EE7B7",   # light emerald - the bottom line
    COST_OF_REV:  "#F87171",   # red           - cost of revenue
    OPEX:         "#FB923C",   # orange        - R&D, S&M, G&A
    TAX_OTHER:    "#EF4444",   # deep red      - tax, interest & other
}

# Keep every waterfall node at its base colour (OFF the rank ramp): only the six
# revenue streams race-recolour, so the two halves never share one colour ramp.
FIXED_COLOR_NODES = {HUB, COST_OF_REV, GROSS_PROFIT, OPEX, OP_INCOME, TAX_OTHER, NET_INCOME}

# Layers, left -> right. Total revenue leads as ONE block, fans OUT into the six
# streams (the revenue mix), which then carry into the income-statement waterfall.
#   [Total revenue] -> [streams] -> [Gross/Cost] -> [OpInc/OpEx] -> [Net/Tax]
PNL_LAYERS_TAIL = [
    [GROSS_PROFIT, COST_OF_REV],
    [OP_INCOME, OPEX],
    [NET_INCOME, TAX_OTHER],
]

ALPHABET_CIK = "0001652044"   # Alphabet Inc., for the SEC EDGAR endpoints
print("six streams:", STREAMS)
print("waterfall nodes (off the rank ramp):", sorted(FIXED_COLOR_NODES))
''')

# =====================================================================
# 4. The hardcoded data tables (from the 10-Ks)
# =====================================================================
md(r"""
### The sourced data tables 📑

Two of our inputs are **transcribed from Alphabet's 10‑K filings** (the SEC `companyfacts` API can't give
them to us — see §2 below). We keep them here, **cited to the source filing**, as plain Python dicts:

- **`REVENUE_BY_STREAM`** — revenue by the six streams, US\$ millions, FY2004–FY2025, straight from the
  10‑K *"Disaggregation of revenue"* tables.
- **`HEDGING`** — the tiny "Hedging gains (losses)" line, the **only** reconciling item between the six
  streams and the EDGAR total.
- **`SEGMENT_OI`** — the three reportable segments' operating income (with **losses**), for the per‑year
  caption.
- **`HISTORICAL_PNL`** — the company‑wide P&L for FY2004–FY2012, the years *before* Alphabet's EDGAR
  `companyfacts` begins (it starts at FY2013).
- **`ERA_NOTES`** — captions marking the **disclosure** changes (so a stream's *appearance* reads as a
  reclassification, not a real business jump).
""")

code(r'''# Alphabet revenue BY STREAM, US$ MILLIONS, full fiscal years (= calendar years).
# Columns: search | youtube | network | subs_platforms_devices | cloud | other_bets
# Source: SEC EDGAR 10-K "Disaggregation of revenue" / "Revenue by Type" tables.
#   (FY2018-2021 the 4th column is labelled "Google other"; FY2022+ Alphabet
#    renamed the identical line "Google subscriptions, platforms, and devices".)
# Each row + that year's HEDGING gain/loss = the EDGAR total Revenues, to the dollar.
REVENUE_BY_STREAM = {
    #      search,  youtube, network,  subs/plat/dev,   cloud,  other_bets
    # --- The "honest era" 2004-2016 (Google Inc, CIK 0001288776): the 10-Ks only
    #     split advertising into "Google websites" (owned & operated -- INCLUDES
    #     YouTube ads, never broken out pre-2017) and "Google Network", plus a
    #     "Licensing and other"/"Other revenues" catch-all (hardware, Play,
    #     licensing, and Cloud-before-2018). So here:
    #       search = Google websites ads (YouTube folded in)  -> youtube column = 0
    #       network = Google Network ads
    #       subs/plat/dev = Licensing & other / Google other  -> cloud column = 0
    #       other_bets = 0 before FY2015 (the Alphabet reorg first split it out)
    #     Each row sums EXACTLY to that year's EDGAR total revenue (hedging 0 here).
    2004: (1589,        0,    1554,        46,           0,       0),
    2005: (3377,        0,    2688,        74,           0,       0),
    2006: (6333,        0,    4160,       112,           0,       0),
    2007: (10625,       0,    5788,       181,           0,       0),
    2008: (14414,       0,    6715,       667,           0,       0),
    2009: (15723,       0,    7166,       762,           0,       0),
    2010: (19444,       0,    8792,      1085,           0,       0),
    2011: (26145,       0,   10386,      1374,           0,       0),
    2012: (31221,       0,   12465,      2353,           0,       0),
    2013: (37422,       0,   13125,      4972,           0,       0),
    2014: (45085,       0,   13971,      6945,           0,       0),
    2015: (52357,       0,   15033,      7151,           0,     448),
    2016: (63785,       0,   15598,     10080,           0,     809),
    # FY2017 uses Alphabet's OWN recast (FY2019 10-K), which DID break out YouTube
    # ads ($8,150M) and Google Cloud ($4,056M) back to 2017 -- the first year those
    # two columns are real. Hedging -169 reconciles (see HEDGING).
    2017: (69811,    8150,   17616,     10914,        4056,     477),
    2018: (85296,   11155,   20010,      14063,        5838,     595),
    2019: (98115,   15149,   21547,      17014,        8918,     659),
    2020: (104062,  19772,   23090,      21711,       13059,     657),
    2021: (148951,  28845,   31701,      28032,       19206,     753),
    2022: (162450,  29243,   32780,      29055,       26280,    1068),
    2023: (175033,  31510,   31312,      34688,       33088,    1527),
    2024: (198084,  36147,   30359,      40340,       43229,    1648),
    2025: (224532,  40367,   29792,      48030,       58705,    1537),
}

# Hedging gains (losses), US$ MILLIONS -- the ONLY reconciling item between the six
# streams and the EDGAR total Revenues. Source: the same 10-K revenue tables. 0 for
# 2004-2016 (any hedging effect was embedded in the advertising lines; the streams
# already sum exactly to the total). FY2017's -169 is from the FY2019 10-K recast.
HEDGING = {2004: 0, 2005: 0, 2006: 0, 2007: 0, 2008: 0, 2009: 0, 2010: 0,
           2011: 0, 2012: 0, 2013: 0, 2014: 0, 2015: 0, 2016: 0, 2017: -169,
           2018: -138, 2019: 455, 2020: 176, 2021: 149,
           2022: 1960, 2023: 236, 2024: 211, 2025: -127}

# SEGMENT operating income (loss), US$ MILLIONS, from the 10-K segment footnote.
# These include LOSSES (Other Bets negative every year; Cloud negative until it
# first turned profitable in FY2023). Surfaced as annotations, NOT as ribbons.
# Columns: services | cloud | other_bets
SEGMENT_OI = {
    2018: (43137,  -4348,  -3358),
    2019: (48999,  -4645,  -4824),
    2020: (54606,  -5607,  -4476),
    2021: (91855,  -3099,  -5281),
    2022: (82699,  -1922,  -4636),
    2023: (95858,   1716,  -4095),   # <- Cloud's first profitable year
    2024: (121263,  6112,  -4444),
    2025: (139404, 13910,  -7515),
}
CLOUD_FLIP_YEAR = 2023   # the year Google Cloud's segment OI flips - -> +

# Era-marker captions so the stream *appearances* read as DISCLOSURE changes
# (reclassifications), not real jumps in the business. For 2018+ the per-year
# segment-OI line takes priority (it carries the live profit story).
ERA_NOTES = {
    "2004": "Google IPO — advertising is ~99% of revenue",
    "2015": "Alphabet reorg — 'Other Bets' split out",
    "2017": "YouTube ad revenue broken out for the first time",
    "2018": "Google Cloud broken out as its own line",
}

# Historical company-wide P&L, US$ MILLIONS, FY2004-2012 -- the years BEFORE EDGAR
# companyfacts (CIK 0001652044) begins (it starts at FY2013). For FY2013+ the live
# companyfacts totals are used; for FY2004-2012 we fall back to this hardcoded
# table, cited to the Google Inc 10-Ks. Each year's revenue MUST equal the sum of
# that year's six streams (+ hedging) so the left and right halves reconcile.
# Columns: revenue | cost_of_revenue | op_income | net_income
#   FY2012 op_income/net_income are the Google-only CONTINUING-OPERATIONS figures
#   as restated by the FY2014 10-K (Motorola Mobility pushed to discontinued ops).
HISTORICAL_PNL = {
    #      revenue,  cost,   op_income, net_income
    2004: (3189,    1469,     640,       399),
    2005: (6139,    2577,    2017,      1465),
    2006: (10605,   4225,    3550,      3077),
    2007: (16594,   6649,    5084,      4204),
    2008: (21796,   8622,    6632,      4227),
    2009: (23651,   8844,    8312,      6520),
    2010: (29321,  10417,   10381,      8505),
    2011: (37905,  13188,   11742,      9737),
    2012: (46039,  17176,   13834,     11553),
}

print(f"REVENUE_BY_STREAM: {len(REVENUE_BY_STREAM)} years, FY{min(REVENUE_BY_STREAM)}-FY{max(REVENUE_BY_STREAM)}")
print(f"SEGMENT_OI: FY{min(SEGMENT_OI)}-FY{max(SEGMENT_OI)}  |  Cloud flips +ve in FY{CLOUD_FLIP_YEAR}")
print(f"HISTORICAL_PNL backfill: FY{min(HISTORICAL_PNL)}-FY{max(HISTORICAL_PNL)} (pre-companyfacts)")
''')

# =====================================================================
# 5. EXTRACT (a) companyfacts P&L
# =====================================================================
md(r"""
## 3 · EXTRACT

### 3a · SEC EDGAR `companyfacts` — the company‑wide P&L (and the disaggregation trap 🪤)

Every US public company files XBRL‑tagged financials with the SEC. The **`companyfacts`** endpoint returns
*all* concepts for one company in a single JSON. It needs a **descriptive `User‑Agent`** (the SEC blocks
anonymous clients) and politely retries on its HTTP 429 throttle. Alphabet's CIK is `0001652044`:

```
https://data.sec.gov/api/xbrl/companyfacts/CIK0001652044.json
```

We read the **company‑wide totals** straight from it — `Revenues`, `CostOfRevenue`, `OperatingIncomeLoss`,
`NetIncomeLoss` — and those become the **right‑hand waterfall**.

We build a tiny `requests` session with the required header + a polite retry, and **disk‑cache** the one
~3 MB download to our local `.nbcache/` so a re‑run is instant.

> ### ⚠️ The trap (same as the Amazon lesson)
> `companyfacts` returns only the **default member** of each concept — the company‑wide **TOTAL** — and
> strips every dimensional / axis‑member fact. So it gives **one** revenue number per year, with **no
> per‑stream breakdown**. We'll get the streams from the 10‑K instead (§3b) and **reconcile** them back.
""")

code(r'''import requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except ImportError:                                  # pragma: no cover
    from requests.packages.urllib3.util.retry import Retry

# A small retrying session carrying the REQUIRED SEC User-Agent (set a real
# SEC_USER_AGENT env var to be a good API citizen). Retries on the 429 throttle.
EDGAR_HEADERS = {"User-Agent": os.environ.get(
    "SEC_USER_AGENT", "gradient-sankey tutorial (contact: you@example.com)")}

def edgar_session():
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["GET"]))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(EDGAR_HEADERS)
    return s

SESSION = edgar_session()
print("User-Agent:", EDGAR_HEADERS["User-Agent"])
print("(tip: set a real SEC_USER_AGENT env var to be a good API citizen)")
print("CIK:", ALPHABET_CIK)
''')

md(r"""
**Download companyfacts (cached) and pull the P&L totals.** We keep the latest‑filed **full‑year**
(~365‑day) value of each concept. `Revenues` is occasionally missing a year, so we merge in the ASC‑606
contract tag as a back‑up (later tags win on overlap).
""")

code(r'''cf = CACHE / f"edgar_companyfacts_CIK{ALPHABET_CIK}.json"
if not cf.exists():
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{ALPHABET_CIK}.json"
    print("downloading Alphabet companyfacts (~3 MB) ...")
    r = SESSION.get(url, timeout=120); r.raise_for_status()
    cf.write_bytes(r.content)
facts = json.loads(cf.read_text(encoding="utf-8"))
gaap = facts["facts"]["us-gaap"]
print("entity:", facts["entityName"])
print("us-gaap concepts available:", len(gaap))

def days(s, e):
    return (date.fromisoformat(e) - date.fromisoformat(s)).days

def annual(*tags):
    "{fiscal_year -> latest-filed full-year (~365d) value}, merging tags (later wins)."
    best = {}                                            # end -> (filed, val)
    for tag in tags:
        if tag not in gaap:
            continue
        for u in gaap[tag]["units"].get("USD", []):
            if u.get("start") and 350 <= days(u["start"], u["end"]) <= 380:
                e, filed = u["end"], u.get("filed", "")
                if e not in best or filed > best[e][0]:
                    best[e] = (filed, u["val"])
    return {int(e[:4]): v for e, (f, v) in best.items()}

PNL = {}
rev  = annual("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax")
cost = annual("CostOfRevenue")
oi   = annual("OperatingIncomeLoss")
ni   = annual("NetIncomeLoss")
for y in sorted(set(rev) & set(cost) & set(oi) & set(ni)):
    PNL[y] = {"revenue": rev[y], "cost_of_revenue": cost[y],
              "op_income": oi[y], "net_income": ni[y]}

print(f"\ncompanyfacts covers FY{min(PNL)}-FY{max(PNL)} (it begins at FY2013).")
print(f"  {'FY':>4} {'Revenue':>9} {'CostOfRev':>10} {'OpIncome':>9} {'NetIncome':>10}  (US$ B)")
for y in sorted(PNL):
    p = PNL[y]
    print(f"  {y:>4} {p['revenue']/1e9:9.1f} {p['cost_of_revenue']/1e9:10.1f} "
          f"{p['op_income']/1e9:9.1f} {p['net_income']/1e9:10.1f}")
''')

md(r"""
#### The pre‑companyfacts years (FY2004–FY2012)

companyfacts only begins at **FY2013**, but our window starts at the **2004 IPO**. So we backfill the small
**hardcoded historical P&L** (`HISTORICAL_PNL`, defined above) for FY2004–FY2012, cited to the Google Inc
10‑Ks, and let companyfacts **win on any overlap**.
""")

code(r'''# Backfill the pre-companyfacts years (companyfacts FY2013+ WINS on overlap), in US$.
for y, (rev_m, cost_m, oi_m, ni_m) in HISTORICAL_PNL.items():
    PNL.setdefault(y, {"revenue": rev_m*1e6, "cost_of_revenue": cost_m*1e6,
                       "op_income": oi_m*1e6, "net_income": ni_m*1e6})
print("P&L now spans FY%d-FY%d (hardcoded 2004-2012 + companyfacts 2013+)." % (min(PNL), max(PNL)))
display(pd.DataFrame.from_dict(HISTORICAL_PNL, orient="index",
        columns=["revenue", "cost", "op_income", "net_income"]).rename_axis("year"))
''')

# =====================================================================
# 6. EXTRACT (b) the 10-K streams table
# =====================================================================
md(r"""
### 3b · The honest decision: the streams come from the 10‑K 📑

`companyfacts` can't disaggregate revenue, so — like a careful analyst — we read the **10‑K**. Alphabet
publishes a **"Disaggregation of revenue" / "Revenue by Type"** table breaking revenue into six streams.
We transcribed those *reported* figures into the **sourced** `REVENUE_BY_STREAM` table above (US\$ millions).

> ### 📅 The disclosure eras (why some streams are \$0 early)
> Alphabet didn't always report these six streams. Before each break‑out, a stream is **folded** into
> another line and shows as **\$0** here, so its later *appearance* is a **disclosure change**, not a
> real jump in the business:
>
> | Year | Disclosure change | Effect in the table |
> |---|---|---|
> | **2004** | Google IPO — advertising is ~99% of revenue | Search = "Google websites" ads (**YouTube folded in**) |
> | **2015** | Alphabet reorg | **Other Bets** first split out ($448M) |
> | **2017** | First YouTube‑ads break‑out (FY2019 10-K recast) | **YouTube** column becomes real ($8,150M); **Cloud** too ($4,056M) |
> | **2018** | Google Cloud broken out as its own line | the six‑stream series is fully consistent |
>
> So **2004–2016**: `youtube = 0`, `cloud = 0` (folded into Search / "Other"); **Other Bets = 0** before
> 2015. *(FY2018–2021 the 4th column was labelled "Google other"; from FY2022 Alphabet renamed the
> identical line "Google subscriptions, platforms, and devices".)*

Each year's six streams **+ a tiny hedging line** sum **exactly** to the EDGAR total. We fold hedging into
**Other Bets** so the six streams reconcile to the dollar.
""")

code(r'''wide = (pd.DataFrame.from_dict(REVENUE_BY_STREAM, orient="index", columns=STREAMS)
        .rename_axis("year"))
print("Alphabet revenue by stream (US$ MILLIONS, as reported in the 10-Ks):")
print("note the $0 columns before each stream's break-out era (YouTube<2017, Cloud<2018, OtherBets<2015)")
display(wide)

print("Hedging gains (losses), US$ MILLIONS — the only reconciling line (0 before FY2017):")
display(pd.Series(HEDGING, name="hedging_$M").rename_axis("year").to_frame())
''')

# =====================================================================
# 7. EXTRACT (c) yfinance share price
# =====================================================================
md(r"""
### 3c · yfinance — the GOOGL split‑adjusted year‑end share price 💵

The footer overlay is **not** revenue (revenue is already the first layer) — it's the **share price**, a
second story about how the market valued all this. We pull GOOGL with `yfinance`, take the **year‑end
(Dec 31) close** for each fiscal year (Alphabet's fiscal year = the calendar year), and use
`auto_adjust=True` so **every split is back‑applied** (the 2014 class split and the 2022 20:1) for one
continuous, comparable curve from the 2004 IPO to today. It's **disk‑cached** in our local `.nbcache/`, and
**falls back to `None` when offline** (we then overlay total revenue instead).
""")

code(r'''def stock_by_year(ticker, years):
    """Split-adjusted year-END close of `ticker` for each fiscal year (cached on disk).

    auto_adjust=True back-applies every split (incl. the 2014 class split and the
    20:1 of 2022), giving one continuous, comparable curve from the 2004 IPO to
    today. Returns a list aligned to `years` (None entries if offline / no data)."""
    cf = CACHE / f"stock_{ticker}_{years[0]}_{years[-1]}.json"
    if cf.exists():
        d = json.loads(cf.read_text(encoding="utf-8"))
        return [d.get(str(y)) for y in years]
    try:
        import yfinance as yf
        s = yf.Ticker(ticker).history(period="max", auto_adjust=True)["Close"]
        if s.empty:
            return None
        s.index = s.index.tz_localize(None)
        first = s.index.min()
        out, d = [], {}
        for y in years:
            ts = pd.Timestamp(f"{y}-12-31")
            v = float(s.asof(ts)) if ts >= first else float("nan")
            if v != v:                                    # NaN -> no data that year
                v = None
            out.append(v)
            d[str(y)] = v
        cf.write_text(json.dumps(d), encoding="utf-8")
        return out
    except Exception as e:
        print(f"  (stock overlay unavailable: {e}; falling back to total revenue)")
        return None

years_all = sorted(REVENUE_BY_STREAM)                    # FY2004 -> FY2025
stock = stock_by_year("GOOGL", years_all)                # list aligned to years_all
stock_df = pd.DataFrame({"year": years_all, "GOOGL_year_end_split_adj_$": stock})
print("GOOGL split-adjusted year-end close (US$), via stock_by_year (cached):")
display(stock_df.set_index("year").T)
if stock and any(v is not None for v in stock):
    lo = min(v for v in stock if v is not None); hi = max(v for v in stock if v is not None)
    print(f"range: ${lo:,.2f}  ->  ${hi:,.2f}  (split-adjusted; offline -> overlay falls back to revenue)")
else:
    print("yfinance offline -> we'll overlay TOTAL REVENUE instead.")
''')

# =====================================================================
# 8. TRANSFORM — reconcile + derive waterfall + margin split
# =====================================================================
md(r"""
## 4 · TRANSFORM — reconcile, derive & label

### 4a · Reconcile the streams to the EDGAR total 🤝

A hardcoded table is only as good as its **check**. We fold **hedging into Other Bets**, sum the six
streams per year, and compare to the **independent** companyfacts `Revenues` total. They must match
within **\$1B** (most to the dollar) — and we **assert** it.
""")

code(r'''wide_rec = wide.copy()
wide_rec[OTHERB] = wide_rec[OTHERB] + pd.Series(HEDGING)     # fold hedging into Other Bets

recon = wide_rec.sum(axis=1).to_frame("streams+hedging_$M")
recon["edgar_total_$M"] = [PNL[y]["revenue"]/1e6 for y in recon.index]
recon["diff_$M"] = (recon["streams+hedging_$M"] - recon["edgar_total_$M"]).round(1)
recon["match (<$1B)"] = recon["diff_$M"].abs() < 1000.0
display(recon)
assert recon["match (<$1B)"].all(), "A stream row does not reconcile to the EDGAR total!"
print("All 22 years reconcile to the EDGAR total within $1B (most to the dollar). ✓")
print(f"(e.g. FY2024 streams+hedging = ${recon.loc[2024,'streams+hedging_$M']/1000:,.1f}B = the reported total.)")
''')

md(r"""
### 4b · Derive the balanced waterfall (and the company‑wide margin split)

Now the heart of the flagship format. We turn the data into a tidy **long** frame: one row per flow,
`[year, source, target, value]`, in the **flipped** layer order.

**Left (where the dollar comes from).** Total revenue → each **stream** (the mix).

**Middle/right (where it goes).** A balanced **waterfall**, with the three residual stages *derived* so
every node's inflow equals its outflow:

$$\text{gross profit} = \text{revenue} - \text{cost of revenue}$$
$$\text{opex} = \text{gross profit} - \text{operating income}$$
$$\text{tax \& other} = \text{operating income} - \text{net income}$$

**Each stream → Gross profit / Cost of revenue** at the **company‑wide margin** ($gm = \text{gross}/\text{rev}$,
$cm = \text{cost}/\text{rev}$). Alphabet does **not** disclose cost *per stream*, so we carry every stream
forward at the *same* company‑wide ratio — the title says so. This keeps the column totals exact and
conserved without ever implying a per‑stream cost.

> ### ⚖️ The tricky years
> In **FY2018, FY2019 and FY2025**, big **gains on equity investments** pushed *other income* above the
> tax bill, so **Net income > Operating income** and "Tax & other" goes **negative**. A Sankey ribbon
> can't be negative, so we **clamp** each kept spine to `[0, parent]` and route the remainder to the
> leak — while the **label** still shows the real signed figure in accounting parentheses, e.g.
> `($3.1B)`. The picture stays conserved; the numbers stay honest. (Same trick as the NVIDIA reel.)

This whole step is the `alphabet_flows` builder, inlined so you can read every line.
""")

code(r'''def _clamp(x, lo, hi):
    return max(lo, min(x, hi))

def _fmt(v):
    """Signed $B label: 59.25 -> '$59.2B' ; -3.13 -> '($3.1B)' (accounting style)."""
    mag = abs(v)
    s = f"{mag:.1f}" if mag < 100 else f"{mag:.0f}"
    return f"(${s}B)" if v < 0 else f"${s}B"

def _seg_b(v_m):
    """Compact signed $B for the caption: 1716 -> '+2' ; -4636 -> '-5'."""
    b = v_m / 1e3
    return f"{'+' if b >= 0 else '-'}{abs(b):.0f}"

def alphabet_flows(y0=2004, y1=None, divisor=1e3):
    """Build the balanced Total-revenue -> streams -> ... -> Net income flow table.

    Returns (df[year, source, target, value, signed], stream_order, total_by_year,
    labels_by_year, annotations). Values are US$ BILLIONS. The stream revenues come
    from REVENUE_BY_STREAM (US$ millions, /1e3, hedging folded into Other Bets); the
    company-wide P&L totals come from EDGAR companyfacts (US$, /1e9, in PNL) and are
    reconciled to the stream sum each year (RuntimeError if they ever drift >$1B)."""
    years = sorted(y for y in REVENUE_BY_STREAM
                   if (y0 is None or y >= y0) and (y1 is None or y <= y1))
    rows, total_by_year, labels_by_year, annotations = [], {}, {}, {}
    for y in years:
        stream_m = dict(zip(STREAMS, REVENUE_BY_STREAM[y]))    # US$ millions
        stream_m[OTHERB] += HEDGING[y]                         # fold hedging -> reconciles
        revenue = sum(stream_m.values()) / divisor            # US$ B (= reconciled total)
        total_by_year[y] = revenue

        if y not in PNL:                                       # offline fallback: skip P&L
            continue
        p = PNL[y]
        rep = p["revenue"] / 1e9
        if abs(rep - revenue) > 1.0:                          # > $1B drift = real problem
            raise RuntimeError(
                f"{y}: streams (+ hedging) sum to ${revenue:,.1f}B but EDGAR "
                f"reports ${rep:,.1f}B total revenue.")
        cost_of_rev = p["cost_of_revenue"] / 1e9
        op_income   = p["op_income"] / 1e9
        net_income  = p["net_income"] / 1e9
        gross_profit = revenue - cost_of_rev
        opex         = gross_profit - op_income
        tax_other    = op_income - net_income

        # balanced waterfall: clamp each kept spine to [0, parent]
        gross_kept = _clamp(gross_profit, 0.0, revenue);  cost = revenue - gross_kept
        op_kept    = _clamp(op_income, 0.0, gross_kept);  opx  = gross_kept - op_kept
        net_kept   = _clamp(net_income, 0.0, op_kept);    tax  = op_kept - net_kept

        # company-wide margins (per-stream cost is NOT disclosed -> same ratio for all)
        gm = gross_kept/revenue if revenue else 0.0
        cm = cost/revenue if revenue else 0.0
        flows = [
            # col 0 -> 1: Total-revenue hub fans OUT into the six streams (the mix)
            *[(HUB, s, stream_m[s]/divisor, stream_m[s]/divisor) for s in STREAMS],
            # col 1 -> 2: each stream carries forward into Gross profit + Cost of revenue
            *[(s, GROSS_PROFIT, (stream_m[s]/divisor)*gm, (stream_m[s]/divisor)*gm) for s in STREAMS],
            *[(s, COST_OF_REV,  (stream_m[s]/divisor)*cm, (stream_m[s]/divisor)*cm) for s in STREAMS],
            # col 2 -> 4: the company-wide P&L waterfall (balanced value, real signed figure)
            (GROSS_PROFIT, OP_INCOME, op_kept, op_income),
            (GROSS_PROFIT, OPEX, opx, opex),
            (OP_INCOME, NET_INCOME, net_kept, net_income),
            (OP_INCOME, TAX_OTHER, tax, tax_other),
        ]
        for src, tgt, value, signed in flows:
            rows.append({"year": y, "source": src, "target": tgt, "value": value, "signed": signed})

        # per-frame labels: real signed $ figures (accounting parentheses on losses)
        node_signed = {s: stream_m[s]/divisor for s in STREAMS}
        node_signed.update({HUB: revenue, GROSS_PROFIT: gross_profit, COST_OF_REV: cost_of_rev,
                            OP_INCOME: op_income, OPEX: opex, NET_INCOME: net_income,
                            TAX_OTHER: tax_other})
        labels_by_year[y] = {n: _fmt(v) for n, v in node_signed.items()}

        # era-marker caption first, then the segment-OI line OVERRIDES it (2018+)
        if str(y) in ERA_NOTES:
            annotations[str(y)] = ERA_NOTES[str(y)]
        if y in SEGMENT_OI:
            sv, cl, ob = SEGMENT_OI[y]
            annotations[str(y)] = (
                f"Segment op. income ($B)\n"
                f"Services {_seg_b(sv)}   Cloud {_seg_b(cl)}   Other Bets {_seg_b(ob)}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No P&L data available (offline?) -- companyfacts is required.")
    # left column ordered by all-time total; ranking_mode re-sorts per frame. Other
    # Bets pinned last as the small remainder.
    stream_order = (df[df.source.isin(STREAMS)].groupby("source")["value"].sum()
                    .drop(index=OTHERB, errors="ignore")
                    .sort_values(ascending=False).index.tolist()) + [OTHERB]
    return df, stream_order, total_by_year, labels_by_year, annotations

# Build it: the tidy flow df + the stream order + totals + signed labels + captions.
df, stream_order, total_by_year, labels_by_year, annotations = alphabet_flows(2004, None)
years = sorted(df["year"].unique())
long = df[["year", "source", "target", "value"]].copy()    # the renderer wants value-only

print(f"{len(long)} tidy [year, source, target, value] rows, FY{years[0]}-FY{years[-1]}")
print("stream order (biggest all-time first, Other Bets pinned last):")
for s in stream_order:
    print("   ", s)
print("\nFY2025 flows (Total revenue fans out, then the waterfall):")
display(long[long.year == years[-1]].reset_index(drop=True))
''')

md(r"""
A quick look at one of the **tricky negative‑tax years** — note `Tax & other` shows in accounting
parentheses while every ribbon value stays non‑negative:
""")

code(r'''y = 2025
print(f"FY{y} signed node labels (what the viewer reads):")
for n in (HUB, GROSS_PROFIT, COST_OF_REV, OP_INCOME, OPEX, NET_INCOME, TAX_OTHER):
    print(f"  {n:<20} {labels_by_year[y][n]:>9}")
print(f"\n  Net income ({labels_by_year[y][NET_INCOME]}) > Operating income "
      f"({labels_by_year[y][OP_INCOME]})  ->  Tax & other is negative -> shown as ()")
# the underlying ribbon values are all non-negative even though the label is signed
ribbons = long[long.year == y]["value"]
print(f"\n  min ribbon value this year = {ribbons.min():.3f}B  (>= 0 -> drawable) ✓")
''')

md(r"""
### 4c · Align the share price to fiscal year‑end (already split‑adjusted)

`stock_by_year` already returned **one value per fiscal year**, taken at the **Dec‑31 close** and
**split‑adjusted**, so it lines up 1:1 with our `years` list and each Sankey frame — no further work
needed. We just keep the aligned list for the overlay.
""")

code(r'''stock_aligned = stock_by_year("GOOGL", years)
if stock_aligned and any(v is not None for v in stock_aligned):
    print("share price aligned to fiscal year-end (split-adjusted):")
    for yy, v in list(zip(years, stock_aligned))[:3] + [("...", "...")] + list(zip(years, stock_aligned))[-3:]:
        print(f"  FY{yy}: " + (f"${v:,.2f}" if isinstance(v, float) else str(v)))
else:
    print("offline -> overlay will fall back to total revenue")
''')

# =====================================================================
# 9. EXPLORE
# =====================================================================
md(r"""
## 5 · EXPLORE — sanity‑check the story before drawing it

Three quick views: the **revenue mix over time**, **when each stream appears** (the disclosure eras), and
the **share‑price curve with its dips**.
""")

code(r'''import matplotlib.pyplot as plt
plt.style.use("dark_background")

mix = wide_rec / 1e3      # US$ B, hedging folded in
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# (1) revenue mix over time — stacked area
axes[0].stackplot(mix.index, *[mix[s] for s in STREAMS], labels=STREAMS,
                  colors=[STREAM_COLORS[s] for s in STREAMS], alpha=0.9)
axes[0].set_title("Revenue mix over time (US$ B)"); axes[0].set_xlabel("fiscal year")
axes[0].legend(loc="upper left", fontsize=7); axes[0].margins(x=0)

# (2) when each stream "appears" (first non-zero year = its disclosure break-out)
first_year = {s: int(mix.index[(mix[s] > 0).values][0]) for s in STREAMS}
axes[1].barh(list(first_year), list(first_year.values()),
             color=[STREAM_COLORS[s] for s in first_year])
axes[1].set_title("First reported year per stream (disclosure era)")
axes[1].set_xlim(2002, 2020)
for s, fy in first_year.items():
    axes[1].text(fy + 0.2, s, str(fy), va="center", fontsize=8)
plt.tight_layout(); plt.show()
print("first reported year per stream:", first_year)
''')

code(r'''# (3) the share-price curve, with the dot-com-recovery / 2022-drawdown dips
if stock_aligned and any(v is not None for v in stock_aligned):
    ser = pd.Series(stock_aligned, index=years).dropna()
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(ser.index, ser.values, color="#FFB000", lw=2.2, marker="o", ms=4)
    ax.fill_between(ser.index, ser.values, color="#FFB000", alpha=0.12)
    ax.set_title("GOOGL split-adjusted year-end share price (US$) — the footer overlay")
    ax.set_xlabel("fiscal year"); ax.set_ylabel("US$ (split-adjusted)"); ax.margins(x=0.01)
    dips = ser.pct_change()
    worst = dips.idxmin()
    ax.annotate(f"FY{worst}: {dips[worst]*100:.0f}% YoY",
                xy=(worst, ser[worst]), xytext=(worst-4, ser[worst]+ser.max()*0.18),
                color="#FF6B6B", fontsize=9, arrowprops=dict(color="#FF6B6B", arrowstyle="->"))
    plt.tight_layout(); plt.show()
else:
    print("share price offline — skipping the price curve (overlay falls back to revenue).")
''')

# =====================================================================
# 10. PREPARE FOR THE SANKEY — layers, colours, annotations, holds
# =====================================================================
md(r"""
## 6 · PREPARE FOR THE SANKEY — shape & label the data

This is the **"labeling/shaping the data for `gradient_sankey`"** step. We assemble exactly what the
renderer wants:

- the tidy **`long`** flow DataFrame (built above),
- the **`LAYERS`** list in the **flipped** order: `[Total revenue] → [streams] → [Gross/Cost] →
  [OpInc/OpEx] → [Net/Tax]`,
- the **cohesive `node_colors`**: stream identity hues (`STREAM_COLORS`) + the waterfall emerald/red
  family (`PNL_COLORS`), with the **slate hub** `#5B6B7E`,
- **`FIXED_COLOR_NODES`** so the rank ramp never touches the waterfall,
- **`annotations`** (era markers + per‑year segment OI) and **`hold_periods`** (linger on FY2023).
""")

code(r'''# --- layers: FLIPPED. Total revenue leads as ONE block, fans out, then the P&L ---
LAYERS = [[HUB], stream_order] + PNL_LAYERS_TAIL
print("layers (left -> right):")
for i, L in enumerate(LAYERS):
    print(f"  L{i}: {L}")

# --- cohesive palette: stream identity hues + waterfall emerald/red, slate hub ---
node_colors = {s: STREAM_COLORS[s] for s in stream_order}
node_colors.update(PNL_COLORS)
print("\nTotal-revenue hub colour:", PNL_COLORS[HUB], "(calm slate, NOT white)")
print("stream colours:", {s: STREAM_COLORS[s] for s in stream_order})
print("fixed (off-ramp) nodes:", sorted(FIXED_COLOR_NODES))
''')

md(r"""
### Segment operating income — the *honest* fix for negatives 🧮

Alphabet's three **reportable segments** (Google Services, Google Cloud, Other Bets) include **losses**:
Other Bets loses money **every** year, and Cloud was a loss‑maker until it **first turned profitable in
FY2023**. You can't draw a negative ribbon — so we surface segment OI as a per‑year **caption**
(`annotations`), with **correct signs**, for FY2018+ (where the segment footnote exists), and **linger on
FY2023** (`hold_periods`) — the moment Cloud's sign flips from − to +. Earlier years get the **era‑marker**
captions instead (`ERA_NOTES`). `alphabet_flows` already assembled both into `annotations` (segment OI
overrides the era note where both exist).
""")

code(r'''hold = {str(CLOUD_FLIP_YEAR): 30} if CLOUD_FLIP_YEAR in years else {}   # ~1s pause at 30fps

print("Google Cloud segment operating income (US$ B) — watch the sign flip:")
for y in sorted(SEGMENT_OI):
    print(f"  FY{y}: {SEGMENT_OI[y][1]/1e3:+5.1f}b"
          + ("   <- first profitable year (hold here)" if y == CLOUD_FLIP_YEAR else ""))
print("\nera-marker captions (the years streams switch on):")
for k, v in ERA_NOTES.items():
    print(f"  {k}: {v}")
print("\nfinal annotation FY2025:\n  " + annotations["2025"])
print("first annotation FY2004:\n  " + annotations.get("2004", "(none)"))
''')

md(r"""
### Build the overlay (share price, with the revenue fallback)

We choose the **share price** overlay when yfinance is available, else fall back to **total revenue** — so
the notebook renders either way.
""")

code(r'''total   = [total_by_year[y] for y in years]
labels  = [labels_by_year[y] for y in years]
xlabels = [str(y) for y in years]

if stock_aligned and any(v is not None for v in stock_aligned):
    ov_series = [v if v is not None else float("nan") for v in stock_aligned]
    ov_label  = "GOOGL share price — year-end, split-adjusted  (US$)"
    ov_suffix, ov_badge, ov_color = "", "GOOGL", "#FFB000"     # amber-gold market line
else:
    ov_series, ov_label = total, "Total revenue  (US$ B / year)"
    ov_suffix, ov_badge, ov_color = "B", "GOOGL", "#7CFF6B"

print("overlay:", ov_label, "| badge:", ov_badge, "| color:", ov_color)
print("first/last overlay points:", round(ov_series[0], 2), "->", round(ov_series[-1], 2))
''')

# =====================================================================
# 11. LOAD / RENDER — stills + animation
# =====================================================================
md(r"""
## 7 · LOAD / RENDER

### Build the renderer and save stills

We hand the tidy `long` + the five **flipped** `LAYERS` to `from_dataframe`, then render stills that
bookend the story. `save_frame()` needs **no FFmpeg**, so this always works.

- **FY2004** — the IPO: advertising is ~99%; YouTube/Cloud/Other Bets are folded in (hidden),
- **FY2018** — all six streams reported; Cloud is a sliver near the bottom,
- **FY2025** — Cloud has climbed near the top of the streams, the whole waterfall far bigger.

The **\$ axis** anchors on the **Total‑revenue hub** (`yaxis_node=HUB`). *(Ranking **colours** and the
segment‑OI **annotations** are animation‑only, so the stills show each stream in its fixed identity
colour — the clip below adds the live race ordering and the caption.)*
""")

code(r'''sk = Sankey.from_dataframe(
    df=long, layers=LAYERS,
    time_col="year", source_col="source", target_col="target", value_col="value",
    node_colors=node_colors,
)
anchor = HUB        # the $ axis scales to the Total-revenue block (the flipped layer 0)
print(f"{len(sk.frames)} frames (years):",
      sk.frames[0]['time_label'], "->", sk.frames[-1]['time_label'])

TITLE = "Alphabet — revenue by stream → income statement"
still_design = dict(
    figsize=(18, 10), dpi=95, font_size=12, title_fontsize=16, padding=2.4,
    margin_top=0.18, margin_bottom=0.22,
    ranking_mode=True, stacked_mode=True,
    theme="dark", link_glow=1, link_alpha=0.55, n_segments=100,
    yaxis_node=anchor, yaxis_suffix=" B", yaxis_gap=0.15,
    layer0_label_side="left", value_prefix="$",
)

year_to_index = {f["time_label"]: i for i, f in enumerate(sk.frames)}
still_paths = []
for yr in ("2004", "2018", "2025"):
    p = os.path.join(ASSETS, f"alphabet_{yr}.png")
    sk.save_frame(p, frame_index=year_to_index[yr], title=TITLE,
                  node_value_labels=labels_by_year[int(yr)], **still_design)
    still_paths.append(p); print("saved", p)

for p in still_paths:
    display(show(p, w=1000))
''')

md(r"""
### One short animation 🎬

A tiny clip to prove the motion, with the **full** flagship design: dark theme, **static curated stream
colours** with the live **ranking race** (`dynamic_color_mode="static"` + `ranking_mode` — *not* a turbo
rank‑recolour), `fixed_color_nodes` keeping the waterfall emerald/red, the **\$ axis**, the **segment‑OI
caption** parked in the empty **bottom‑right** (`annotation_xy=(0.985, 0.27)`, `ha="right"`, `va="bottom"`),
the **hold** on FY2023, and the **GOOGL share‑price** overlay. We keep it short for speed; it **skips
gracefully** if FFmpeg isn't installed.
""")

code(r'''sample_mp4 = os.path.join(ASSETS, "alphabet_sample.mp4")
try:
    sk.animate(
        sample_mp4,
        figsize=(18, 10), fps=24, duration_seconds=12, quality="medium", n_workers=2,
        title=TITLE,
        font_size=12, title_fontsize=17, padding=2.4,
        margin_top=0.14, margin_bottom=0.22,
        ranking_mode=True, stacked_mode=True,
        dynamic_color_mode="static", dynamic_colormap="turbo",   # STABLE curated hues; ranking re-orders
        fixed_color_nodes=FIXED_COLOR_NODES,                      # ...waterfall stays emerald/red
        link_glow=1, link_alpha=0.55, n_segments=100,
        node_value_labels_per_frame=labels,                      # signed $ figures (losses in parens)
        yaxis_node=anchor, yaxis_suffix=" B", yaxis_gap=0.15,
        layer0_label_side="left", value_prefix="$",
        annotations=annotations,                                 # era + segment-OI caption per year
        annotation_xy=(0.985, 0.27), annotation_ha="right",      # park it in the empty BOTTOM-RIGHT
        annotation_va="bottom",                                  # (above the footer), clear of layer-3
        hold_periods=hold,                                       # pause on Cloud's first profit (FY2023)
        overlay_series=ov_series, overlay_x_labels=xlabels,
        overlay_label=ov_label,
        overlay_color=ov_color, overlay_value_suffix=ov_suffix, overlay_badge=ov_badge,
        overlay_band=(0.18, 0.56),                               # push the footer line lower
    )
    from IPython.display import Video
    print("rendered:", sample_mp4)
    display(Video(sample_mp4, embed=True, width=900))
except Exception as e:
    print("Skipping the MP4 (FFmpeg likely not installed) — the stills above tell the story.")
    print("Reason:", repr(e))
''')

# =====================================================================
# 12. Recap
# =====================================================================
md(r"""
## 8 · Recap, honesty notes & exercises

🎉 **You rebuilt the flagship company‑financials reel** — Alphabet's whole money story in one conserved
diagram, FY2004 → FY2025 — from scratch, with **no external project files**.

**The ETL, in one breath:** EDGAR **companyfacts** P&L totals (+ a hardcoded 2004–2012 backfill) →
discover it can't disaggregate revenue → use the **10‑K stream table** (with era folding so pre‑break‑out
streams sit at \$0) → fold **hedging** into Other Bets and **reconcile** to the EDGAR total **to the
dollar** → **derive** the balanced waterfall residuals → split each stream into Gross/Cost at the
**company‑wide margin** → assemble the **flipped five layers** (Total revenue → streams → cost/gross →
opex/op‑income → tax/net) → **static curated stream colours** + a **ranking race** + **`fixed_color_nodes`**
on the waterfall → **era + segment‑OI annotations** + a **hold** on FY2023 → pull the **GOOGL
split‑adjusted** price for the overlay → `save_frame` stills + `animate`.

### 💙 Honesty notes (the things that make it trustworthy)

1. **The per‑stream cost is company‑wide.** Alphabet discloses cost/gross/tax only at the company level,
   so every stream is carried into the waterfall at the *same* company‑wide margin — the title says so.
   We never imply a per‑stream cost.
2. **Era folding is a disclosure choice, not a business jump.** YouTube ads (2017), Cloud (2018) and
   Other Bets (2015) appear as \$0 before their break‑out years; annotations mark the IPO/reorg/break‑out.
3. **The price is split‑adjusted.** `auto_adjust=True` back‑applies the 2014 and 2022 splits for one
   continuous curve; offline it falls back to total revenue.
4. **Reconciled to the dollar.** The six streams + hedging equal the independent EDGAR total every year;
   the build *asserts* it.
5. **Negatives stay honest.** Where Net income > Operating income, the ribbon clamps to `[0, parent]`
   but the label shows the real signed figure in accounting parentheses.

### 🧪 Try it yourself

1. **Margins.** Add `operating_margin = operating_income / revenue` and animate it as a second overlay band.
2. **Another colormap.** Pass a different `dynamic_colormap` (`plasma`, `viridis`) — `ranking_mode` still
   re‑orders the streams.
3. **A different window.** Call `alphabet_flows(2018, 2025)` to focus on the Cloud‑profitability era.
4. **Another filer.** Point the companyfacts loader at a different CIK (Microsoft `0000789019`, Apple
   `0000320193`), then find *its* 10‑K revenue disaggregation table.
5. **Share of revenue.** Normalise each stream by the year's total to animate the **revenue mix %**.

### 🔗 Links

- The library: `gradient_sankey.py` (repo root) · `pip install gradient-sankey`
- SEC EDGAR companyfacts: https://data.sec.gov/api/xbrl/companyfacts/CIK0001652044.json (needs a User‑Agent)
- Alphabet 10‑K filings: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001652044&type=10-K
- Companion tutorials: `notebooks/09_tutorial_bigtech_amazon.ipynb` (streams → one hub),
  `notebooks/01_tutorial_nvidia_income.ipynb` (same SEC P&L world)

*Built com todo cuidado e carinho. Data: SEC EDGAR (companyfacts P&L) + Alphabet 10‑K revenue
disaggregation & segment footnotes, reconciled to the dollar; GOOGL price via yfinance. Happy hacking!*
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
# Write to notebooks/ (the parent of this builders/ dir), keeping the output name.
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "11_tutorial_alphabet.ipynb")
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
