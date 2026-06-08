"""
NVIDIA income statement (P&L / "DRE") waterfall - SEC EDGAR, deep quarterly history.
Builds the data for an animated multi-layer Sankey of the flow down to net income.

Data: https://data.sec.gov  (XBRL company concepts, free, US filers).

Approach: pull only four clean series that go back to ~2008
    Revenue, Gross Profit, Operating Income, Net Income
and DERIVE the "leak" flows as residuals so the Sankey always balances:
    COGS       = Revenue          - Gross Profit
    OpEx       = Gross Profit     - Operating Income
    Tax+Other  = Operating Income - Net Income

Two subtleties that this module handles (and the naive version got wrong):

1. Fiscal Q4 is never reported by XBRL as a standalone ~91-day duration -- only the
   full year (~365d) and the nine-month YTD (~273d) exist. So we DERIVE it:
       Q4 = FullYear - NineMonthYTD      (fallback: FullYear - sum(Q1..Q3)).
   Without this, the time series has a hole every fiscal year.

2. NVIDIA's fiscal year ends in late January, so we label quarters by FISCAL period
   ("FY2025 Q1") rather than by calendar quarter.

Network calls are retried with backoff and a proper SEC User-Agent. Results are
cached to ``nvidia_dre_wide.csv`` so renders are reproducible and work offline;
pass ``refresh=True`` (or ``--refresh`` to the script) to re-scrape.
"""
import os
import requests
import pandas as pd
from datetime import date
from requests.adapters import HTTPAdapter

try:                                    # urllib3 ships with requests
    from urllib3.util.retry import Retry
except ImportError:                     # pragma: no cover - very old setups
    from requests.packages.urllib3.util.retry import Retry

# SEC requires a descriptive User-Agent identifying the requester.
H = {'User-Agent': os.environ.get('SEC_USER_AGENT', 'gradient-sankey example (contact: you@example.com)')}
CIK = '0001045810'  # NVIDIA
HERE = os.path.dirname(os.path.abspath(__file__))
WIDE_CSV = os.path.join(HERE, "nvidia_dre_wide.csv")
FLOWS_CSV = os.path.join(HERE, "nvidia_dre.csv")

# Revenue tag changed over time; the later (current-standard) tag wins per period.
REVENUE_TAGS = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax']
REQUIRED_COLS = ['revenue', 'gross_profit', 'op_income', 'net_income']


def _session() -> requests.Session:
    """A requests session that retries on SEC throttling (429) and 5xx."""
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(['GET']))
    s.mount('https://', HTTPAdapter(max_retries=retry))
    s.headers.update(H)
    return s


_SESSION = _session()


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _latest_by_end(units, lo: int, hi: int) -> dict:
    """{period_end -> value} for datapoints whose duration is in [lo, hi] days,
    keeping the most recently FILED figure on duplicates."""
    best = {}  # end -> (filed, val)
    for v in units:
        if not v.get('start'):
            continue
        if lo <= _days(v['start'], v['end']) <= hi:
            end, filed = v['end'], v.get('filed', '')
            if end not in best or filed > best[end][0]:
                best[end] = (filed, v['val'])
    return {e: val for e, (f, val) in best.items()}


def concept_periods(tag: str) -> dict:
    """Return {period_end -> quarterly value} for a us-gaap concept, with the
    missing fiscal Q4 DERIVED from the full-year and nine-month figures."""
    url = f'https://data.sec.gov/api/xbrl/companyconcept/CIK{CIK}/us-gaap/{tag}.json'
    r = _SESSION.get(url, timeout=30)
    if r.status_code == 404:            # tag simply doesn't exist for this filer
        return {}
    r.raise_for_status()
    units = r.json().get('units', {}).get('USD', [])

    quarters = _latest_by_end(units, 80, 100)    # ~91-day quarters (Q1..Q3)
    nine_m = _latest_by_end(units, 260, 285)     # ~273-day nine-month YTD
    annual = _latest_by_end(units, 350, 380)     # ~365-day full year

    # Derive each fiscal Q4 = FullYear - NineMonthYTD (fallback: FY - sum(Q1..Q3)).
    for ae, av in annual.items():
        if ae in quarters:              # a real standalone Q4 already exists
            continue
        cand = [ne for ne in nine_m if 80 <= _days(ne, ae) <= 100]
        if cand:
            quarters[ae] = av - nine_m[max(cand)]
        else:
            three = sorted(qe for qe in quarters if 30 < _days(qe, ae) < 300)[-3:]
            if len(three) == 3:
                quarters[ae] = av - sum(quarters[qe] for qe in three)
    return quarters


def merged_revenue() -> dict:
    """Merge the historical revenue tags; the later (current-standard) tag wins."""
    out = {}
    for tag in REVENUE_TAGS:            # later tag overrides earlier per period
        out.update(concept_periods(tag))
    return out


def fiscal_label(period: str) -> str:
    """Fiscal-quarter label for NVIDIA (fiscal year ends in late January).

    A period ending in Jan/Feb is Q4 of that fiscal year; Mar-May -> Q1 of FY+1,
    Jun-Aug -> Q2, Sep-Dec -> Q3. Labels sort chronologically as plain strings.
    """
    y, m, _ = (int(x) for x in period.split("-"))
    if m <= 2:
        fy, q = y, 4
    elif m <= 5:
        fy, q = y + 1, 1
    elif m <= 8:
        fy, q = y + 1, 2
    else:
        fy, q = y + 1, 3
    return f"FY{fy} Q{q}"


def build(refresh: bool = False) -> pd.DataFrame:
    """Return the wide quarterly P&L DataFrame.

    Columns: period (ISO end date), quarter (fiscal label), revenue, gross_profit,
    op_income, net_income, cogs, opex, tax_other. Cached to ``nvidia_dre_wide.csv``
    for offline / reproducible runs; pass ``refresh=True`` to re-scrape SEC EDGAR.
    """
    if not refresh and os.path.exists(WIDE_CSV):
        return pd.read_csv(WIDE_CSV)

    rev = merged_revenue()
    gp = concept_periods('GrossProfit')
    oi = concept_periods('OperatingIncomeLoss')
    ni = concept_periods('NetIncomeLoss')

    for name, d in [('Revenue', rev), ('GrossProfit', gp),
                    ('OperatingIncomeLoss', oi), ('NetIncomeLoss', ni)]:
        if not d:
            raise RuntimeError(
                f"SEC EDGAR returned no quarterly data for '{name}'. The API may be "
                "throttling (set a real SEC_USER_AGENT env var) or the tag changed."
            )

    ends = sorted(set(rev) & set(gp) & set(oi) & set(ni))
    if not ends:
        raise RuntimeError("No overlapping quarters across the four required concepts.")

    rows = [{
        'period': e,
        'quarter': fiscal_label(e),
        'revenue': rev[e],
        'gross_profit': gp[e],
        'op_income': oi[e],
        'net_income': ni[e],
    } for e in ends]
    df = pd.DataFrame(rows)
    df['cogs'] = df['revenue'] - df['gross_profit']
    df['opex'] = df['gross_profit'] - df['op_income']
    df['tax_other'] = df['op_income'] - df['net_income']

    df.to_csv(WIDE_CSV, index=False)    # cache for offline / reproducible renders
    return df


# Waterfall structure: layers (left -> right) and the flows between them.
LAYERS = [
    ["Revenue"],
    ["Gross Profit", "Cost of Revenue"],
    ["Operating Income", "Operating Expenses"],
    ["Net Income", "Tax & Other"],
]

# Each flow: (source, target, wide-df column holding the SIGNED value)
FLOWS = [
    ("Revenue", "Gross Profit", "gross_profit"),
    ("Revenue", "Cost of Revenue", "cogs"),
    ("Gross Profit", "Operating Income", "op_income"),
    ("Gross Profit", "Operating Expenses", "opex"),
    ("Operating Income", "Net Income", "net_income"),
    ("Operating Income", "Tax & Other", "tax_other"),
]


def to_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy long-form Sankey flows, one row per (quarter, flow).

    Emitted columns:
        period   - ISO period-end date (chronological key)
        quarter  - fiscal label, e.g. "FY2025 Q1"
        source   - source node name
        target   - target node name
        value    - |signed|  (bar size is the magnitude)
        signed   - the real (possibly negative) figure
        neg      - True if ``signed`` < 0  (render with accounting parentheses)

    NOTE: a stage with a loss (e.g. operating loss) makes ``value`` exceed the
    profit spine, so the waterfall stops balancing for that quarter. Such rows are
    flagged via ``neg``; the reel defaults to start_year=2015, where NVIDIA's
    stages are profitable.
    """
    rows = []
    for _, r in df.iterrows():
        for src, tgt, col in FLOWS:
            signed = r[col]
            rows.append({
                "period": r["period"],
                "quarter": r["quarter"],
                "source": src,
                "target": tgt,
                "value": abs(signed),       # magnitude -> bar size
                "signed": signed,
                "neg": signed < 0,          # -> parenthesized label
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Scrape/cache NVIDIA P&L data from SEC EDGAR.")
    ap.add_argument("--refresh", action="store_true", help="Re-scrape instead of using the cached CSV.")
    args = ap.parse_args()

    df = build(refresh=args.refresh)
    flows = to_flows(df)
    flows.to_csv(FLOWS_CSV, index=False)
    print(f"Saved {len(flows)} flow rows ({df['period'].nunique()} quarters) -> {FLOWS_CSV}\n")

    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 200)
    show = df.copy()
    for c in ['revenue', 'gross_profit', 'op_income', 'net_income', 'cogs', 'opex', 'tax_other']:
        show[c] = (show[c] / 1e9).round(2)
    print(f"{len(df)} quarters: {df['period'].iloc[0]} .. {df['period'].iloc[-1]}  (values in $B)")
    print(show.to_string(index=False))

    bad = df[(df[['revenue', 'gross_profit', 'op_income', 'net_income',
                  'cogs', 'opex', 'tax_other']] < 0).any(axis=1)]
    print(f"\nQuarters with a negative flow (waterfall won't balance there): {len(bad)}")
    if len(bad):
        print(bad['period'].tolist())
