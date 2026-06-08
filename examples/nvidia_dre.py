"""
NVIDIA income statement (DRE) waterfall - SEC EDGAR, deep quarterly history.
Builds an animated multi-layer Sankey of the P&L flow to net income.

Data: https://data.sec.gov  (XBRL company concepts, free, US filers)

Robust approach: pull only 4 clean series that go back to 2008
    Revenue, Gross Profit, Operating Income, Net Income
and DERIVE the "leak" flows as residuals so the Sankey always balances:
    COGS       = Revenue        - Gross Profit
    OpEx       = Gross Profit   - Operating Income
    Tax+Other  = Operating Inc. - Net Income
"""
import requests
import pandas as pd
from datetime import date

H = {'User-Agent': 'Felipe Gabriel felipegabriel.mecanica@gmail.com'}
CIK = '0001045810'  # NVIDIA

# Revenue tag changed over time -> merge these (later ones win per period)
REVENUE_TAGS = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax']


def _days(s, e):
    return (date.fromisoformat(e) - date.fromisoformat(s)).days


def concept_quarterly(tag: str) -> dict:
    """Return {period_end -> value} for ~3-month (quarterly) datapoints,
    deduped keeping the most recently FILED figure."""
    url = f'https://data.sec.gov/api/xbrl/companyconcept/CIK{CIK}/us-gaap/{tag}.json'
    r = requests.get(url, headers=H, timeout=30)
    if r.status_code != 200:
        return {}
    vals = r.json().get('units', {}).get('USD', [])
    best = {}  # end -> (filed, val)
    for v in vals:
        if not v.get('start'):
            continue
        if 80 <= _days(v['start'], v['end']) <= 100:   # quarter (~91 days)
            end = v['end']
            filed = v.get('filed', '')
            if end not in best or filed > best[end][0]:
                best[end] = (filed, v['val'])
    return {e: val for e, (f, val) in best.items()}


def merged_revenue() -> dict:
    out = {}
    for tag in REVENUE_TAGS:
        for end, val in concept_quarterly(tag).items():
            out.setdefault(end, val)   # first (Revenues) wins; alias fills gaps
    # fill gaps from alias where Revenues missing
    alias = concept_quarterly(REVENUE_TAGS[1])
    for end, val in alias.items():
        out.setdefault(end, val)
    return out


def build():
    rev = merged_revenue()
    gp = concept_quarterly('GrossProfit')
    oi = concept_quarterly('OperatingIncomeLoss')
    ni = concept_quarterly('NetIncomeLoss')

    ends = sorted(set(rev) & set(gp) & set(oi) & set(ni))
    rows = []
    for e in ends:
        rows.append({
            'period': e,
            'revenue': rev[e],
            'gross_profit': gp[e],
            'op_income': oi[e],
            'net_income': ni[e],
        })
    df = pd.DataFrame(rows)
    # derived leaks
    df['cogs'] = df['revenue'] - df['gross_profit']
    df['opex'] = df['gross_profit'] - df['op_income']
    df['tax_other'] = df['op_income'] - df['net_income']
    return df


# Waterfall structure: node -> (layer, is_leak, signed source column)
LAYERS = [
    ["Revenue"],
    ["Gross Profit", "Cost of Revenue"],
    ["Operating Income", "Operating Expenses"],
    ["Net Income", "Tax & Other"],
]

# Each flow: (source, target, column_for_signed_value)
FLOWS = [
    ("Revenue", "Gross Profit", "gross_profit"),
    ("Revenue", "Cost of Revenue", "cogs"),
    ("Gross Profit", "Operating Income", "op_income"),
    ("Gross Profit", "Operating Expenses", "opex"),
    ("Operating Income", "Net Income", "net_income"),
    ("Operating Income", "Tax & Other", "tax_other"),
]


def quarter_label(period: str) -> str:
    """Calendar-quarter label from an end date, e.g. '2024 Q1'."""
    y, m, _ = period.split("-")
    q = (int(m) - 1) // 3 + 1
    return f"{y} Q{q}"


def to_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Long-form Sankey flows. value = |signed| (bar size by magnitude);
    'neg' marks flows whose underlying figure is negative -> label in (parentheses)."""
    rows = []
    for _, r in df.iterrows():
        label = quarter_label(r["period"])
        for src, tgt, col in FLOWS:
            signed = r[col]
            rows.append({
                "period": r["period"],
                "quarter": label,
                "source": src,
                "target": tgt,
                "value": abs(signed),       # módulo -> tamanho da barra
                "signed": signed,
                "neg": signed < 0,          # -> rótulo entre parênteses
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = build()
    flows = to_flows(df)
    import os
    out = os.path.join(os.path.dirname(__file__), "nvidia_dre.csv")
    flows.to_csv(out, index=False)
    print(f"Saved {len(flows)} flow rows ({df['period'].nunique()} quarters) -> {out}\n")
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 100)
    # show in $B
    show = df.copy()
    for c in ['revenue', 'gross_profit', 'op_income', 'net_income', 'cogs', 'opex', 'tax_other']:
        show[c] = (show[c] / 1e9).round(2)
    print(f"{len(df)} quarters: {df['period'].iloc[0]} .. {df['period'].iloc[-1]}")
    print(show.to_string(index=False))
    # flag any negatives that would break the Sankey
    bad = df[(df[['revenue','gross_profit','op_income','net_income','cogs','opex','tax_other']] < 0).any(axis=1)]
    print(f"\nQuarters with a negative flow (would break Sankey): {len(bad)}")
    if len(bad):
        print(bad['period'].tolist())
