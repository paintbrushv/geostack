"""Build publishable multifamily supply exhibits from CoStar exports.

Reads every export in data/costar/raw/ (see geostack.costar_market) and writes
PNG + SVG + CSV per exhibit to:

    outputs/costar_exhibits/<YYYY-MM-DD>/

Every exhibit cites "Source: CoStar". Headlines are computed from the data with
neutral fallbacks; review them before posting.

Usage:
    python scripts/build_costar_exhibits.py
    python scripts/build_costar_exhibits.py --brand "Your Name"
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from geostack import exhibit_style as xs
from geostack.costar_market import as_of, load_costar, series
from geostack.housing_supply import TEXAS_METROS

ROOT = Path(__file__).resolve().parent.parent
BIG4 = list(TEXAS_METROS)
NAMES = dict(TEXAS_METROS)
STARTS = xs.CATEGORICAL[2]
DELIVERIES = xs.SUPPLY
ABSORPTION = xs.DEMAND


def _signed(v: float, decimals: int = 0, unit: str = "%") -> str:
    if round(v, decimals) == 0:
        return f"0{unit}"
    sign = "+" if v > 0 else "−"
    return f"{sign}{abs(v):.{decimals}f}{unit}"


def _ts(idx) -> pd.DatetimeIndex:
    """Quarter periods -> mid-quarter timestamps for plotting."""
    return pd.PeriodIndex(idx, freq="Q").to_timestamp() + pd.Timedelta(days=45)


class Ctx:
    def __init__(self, df: pd.DataFrame, out: Path):
        self.df = df
        self.out = out
        self.as_of = as_of(df)
        hist = df[(df.status == "history") & (df.concept == "Inventory Units")]
        self.last_hist = hist.period.max()
        fc = df[df.status == "forecast"]
        self.fc_end = fc.period.max() if len(fc) else self.last_hist
        self.source = (f"CoStar, multifamily, as of {self.as_of.year} Q{self.as_of.quarter} "
                       f"(forecasts are CoStar's Base Case)")

    def s(self, concept, **kw) -> pd.DataFrame:
        return series(self.df, concept, **kw)

    @property
    def last_label(self) -> str:
        return f"{self.last_hist.year} Q{self.last_hist.quarter}"

    def band(self, ax):
        xs.forecast_band(ax, _ts([self.last_hist])[0], _ts([self.fc_end])[0] +
                         pd.Timedelta(days=60), label="CoStar forecast")


# ---------------------------------------------------------------------------
# Big-four Texas metros
# ---------------------------------------------------------------------------

def ex_pipeline(c: Ctx, start="2008Q1"):
    st = c.s("Construction Starts Units 12 Mo")[BIG4].loc[start:]
    dl = c.s("Net Delivered Units 12 Mo")[BIG4].loc[start:]
    ab = c.s("Absorption Units 12 Mo")[BIG4].loc[start:]
    lh = c.last_hist
    below = [m for m in BIG4 if st.loc[lh, m] < dl.loc[lh, m]]
    if len(below) == len(BIG4):
        title = "Apartment starts have fallen below deliveries in all four big Texas metros"
    elif below:
        title = f"Apartment starts are running below deliveries in {len(below)} of 4 Texas metros"
    else:
        title = "Apartment starts, deliveries and absorption, Texas' largest metros"
    fig, axes = xs.new_exhibit(
        title,
        f"Multifamily units, trailing 12 months, {start[:4]} – {c.last_label}, with CoStar's "
        f"forecast for deliveries and absorption through {c.fc_end.year}",
        source=c.source,
        legend=[("Construction starts", STARTS), ("Deliveries (net)", DELIVERIES),
                ("Absorption", ABSORPTION)],
        nrows=2, ncols=2, sharex=True, figsize=(8, 6.0))
    ymax = max(st.max().max(), dl.max().max(), ab.max().max()) * 1.1
    for ax, m in zip(axes.flat, BIG4):
        c.band(ax)
        ax.plot(_ts(dl.index), dl[m], color=DELIVERIES, lw=1.7)
        ax.plot(_ts(ab.index), ab[m], color=ABSORPTION, lw=1.7)
        sm = st[m].dropna()
        ax.plot(_ts(sm.index), sm, color=STARTS, lw=1.7)
        ax.axhline(0, color=xs.BASELINE, lw=0.8)
        ax.set_title(NAMES[m], loc="left")
        ax.text(1, 1.04, f"{c.last_label}: {xs.fmt_k(sm.iloc[-1])} starts · "
                f"{xs.fmt_k(dl.loc[lh, m])} delivered", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8, color=xs.INK_2)
        ax.yaxis.set_major_formatter(xs.FuncFormatter(lambda v, _: xs.fmt_k(v)))
        ax.set_ylim(min(0, ab.min().min() * 1.1), ymax)
    pd.concat({"starts_12mo": st, "deliveries_12mo": dl, "absorption_12mo": ab}, axis=1) \
        .rename(columns=NAMES, level=1).to_csv(c.out / "20_tx_pipeline_starts_deliveries.csv")
    xs.save_exhibit(fig, c.out / "20_tx_pipeline_starts_deliveries")
    return pd.DataFrame({"starts": st.loc[lh], "deliveries": dl.loc[lh], "absorption": ab.loc[lh]})


def ex_uc_share(c: Ctx, start="2005Q1"):
    uc = c.s("Under Construction Units")[BIG4]
    inv = c.s("Inventory Units")[BIG4]
    pct = (100 * uc / inv).loc[start:c.last_hist].dropna(how="all")
    lh = c.last_hist
    lowest_since = {}
    for m in BIG4:
        prior = pct[m].loc[:lh - 1]
        lower = prior[prior <= pct.loc[lh, m]]
        lowest_since[m] = lower.index.max() if len(lower) else None
    multi_year = {m: p for m, p in lowest_since.items() if p is not None and p.year <= lh.year - 8}
    if len(multi_year) == len(BIG4):
        title = (f"Texas apartment pipelines are at their smallest share of inventory in "
                 f"{lh.year - max(p.year for p in multi_year.values())}+ years")
    elif len(multi_year) >= 2:
        names = [NAMES[m].replace("Dallas-Fort Worth", "DFW") for m in multi_year]
        yrs = sorted(p.year for p in multi_year.values())
        title = (f"{', '.join(names[:-1])} and {names[-1]} pipelines are the thinnest since "
                 f"{yrs[0]}–{str(yrs[-1])[2:]}")
    else:
        title = "Apartments under construction as a share of inventory, Texas' largest metros"
    fig, ax = xs.new_exhibit(
        title, f"Multifamily units under construction ÷ existing inventory, {start[:4]} – "
               f"{c.last_label}", source=c.source,
        legend=[(NAMES[m], xs.METRO_COLORS[m]) for m in BIG4], right=1.75)
    for m in BIG4:
        ax.plot(_ts(pct.index), pct[m], color=xs.METRO_COLORS[m])
    ax.set_ylim(0, pct.max().max() * 1.1)
    ax.yaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10], integer=True))
    last_x = _ts([lh])[0]
    ax.set_xlim(_ts(pct.index)[0], last_x + pd.Timedelta(days=30))
    ax.yaxis.set_major_formatter(xs.fmt_pct(0))
    xs.end_labels(ax, [(last_x, pct.loc[lh, m], f"{NAMES[m]}  {pct.loc[lh, m]:.1f}%",
                        xs.METRO_COLORS[m]) for m in BIG4], x_pad_frac=0.006)
    pct.rename(columns=NAMES).round(3).to_csv(c.out / "21_tx_under_construction_share.csv")
    xs.save_exhibit(fig, c.out / "21_tx_under_construction_share")
    return {NAMES[m]: (round(pct.loc[lh, m], 2), str(lowest_since[m])) for m in BIG4}


def _forecast_lines(c: Ctx, ax, w: pd.DataFrame, fmt, ylim, label_at_end=True):
    ax.set_ylim(*ylim)  # before end labels, which space themselves by the y range
    c.band(ax)
    for m in BIG4:
        s = w[m].dropna()
        ax.plot(_ts(s.index), s, color=xs.METRO_COLORS[m])
        ax.plot(_ts([c.last_hist])[0], w.loc[c.last_hist, m], "o", ms=5,
                color=xs.METRO_COLORS[m], mec=xs.SURFACE, mew=1.2, zorder=4)
    end = w.dropna(how="all").index.max()
    end_x = _ts([end])[0]
    ax.set_xlim(_ts(w.index)[0], end_x + pd.Timedelta(days=30))
    if label_at_end:
        xs.end_labels(ax, [(end_x, w.loc[end, m], f"{NAMES[m]}  {fmt(w.loc[end, m])}",
                            xs.METRO_COLORS[m]) for m in BIG4], x_pad_frac=0.006)


def ex_vacancy(c: Ctx, start="2005Q1"):
    w = (100 * c.s("Vacancy Rate")[BIG4]).loc[start:]
    lh, end = c.last_hist, w.dropna(how="all").index.max()
    falling = [m for m in BIG4 if w.loc[end, m] < w.loc[lh, m] - 0.5]
    if len(falling) == len(BIG4):
        title = (f"CoStar expects apartment vacancy to fall in all four big Texas metros "
                 f"by {end.year}")
    else:
        title = "Apartment vacancy, Texas' largest metros"
    nb = "\u00a0"  # keep "San Antonio 15.0%" together when the subtitle wraps
    now = ", ".join(f"{NAMES[m].replace('Dallas-Fort Worth', 'DFW').replace(' ', nb)}"
                    f"{nb}{w.loc[lh, m]:.1f}%" for m in BIG4)
    fig, ax = xs.new_exhibit(
        title, f"Multifamily vacancy rate, {start[:4]} – {end.year}. Dots mark {c.last_label}: "
               f"{now}", source=c.source,
        legend=[(NAMES[m], xs.METRO_COLORS[m]) for m in BIG4], right=1.75)
    _forecast_lines(c, ax, w, lambda v: f"{v:.1f}%", ylim=(0, w.max().max() * 1.12))
    ax.yaxis.set_major_formatter(xs.fmt_pct(0))
    w.rename(columns=NAMES).round(2).to_csv(c.out / "22_tx_vacancy.csv")
    xs.save_exhibit(fig, c.out / "22_tx_vacancy")
    return w.loc[[lh, end]].rename(columns=NAMES)


def ex_rent_growth(c: Ctx, start="2005Q1"):
    w = (100 * c.s("Market Effective Rent Growth 12 Mo")[BIG4]).loc[start:]
    lh = c.last_hist
    nxt = pd.Period(f"{lh.year + 1}Q4", freq="Q")
    neg_now = [m for m in BIG4 if w.loc[lh, m] < 0]
    pos_next = [m for m in BIG4 if nxt in w.index and w.loc[nxt, m] > 0]
    if len(neg_now) == len(BIG4) and len(pos_next) == len(BIG4):
        title = (f"Texas apartment rents are still falling; CoStar sees growth returning "
                 f"in {nxt.year}")
    elif len(neg_now) == len(BIG4):
        title = "Effective apartment rents are falling in all four big Texas metros"
    else:
        title = "Effective apartment rent growth, Texas' largest metros"
    fig, ax = xs.new_exhibit(
        title, f"Year-over-year change in effective rent per unit, {start[:4]} – "
               f"{w.dropna(how='all').index.max().year} (dot = {c.last_label})",
        source=c.source, legend=[(NAMES[m], xs.METRO_COLORS[m]) for m in BIG4])
    lo, hi = w.min().min(), w.max().max()
    # lines converge in the forecast, so the legend carries identity (no stacked end labels)
    _forecast_lines(c, ax, w, lambda v: _signed(v, 1), ylim=(lo - 1, hi + 1.5),
                    label_at_end=False)
    ax.axhline(0, color=xs.INK_2, lw=0.9)
    ax.yaxis.set_major_formatter(xs.FuncFormatter(lambda v, _: _signed(v)))
    w.rename(columns=NAMES).round(2).to_csv(c.out / "23_tx_rent_growth.csv")
    xs.save_exhibit(fig, c.out / "23_tx_rent_growth")
    return w.loc[[lh, nxt]].rename(columns=NAMES)


def ex_deliveries_per_household(c: Ctx, windows=((2015, 2019), (2020, 2024), (2025, 2030))):
    dl = c.s("Net Delivered Units")[BIG4]
    hh = c.s("Households")[BIG4]
    rows = []
    for a, b in windows:
        hh_gain = hh.loc[pd.Period(f"{b}Q4")] - hh.loc[pd.Period(f"{a - 1}Q4")]
        deliv = dl.loc[pd.Period(f"{a}Q1"):pd.Period(f"{b}Q4")].sum()
        for m in BIG4:
            rows.append({"metro": NAMES[m], "cbsa_code": m, "window": f"{a}–{str(b)[2:]}",
                         "deliveries": deliv[m], "households_added": hh_gain[m],
                         "ratio": deliv[m] / hh_gain[m]})
    r = pd.DataFrame(rows)
    w = r.pivot(index="cbsa_code", columns="window", values="ratio").loc[BIG4]
    labels = list(w.columns)
    cur, fut = labels[-2], labels[-1]
    drop = 100 * (1 - w[fut] / w[cur])
    falling = [m for m in BIG4 if drop[m] > 10]
    if len(falling) == len(BIG4):
        title = (f"Apartment deliveries per new household are set to fall {drop.min():.0f}–"
                 f"{drop.max():.0f}% across Texas' big four")
    elif falling:
        names = [NAMES[m].replace("Dallas-Fort Worth", "DFW") for m in falling]
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        title = f"Fewer new apartments per new household ahead in {joined}"
    else:
        title = "Apartments delivered per new household, Texas' largest metros"
    ramp = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal blue steps (250 / 450 / 650)
    fig, ax = xs.new_exhibit(
        title,
        "Net multifamily units delivered ÷ households added (all tenures), by period",
        source=c.source,
        note=f"{fut} combines history through {c.last_label} with CoStar's Base Case forecast. "
             "Multifamily only; single-family and other housing meet the rest of demand.",
        legend=[(f"{lab}{' (forecast)' if lab == fut else ''}", col)
                for lab, col in zip(labels, ramp)], legend_kind="patch",
        figsize=(8, 5.4), left=1.45)
    y = np.arange(len(BIG4))
    h = 0.22
    for i, (lab, col) in enumerate(zip(labels, ramp)):
        off = (i - 1) * (h + 0.05)
        vals = w[lab].values
        ax.barh(y + off, vals, height=h, color=col, edgecolor=xs.SURFACE, lw=0.8)
        for yy, v in zip(y + off, vals):
            ax.text(v + 0.006, yy, f"{v:.2f}", va="center", ha="left", fontsize=8.5,
                    color=xs.INK)
    ax.set_yticks(y, [NAMES[m] for m in BIG4])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, labelsize=10, labelcolor=xs.INK)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, w.max().max() * 1.15)
    ax.xaxis.set_major_formatter(xs.fmt_thousands(2))
    r.round(3).to_csv(c.out / "24_tx_deliveries_per_household.csv", index=False)
    xs.save_exhibit(fig, c.out / "24_tx_deliveries_per_household")
    return w.rename(index=NAMES).round(3)


# ---------------------------------------------------------------------------
# All Texas metros
# ---------------------------------------------------------------------------

def ex_texas_metro_pipeline(c: Ctx, min_inventory=10_000):
    lh = c.last_hist
    metros = c.df[(c.df.geo_type == "Metro") & c.df.geo_name.str.contains(" - TX ")]
    names = metros.drop_duplicates("cbsa_code").set_index("cbsa_code")["market"]
    uc = c.s("Under Construction Units").loc[lh]
    inv = c.s("Inventory Units").loc[lh]
    keep = [m for m in names.index if m in inv.index and inv[m] >= min_inventory]
    pct_all = (100 * uc[keep] / inv[keep]).sort_values(ascending=False)
    pct = pct_all[pct_all > 0]
    none = sorted(names[m] for m in pct_all.index if pct_all[m] == 0)
    top = names[pct.index[0]]
    fig, ax = xs.new_exhibit(
        f"{top} has the deepest apartment pipeline among Texas metros",
        f"Multifamily units under construction as a share of inventory, {c.last_label} "
        f"(metros with {min_inventory // 1000}K+ units)", source=c.source,
        note="Big-four metros in color."
             + (f" Nothing under construction in {', '.join(none)}." if none else ""),
        figsize=(8, 6.8), left=1.6)
    y = np.arange(len(pct))
    ax.barh(y, pct.values, height=0.6,
            color=[xs.METRO_COLORS.get(m, xs.PEER) for m in pct.index],
            edgecolor=xs.SURFACE, lw=0.8)
    for yy, m in zip(y, pct.index):
        ax.text(pct[m] + pct.max() * 0.012, yy, f"{pct[m]:.1f}%  ·  {uc[m]:,.0f} units",
                va="center", ha="left", fontsize=8, color=xs.INK,
                fontweight="bold" if m in BIG4 else "normal")
    ax.set_yticks(y, [names[m] for m in pct.index])
    for lab, m in zip(ax.get_yticklabels(), pct.index):
        if m in BIG4:
            lab.set_fontweight("bold")
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, labelsize=9, labelcolor=xs.INK)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, pct.max() * 1.35)
    ax.xaxis.set_major_formatter(xs.fmt_pct(0))
    pd.DataFrame({"metro": names[pct_all.index].values, "uc_pct_inventory": pct_all.values,
                  "under_construction": uc[pct_all.index].values,
                  "inventory": inv[pct_all.index].values}).round(3) \
        .to_csv(c.out / "25_texas_metros_pipeline.csv", index=False)
    xs.save_exhibit(fig, c.out / "25_texas_metros_pipeline")
    return pct.rename(names).round(2)


# ---------------------------------------------------------------------------
# Submarkets
# ---------------------------------------------------------------------------

def ex_submarket_pipeline(c: Ctx, cbsa: str, num: int, top_n: int = 15):
    lh = c.last_hist
    kw = dict(geo_type="Submarket", by="submarket")
    sub = c.df[(c.df.geo_type == "Submarket") & (c.df.cbsa_code == cbsa)]
    if sub.empty:
        return None
    subs = sub.submarket.unique()
    uc = c.s("Under Construction Units", **kw).loc[lh, subs].fillna(0)
    inv = c.s("Inventory Units", **kw).loc[lh, subs]
    dl = c.s("Net Delivered Units 12 Mo", **kw).loc[lh, subs].fillna(0)
    total = uc.sum()
    top = uc[uc > 0].sort_values(ascending=False).head(top_n)
    share5 = 100 * top.head(5).sum() / total
    name = NAMES.get(cbsa, sub.market.iloc[0])
    short = name.replace("Dallas-Fort Worth", "DFW")
    fig, ax = xs.new_exhibit(
        f"Five submarkets hold {share5:.0f}% of {short}'s apartment pipeline",
        f"Multifamily units under construction by submarket, {c.last_label} — top {len(top)} "
        f"of {len(subs)}; {short} total {total:,.0f} units", source=c.source,
        note="Labels: units under construction · share of the submarket's existing inventory "
             "· units delivered in the past 12 months.", figsize=(8, 6.8), left=2.0)
    y = np.arange(len(top))
    ax.barh(y, top.values, height=0.6, color=xs.METRO_COLORS.get(cbsa, xs.SUPPLY),
            edgecolor=xs.SURFACE, lw=0.8)
    for yy, s in zip(y, top.index):
        ax.text(top[s] + top.max() * 0.012, yy,
                f"{top[s]:,.0f}  ·  {100 * top[s] / inv[s]:.0f}% of stock  ·  "
                f"{dl[s]:,.0f}".replace("-", "\u2212") + " delivered",
                va="center", ha="left", fontsize=8, color=xs.INK)
    ax.set_yticks(y, list(top.index))
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, labelsize=9, labelcolor=xs.INK)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, top.max() * 1.6)
    ax.xaxis.set_major_formatter(xs.FuncFormatter(lambda v, _: xs.fmt_k(v)))
    slug = short.lower().replace(" ", "_")
    pd.DataFrame({"submarket": subs, "under_construction": uc[subs].values,
                  "inventory": inv[subs].values, "delivered_12mo": dl[subs].values}) \
        .sort_values("under_construction", ascending=False) \
        .to_csv(c.out / f"{num}_{slug}_submarket_pipeline.csv", index=False)
    xs.save_exhibit(fig, c.out / f"{num}_{slug}_submarket_pipeline")
    return {"metro": short, "total_uc": total, "top5_share": round(share5, 1),
            "top": list(top.head(5).index)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--brand", default=None, help="Footer credit (default: none)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.brand is not None:
        xs.BRAND = args.brand
    out = args.out or ROOT / "outputs" / "costar_exhibits" / date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)

    c = Ctx(load_costar(), out)
    print(f"CoStar as of {c.as_of}; history through {c.last_hist}; forecast to {c.fc_end}")
    print(ex_pipeline(c).round(0))
    print(ex_uc_share(c))
    print(ex_vacancy(c).round(1))
    print(ex_rent_growth(c).round(1))
    print(ex_deliveries_per_household(c))
    print(ex_texas_metro_pipeline(c).to_string())
    for num, cbsa in [(26, "19100"), (27, "12420"), (28, "41700"), (29, "26420")]:
        print(ex_submarket_pipeline(c, cbsa, num) or f"No submarket data for {NAMES[cbsa]}")
    print(f"Wrote exhibits to {out}")


if __name__ == "__main__":
    main()
