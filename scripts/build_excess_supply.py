"""Estimate excess housing supply: units added vs. households (and renter households).

Three lenses, each a different definition of "excess":

1. Excess vacant apartments (CoStar). Vacant units above what the metro's
   pre-pandemic vacancy rate (2015-19 average) would leave empty:
       excess(t) = inventory(t) x (vacancy(t) - baseline vacancy)
   The same number falls out of the flow identity since 2019 Q4:
       deliveries - absorption - baseline vacancy x deliveries.
   Includes CoStar's Base Case path for when the excess clears.

2. Apartment deliveries vs. renter households (CoStar + ACS), 2021->2024.
   Renter households living in 5+ unit buildings are the closest public match
   to CoStar's multifamily inventory.

3. Excess vacant rental units, all rentals (ACS), 2024 vs. 2019 rental vacancy,
   for the Texas metros, the peer set and the U.S., with 90% error margins.

Writes PNG + SVG + CSV per exhibit and summary.md to
outputs/excess_supply/<YYYY-MM-DD>/. CoStar-based exhibits cite CoStar.

Usage:
    python scripts/build_excess_supply.py
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
from geostack.housing_supply import PEER_METROS, TEXAS_METROS, US, cbsa_tenure_vacancy

ROOT = Path(__file__).resolve().parent.parent
BIG4 = list(TEXAS_METROS)
CMP = [*BIG4, *PEER_METROS]
NAMES = {**TEXAS_METROS, **PEER_METROS, US: "United States"}
SHORT = {**NAMES, "19100": "DFW"}
BASE = ("2015Q1", "2019Q4")
SRC_ACS = "U.S. Census Bureau, American Community Survey 1-year"


def _k(v: float) -> str:
    return xs.fmt_k(v).replace("-", "−")


def _signed(v: float, decimals: int = 1, unit: str = "%") -> str:
    if round(v, decimals) == 0:
        return f"0{unit}"
    return f"{'+' if v > 0 else chr(0x2212)}{abs(v):.{decimals}f}{unit}"


def _ts(idx) -> pd.DatetimeIndex:
    return pd.PeriodIndex(idx, freq="Q").to_timestamp() + pd.Timedelta(days=45)


def _qlabel(p: pd.Period) -> str:
    return f"{p.year} Q{p.quarter}"


# ---------------------------------------------------------------------------
# 1. CoStar: excess vacant apartments
# ---------------------------------------------------------------------------

def costar_excess(df: pd.DataFrame):
    vac, inv = series(df, "Vacancy Rate"), series(df, "Inventory Units")
    dl, ab = series(df, "Net Delivered Units"), series(df, "Absorption Units")
    hist = df[(df.status == "history") & (df.concept == "Inventory Units")].period.max()
    vb = vac.loc[BASE[0]:BASE[1]].mean()
    vb_long = vac.loc["2010Q1":BASE[1]].mean()
    excess = inv * (vac - vb)
    excess_long = inv * (vac - vb_long)
    rows = []
    for m in [c for c in excess.columns if c in vb.dropna().index]:
        e = excess[m].dropna()
        h = e.loc[:hist]
        fc = e.loc[hist + 1:]
        cleared = fc[fc <= 0.05 * h.max()]  # within 5% of peak counts as cleared
        d = dl[m].loc["2020Q1":hist].sum()
        a = ab[m].loc["2020Q1":hist].sum()
        rows.append({
            "cbsa_code": m,
            "inventory": inv.loc[hist, m],
            "vacancy_now": vac.loc[hist, m],
            "vacancy_baseline": vb[m],
            "excess_now": e.loc[hist],
            "excess_now_2010_19_base": excess_long.loc[hist, m],
            "excess_pct_inventory": 100 * e.loc[hist] / inv.loc[hist, m],
            "excess_2019q4": e.loc["2019Q4"],
            "peak_excess": h.loc["2020Q1":].max(),
            "peak_quarter": str(h.loc["2020Q1":].idxmax()),
            "delivered_since_2020": d,
            "absorbed_since_2020": a,
            "normal_vacancy_on_new": vb[m] * d,
            "excess_built_since_2020": d - a - vb[m] * d,
            "excess_end_of_forecast": fc.iloc[-1] if len(fc) else np.nan,
            "clears_by": str(cleared.index[0]) if len(cleared) else None,
        })
    return pd.DataFrame(rows).set_index("cbsa_code"), excess, hist


def ex_excess_timeline(excess, hist, fc_end, summ, out, source):
    e = excess[BIG4].loc["2010Q1":]
    total_now = summ.loc[BIG4, "excess_now"].sum()
    clears = {m: summ.loc[m, "clears_by"] for m in BIG4}
    left = summ.loc[BIG4, "excess_end_of_forecast"].sum()
    title = (f"Texas' big four hold ~{total_now / 1000:.0f}K excess vacant apartments; "
             f"CoStar sees ~{left / 1000:.0f}K left in {fc_end.year}")
    fig, axes = xs.new_exhibit(
        title,
        f"Vacant apartments above each metro's 2015–19 average vacancy rate, 2010 – "
        f"{fc_end.year}, with CoStar's Base Case forecast",
        source=source,
        note="Excess = inventory × (vacancy − 2015–19 average vacancy). Below zero = tighter "
             "than the pre-pandemic norm.",
        nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8, 6.2))
    lim_hi = e.max().max() * 1.18
    lim_lo = min(e.min().min() * 1.15, -lim_hi * 0.15)
    for ax, m in zip(axes.flat, BIG4):
        s = e[m].dropna()
        x = _ts(s.index)
        color = xs.METRO_COLORS[m]
        xs.forecast_band(ax, _ts([hist])[0], x[-1] + pd.Timedelta(days=60),
                         label="CoStar forecast")
        ax.fill_between(x, 0, s.clip(lower=0), color=color, alpha=0.18, lw=0)
        ax.fill_between(x, 0, s.clip(upper=0), color=xs.PEER, alpha=0.25, lw=0)
        ax.plot(x, s, color=color, lw=1.8)
        ax.axhline(0, color=xs.INK_2, lw=0.8)
        now = s.loc[hist]
        ax.plot(_ts([hist])[0], now, "o", ms=6, color=color, mec=xs.SURFACE, mew=1.5,
                zorder=5)
        ax.set_title(NAMES[m], loc="left")
        clear_txt = (f"clears ~{clears[m][:4]}" if isinstance(clears[m], str)
                     else f"{_k(summ.loc[m, 'excess_end_of_forecast'])} in {fc_end.year}")
        ax.text(1, 1.04, f"Now {_k(now)} · peak {_k(summ.loc[m, 'peak_excess'])} · {clear_txt}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=xs.INK_2)
        ax.set_ylim(lim_lo, lim_hi)
        ax.yaxis.set_major_formatter(xs.FuncFormatter(lambda v, _: _k(v)))
    e.rename(columns=NAMES).round(0).to_csv(out / "30_excess_vacant_apartments_timeline.csv")
    xs.save_exhibit(fig, out / "30_excess_vacant_apartments_timeline")


def ex_supply_balance(summ, hist, out, source):
    s = summ.loc[BIG4]
    fig, ax = xs.new_exhibit(
        f"Since 2020, Texas' big four delivered {s.delivered_since_2020.sum() / 1000:.0f}K "
        f"apartments and filled {s.absorbed_since_2020.sum() / 1000:.0f}K",
        f"Multifamily units, 2020 Q1 – {_qlabel(hist)}: net deliveries, absorption (net "
        f"move-ins), and the excess left vacant beyond a normal vacancy rate",
        source=source,
        note="Excess built since 2020 = deliveries − absorption − (2015–19 vacancy rate × "
             "deliveries). Total excess today also includes any excess already present in 2019.",
        legend=[("Delivered", xs.SUPPLY), ("Absorbed", xs.DEMAND),
                ("Excess vacant (built since 2020)", xs.CATEGORICAL[7])], legend_kind="patch",
        figsize=(8, 5.6), left=1.45)
    y = np.arange(len(BIG4))
    h = 0.22
    cols = [("delivered_since_2020", xs.SUPPLY), ("absorbed_since_2020", xs.DEMAND),
            ("excess_built_since_2020", xs.CATEGORICAL[7])]
    for i, (col, color) in enumerate(cols):
        off = (i - 1) * (h + 0.05)
        vals = s[col].values
        ax.barh(y + off, vals, height=h, color=color, edgecolor=xs.SURFACE, lw=0.8)
        for yy, v in zip(y + off, vals):
            ax.text(v + s.delivered_since_2020.max() * 0.012, yy, f"{v:,.0f}", va="center",
                    ha="left", fontsize=8.5, color=xs.INK)
    ax.set_yticks(y, [NAMES[m] for m in BIG4])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, labelsize=10, labelcolor=xs.INK)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, s.delivered_since_2020.max() * 1.15)
    ax.xaxis.set_major_formatter(xs.FuncFormatter(lambda v, _: _k(v)))
    s[[c for c, _ in cols]].rename(index=NAMES).round(0).to_csv(
        out / "31_apartment_supply_balance_since_2020.csv")
    xs.save_exhibit(fig, out / "31_apartment_supply_balance_since_2020")


# ---------------------------------------------------------------------------
# 2. CoStar deliveries vs. ACS renter households, 2021 -> 2024
# ---------------------------------------------------------------------------

def renter_flow(df, acs, summ, y0=2021, y1=2024):
    """ACS 1-year estimates describe a calendar-year average (centred mid-year), so
    CoStar quarters from mid-y0 to mid-y1 (Q3 y0 .. Q2 y1) line up with the change."""
    dl, ab = series(df, "Net Delivered Units"), series(df, "Absorption Units")
    window = (pd.Period(f"{y0}Q3"), pd.Period(f"{y1}Q2"))
    a = acs.set_index(["cbsa_code", "year"])
    rows = []
    for m in BIG4:
        d = dl.loc[window[0]:window[1], m].sum()
        absorbed = ab.loc[window[0]:window[1], m].sum()
        r_all = a.loc[(m, y1), "renter_hh"] - a.loc[(m, y0), "renter_hh"]
        r5 = a.loc[(m, y1), "renter_hh_5plus"] - a.loc[(m, y0), "renter_hh_5plus"]
        r5_moe = np.hypot(a.loc[(m, y1), "renter_hh_5plus_moe"],
                          a.loc[(m, y0), "renter_hh_5plus_moe"])
        vb = summ.loc[m, "vacancy_baseline"]
        rows.append({
            "cbsa_code": m, "metro": NAMES[m],
            "apartments_delivered": d, "apartments_absorbed": absorbed,
            "renter_hh_added": r_all,
            "renter_hh_5plus_added": r5, "renter_hh_5plus_added_moe90": r5_moe,
            "implied_excess_vs_5plus_renters": d - r5 - vb * d,
            "implied_excess_low": d - (r5 + r5_moe) - vb * d,
            "implied_excess_high": d - (r5 - r5_moe) - vb * d,
            "deliveries_per_5plus_renter_hh": d / r5,
        })
    return pd.DataFrame(rows).set_index("cbsa_code"), window


def ex_renter_flow(rf, window, out, source):
    fig, ax = xs.new_exhibit(
        "Apartment deliveries vs. renter household growth, 2021–2024",
        f"CoStar apartments delivered and absorbed ({_qlabel(window[0])} – "
        f"{_qlabel(window[1])}) vs. ACS renter households added (2021\u00a0→\u00a02024)",
        source=source,
        note="Renter households in 5+ unit buildings are the closest public match to CoStar "
             "multifamily; whiskers show the ACS 90% margin of error.",
        legend=[("Apartments delivered (CoStar)", xs.SUPPLY),
                ("Apartments absorbed (CoStar)", xs.DEMAND),
                ("Renters in 5+ unit buildings (ACS)", xs.CATEGORICAL[2]),
                ("All renter households (ACS)", xs.PEER)], legend_kind="patch", legend_ncol=2,
        figsize=(8, 6.2), left=1.45)
    cols = [("apartments_delivered", xs.SUPPLY), ("apartments_absorbed", xs.DEMAND),
            ("renter_hh_5plus_added", xs.CATEGORICAL[2]), ("renter_hh_added", xs.PEER)]
    y = np.arange(len(rf))
    h = 0.18
    span = rf[[c for c, _ in cols]].max().max()
    for i, (col, color) in enumerate(cols):
        off = (i - 1.5) * (h + 0.03)
        vals = rf[col].values
        ax.barh(y + off, vals, height=h, color=color, edgecolor=xs.SURFACE, lw=0.8)
        if col == "renter_hh_5plus_added":
            ax.errorbar(vals, y + off, xerr=rf["renter_hh_5plus_added_moe90"].values,
                        fmt="none", ecolor=xs.INK_2, elinewidth=0.9, capsize=2.5)
        for yy, v, extra in zip(y + off, vals, rf["renter_hh_5plus_added_moe90"].values):
            x = v + (extra if col == "renter_hh_5plus_added" else 0) + span * 0.012
            ax.text(x, yy, f"{v:,.0f}", va="center", ha="left", fontsize=7.5, color=xs.INK)
    ax.set_yticks(y, rf.metro)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, labelsize=10, labelcolor=xs.INK)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, span * 1.18)
    ax.xaxis.set_major_formatter(xs.FuncFormatter(lambda v, _: _k(v)))
    rf.round(0).to_csv(out / "32_deliveries_vs_renter_households.csv")
    xs.save_exhibit(fig, out / "32_deliveries_vs_renter_households")


# ---------------------------------------------------------------------------
# 3. ACS: excess vacant rental units (all rentals), Texas + peers + U.S.
# ---------------------------------------------------------------------------

def acs_rental_excess(acs, y0=2019, y1=2024):
    a = acs.set_index(["cbsa_code", "year"])
    rows = []
    for m in [*CMP, US]:
        b, n = a.loc[(m, y0)], a.loc[(m, y1)]
        excess = n.for_rent - b.rental_vacancy * n.rental_stock
        # MOE: latest for-rent count, plus the baseline rate's uncertainty scaled to today
        moe = np.hypot(n.for_rent_moe, n.rental_stock * b.for_rent_moe / b.rental_stock)
        rows.append({
            "cbsa_code": m, "metro": NAMES[m],
            f"rental_vacancy_{y0}": 100 * b.rental_vacancy,
            f"rental_vacancy_{y1}": 100 * n.rental_vacancy,
            "rental_stock": n.rental_stock,
            "excess_vacant_rentals": excess, "excess_moe90": moe,
            "excess_pct_stock": 100 * excess / n.rental_stock,
            "excess_pct_moe90": 100 * moe / n.rental_stock,
            "housing_units_added_21_24": a.loc[(m, 2024), "housing_units"]
            - a.loc[(m, 2021), "housing_units"],
            "households_added_21_24": a.loc[(m, 2024), "households"]
            - a.loc[(m, 2021), "households"],
            "renter_hh_added_21_24": a.loc[(m, 2024), "renter_hh"] - a.loc[(m, 2021), "renter_hh"],
        })
    return pd.DataFrame(rows).set_index("cbsa_code")


def ex_acs_rental_excess(ar, out, y0=2019, y1=2024):
    v = ar.loc[CMP].sort_values("excess_pct_stock", ascending=False)
    sig_pos = v[(v.excess_pct_stock - v.excess_pct_moe90) > 0].metro.tolist()
    us = ar.loc[US]
    if len(sig_pos) == 1:
        title = f"Across all rentals, only {sig_pos[0]} is clearly emptier than in {y0}"
    elif sig_pos and len(sig_pos) < len(CMP) / 2:
        title = (f"Across all rentals, only {len(sig_pos)} of {len(CMP)} metros are clearly "
                 f"emptier than in {y0}")
    elif sig_pos:
        title = f"{len(sig_pos)} of {len(CMP)} metros have clearly more vacant rentals than in {y0}"
    else:
        title = f"Across all rentals, no metro is clearly emptier than in {y0}"
    fig, ax = xs.new_exhibit(
        title,
        f"Vacant for-rent units above the {y0} rental vacancy rate, {y1}, as a share of all "
        f"rental housing (apartments, single-family rentals and small buildings)",
        source=SRC_ACS,
        note=f"Whiskers: 90% margin of error. U.S. {_signed(us.excess_pct_stock)} "
             f"({_k(us.excess_vacant_rentals)} units). Texas metros in color; peers in gray. "
             f"ACS counts newly built units in lease-up with a lag.",
        figsize=(8, 6.4), left=1.45)
    y = np.arange(len(v))
    colors = [xs.METRO_COLORS.get(m, xs.PEER) for m in v.index]
    ax.barh(y, v.excess_pct_stock, height=0.56, color=colors, edgecolor=xs.SURFACE, lw=0.8)
    ax.errorbar(v.excess_pct_stock, y, xerr=v.excess_pct_moe90, fmt="none", ecolor=xs.INK_2,
                elinewidth=0.9, capsize=2.5)
    span = (v.excess_pct_stock.abs() + v.excess_pct_moe90).max()
    for yy, (m, r) in zip(y, v.iterrows()):
        right = r.excess_pct_stock >= 0
        x = r.excess_pct_stock + (1 if right else -1) * (r.excess_pct_moe90 + span * 0.02)
        ax.text(x, yy, f"{_signed(r.excess_pct_stock)} · {_k(r.excess_vacant_rentals)}",
                va="center", ha="left" if right else "right", fontsize=8, color=xs.INK,
                fontweight="bold" if m in BIG4 else "normal")
    ax.axvline(0, color=xs.INK_2, lw=1)
    ax.set_yticks(y, v.metro)
    for lab, m in zip(ax.get_yticklabels(), v.index):
        if m in BIG4:
            lab.set_fontweight("bold")
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, labelsize=10, labelcolor=xs.INK)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(-span * 1.45, span * 1.45)
    ax.xaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10], integer=True))
    ax.xaxis.set_major_formatter(xs.FuncFormatter(lambda val, _: _signed(val, 0)))
    ar.round(3).to_csv(out / "33_acs_excess_vacant_rentals.csv")
    xs.save_exhibit(fig, out / "33_acs_excess_vacant_rentals")
    return sig_pos


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(summ, rf, ar, hist, fc_end, as_of_q, window, out):
    s = summ.loc[BIG4]
    lines = [
        "# Excess housing supply estimate",
        "",
        f"CoStar data as of {_qlabel(as_of_q)} (history through {_qlabel(hist)}, Base Case "
        f"forecast to {_qlabel(fc_end)}); ACS 1-year 2019 and 2021–2024.",
        "",
        "## 1. Excess vacant apartments (CoStar)",
        "",
        "Vacant units above each metro's 2015–19 average vacancy rate.",
        "",
        "| Metro | Inventory | Vacancy now | 2015–19 avg | Excess vacant now | % of inventory "
        "| Peak (quarter) | Delivered since 2020 | Absorbed since 2020 | Clears (CoStar BC) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m in BIG4:
        r = summ.loc[m]
        clears = (r.clears_by[:4] if isinstance(r.clears_by, str)
                  else f"not by {fc_end.year} ({_k(r.excess_end_of_forecast)} left)")
        lines.append(
            f"| {NAMES[m]} | {r.inventory:,.0f} | {100 * r.vacancy_now:.1f}% | "
            f"{100 * r.vacancy_baseline:.1f}% | **{r.excess_now:,.0f}** | "
            f"{r.excess_pct_inventory:.1f}% | {r.peak_excess:,.0f} ({r.peak_quarter}) | "
            f"{r.delivered_since_2020:,.0f} | {r.absorbed_since_2020:,.0f} | {clears} |")
    lines += [
        f"| **Big four total** | {s.inventory.sum():,.0f} | | | **{s.excess_now.sum():,.0f}** | "
        f"{100 * s.excess_now.sum() / s.inventory.sum():.1f}% | | "
        f"{s.delivered_since_2020.sum():,.0f} | {s.absorbed_since_2020.sum():,.0f} | |",
        "",
        f"Using a 2010–19 baseline instead: {s.excess_now_2010_19_base.sum():,.0f} excess units "
        f"for the big four.",
        "",
        "## 2. Apartment deliveries vs. renter households (CoStar + ACS)",
        "",
        f"CoStar {_qlabel(window[0])} – {_qlabel(window[1])} vs. ACS 2021 → 2024.",
        "",
        "| Metro | Apartments delivered | Absorbed | Renter HH added (all) | "
        "Renter HH added, 5+ unit bldgs (±90% MOE) | Implied excess vs. 5+ renters |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m, r in rf.iterrows():
        lines.append(
            f"| {r.metro} | {r.apartments_delivered:,.0f} | {r.apartments_absorbed:,.0f} | "
            f"{r.renter_hh_added:,.0f} | {r.renter_hh_5plus_added:,.0f} "
            f"(±{r.renter_hh_5plus_added_moe90:,.0f}) | {r.implied_excess_vs_5plus_renters:,.0f} "
            f"({r.implied_excess_low:,.0f} to {r.implied_excess_high:,.0f}) |")
    lines += [
        "",
        "## 3. All rentals (ACS): vacant for-rent units above the 2019 rate, 2024",
        "",
        "| Metro | Rental vacancy 2019 | 2024 | Excess vacant rentals (±90% MOE) "
        "| % of rental stock | Housing units added 21→24 | Households added 21→24 "
        "| Renter HH added 21→24 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m, r in ar.iterrows():
        lines.append(
            f"| {r.metro} | {r.rental_vacancy_2019:.1f}% | {r.rental_vacancy_2024:.1f}% | "
            f"{r.excess_vacant_rentals:,.0f} (±{r.excess_moe90:,.0f}) | "
            f"{r.excess_pct_stock:+.1f}% | {r.housing_units_added_21_24:,.0f} | "
            f"{r.households_added_21_24:,.0f} | {r.renter_hh_added_21_24:,.0f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--brand", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.brand is not None:
        xs.BRAND = args.brand
    out = args.out or ROOT / "outputs" / "excess_supply" / date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)

    df = load_costar()
    as_of_q = as_of(df)
    fc_end = df[df.status == "forecast"].period.max()
    src_costar = (f"CoStar, multifamily, as of {_qlabel(as_of_q)} (forecasts are CoStar's "
                  f"Base Case)")
    summ, excess, hist = costar_excess(df)
    summ.assign(metro=lambda x: x.index.map(lambda c: SHORT.get(c, c))).round(4) \
        .to_csv(out / "costar_excess_by_metro.csv")
    acs = cbsa_tenure_vacancy(CMP, [2019, 2021, 2022, 2023, 2024])
    acs.to_csv(out / "acs_tenure_vacancy.csv", index=False)

    ex_excess_timeline(excess, hist, fc_end, summ, out, src_costar)
    ex_supply_balance(summ, hist, out, src_costar)
    rf, window = renter_flow(df, acs, summ)
    ex_renter_flow(rf, window, out, f"CoStar; {SRC_ACS}")
    ar = acs_rental_excess(acs)
    ex_acs_rental_excess(ar, out)
    write_summary(summ, rf, ar, hist, fc_end, as_of_q, window, out)

    pd.set_option("display.width", 220)
    print(summ.loc[BIG4, ["excess_now", "excess_pct_inventory", "peak_excess", "peak_quarter",
                          "delivered_since_2020", "absorbed_since_2020",
                          "excess_built_since_2020", "clears_by",
                          "excess_now_2010_19_base"]].round(1))
    print(rf.round(0).to_string())
    print(ar[["metro", "rental_vacancy_2019", "rental_vacancy_2024", "excess_vacant_rentals",
              "excess_moe90", "excess_pct_stock", "housing_units_added_21_24",
              "households_added_21_24"]].round(1).to_string())
    print(f"Wrote to {out}")


if __name__ == "__main__":
    main()
