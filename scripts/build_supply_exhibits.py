"""Build publishable housing supply vs. demand exhibits for Texas metros.

Public, keyless data only (Census BPS / PEP / ACS, FRED). Writes PNG + SVG per
exhibit, the CSV behind each chart, and the full metro panel to:

    outputs/supply_exhibits/<YYYY-MM-DD>/

Headlines are computed from the data (with neutral fallbacks), so a re-run on
newer data never publishes a stale claim. Review them before posting anyway.

Usage:
    python scripts/build_supply_exhibits.py
    python scripts/build_supply_exhibits.py --brand "Your Name"
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

from geostack import exhibit_style as xs
from geostack.housing_supply import (
    PEER_METROS,
    TEXAS_METROS,
    US,
    cbsa_housing_units_2020,
    cbsa_permits_ytd,
    construction_monthly,
    metro_panel,
)

ROOT = Path(__file__).resolve().parent.parent
NAMES = {**TEXAS_METROS, **PEER_METROS, US: "United States"}
ORDER = list(TEXAS_METROS)  # DFW, Houston, Austin, San Antonio
PEERS = list(PEER_METROS)
ALL = [*ORDER, US]
CMP = [*ORDER, *PEERS]  # Texas + peer comparison set

SRC_BPS = ("U.S. Census Bureau, Building Permits Survey "
           "(counties aggregated to 2023 metro definitions)")
SRC_PEP = "Census Population Estimates"
SRC_ACS = "Census American Community Survey 1-year"
SRC_NRC = "U.S. Census Bureau & HUD, New Residential Construction, via FRED"
SRC_CES = "BLS Current Employment Statistics, via FRED"

MONTHS = ["", "January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def _signed(v: float, decimals: int = 0) -> str:
    """+12% / −3% / 0% with a true minus sign."""
    if round(v, decimals) == 0:
        return "0%"
    sign = "+" if v > 0 else "\u2212"
    return f"{sign}{abs(v):.{decimals}f}%"


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _style_category_axis(ax, cats):
    y = np.arange(len(cats))
    ax.set_yticks(y, cats)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.tick_params(axis="y", length=0, labelsize=10, labelcolor=xs.INK)
    return y


def _hbar_group(ax, cats, series, height=0.2, label_fmt=lambda v: f"{v:.1f}%"):
    """Grouped horizontal bars. series: list of (values, color)."""
    n = len(series)
    y = _style_category_axis(ax, cats)
    span = max(abs(v) for vals, _ in series for v in vals)
    label_bg = dict(boxstyle="square,pad=0.1", fc=xs.SURFACE, ec="none")
    for i, (vals, color) in enumerate(series):
        off = (i - (n - 1) / 2) * (height + 0.05)
        ax.barh(y + off, vals, height=height, color=color, edgecolor=xs.SURFACE, lw=0.8)
        for yy, v in zip(y + off, vals):
            ax.text(v + (1 if v >= 0 else -1) * span * 0.015, yy, label_fmt(v), va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.5, color=xs.INK,
                    bbox=label_bg)
    ax.axvline(0, color=xs.BASELINE, lw=1)


# ---------------------------------------------------------------------------
# Exhibits
# ---------------------------------------------------------------------------

def ex_us_multifamily(cons: pd.DataFrame, out: Path):
    d = cons[["starts_5p_us", "completions_5p_us"]].loc["2000":].rolling(3).mean().dropna()
    last = d.index[-1]
    s_now, c_now = d["starts_5p_us"].iloc[-1], d["completions_5p_us"].iloc[-1]
    comp_peak = d.loc["2020":, "completions_5p_us"].idxmax()
    start_peak = d.loc["2020":, "starts_5p_us"].idxmax()
    comp_off = 100 * (1 - c_now / d.loc[comp_peak, "completions_5p_us"])
    if comp_off >= 20 and s_now > c_now:
        title = (f"Multifamily completions are {comp_off:.0f}% off their {comp_peak:%Y} peak; "
                 f"starts now run higher")
    elif comp_off >= 20:
        title = f"Multifamily completions are down {comp_off:.0f}% from their {comp_peak:%Y} peak"
    else:
        title = "U.S. multifamily starts vs. completions"
    fig, ax = xs.new_exhibit(
        title,
        f"U.S. units in 5+ unit buildings, thousands, SAAR (3-month average), "
        f"Jan 2000 – {last:%b %Y}",
        source=SRC_NRC, legend=[("Starts", xs.SUPPLY), ("Completions", xs.DEMAND)],
        right=1.55)
    ax.plot(d.index, d["starts_5p_us"], color=xs.SUPPLY)
    ax.plot(d.index, d["completions_5p_us"], color=xs.DEMAND)
    ax.set_ylim(0, d.max().max() * 1.15)
    ax.set_xlim(d.index[0], last + pd.Timedelta(days=60))
    ax.yaxis.set_major_formatter(xs.fmt_thousands())
    xs.end_labels(ax, [(last, s_now, f"Starts {s_now:,.0f}K", xs.SUPPLY),
                       (last, c_now, f"Completions {c_now:,.0f}K", xs.DEMAND)],
                  x_pad_frac=0.006)
    peaks = [(start_peak, "starts_5p_us", f"Starts peak\n{start_peak:%b %Y}"),
             (comp_peak, "completions_5p_us", f"Completions peak\n{comp_peak:%b %Y}")]
    for when, col, txt in peaks:
        ax.annotate(txt, (when, d.loc[when, col]), xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=xs.INK_2)
    d.to_csv(out / "01_us_multifamily_starts_completions.csv")
    xs.save_exhibit(fig, out / "01_us_multifamily_starts_completions")
    return {"mf_completions_off_peak_pct": round(comp_off, 1), "mf_starts_now": s_now,
            "mf_completions_now": c_now}


def ex_regions(cons: pd.DataFrame, out: Path):
    regions = [("south", "South"), ("west", "West"), ("midwest", "Midwest"),
               ("northeast", "Northeast")]
    d = cons.loc["2000":].rolling(3).mean().dropna(how="all")
    last = d.index[-1]
    latest = {k: d[f"starts_total_{k}"].iloc[-1] for k, _ in regions}
    others = sum(v for k, v in latest.items() if k != "south")
    if latest["south"] > others:
        title = "The South is starting more homes than the rest of the U.S. combined"
    else:
        title = "Housing starts vs. completions by Census region"
    fig, axes = xs.new_exhibit(
        title,
        f"New housing units, all structure types, thousands SAAR (3-month average), "
        f"Jan 2000 – {last:%b %Y}. Panels have different scales.",
        source=SRC_NRC, legend=[("Starts", xs.SUPPLY), ("Completions", xs.DEMAND)],
        nrows=2, ncols=2, sharex=True, figsize=(8, 5.8))
    rows = []
    for ax, (key, name) in zip(axes.flat, regions):
        s, c = d[f"starts_total_{key}"], d[f"completions_total_{key}"]
        ax.plot(s.index, s, color=xs.SUPPLY, lw=1.6)
        ax.plot(c.index, c, color=xs.DEMAND, lw=1.6)
        ax.set_ylim(0, max(s.max(), c.max()) * 1.12)
        ax.set_title(name, loc="left")
        ax.text(1, 1.04, f"Latest: {s.iloc[-1]:,.0f}K starts · {c.iloc[-1]:,.0f}K completions",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=xs.INK_2)
        ax.yaxis.set_major_formatter(xs.fmt_thousands())
        rows.append({"region": name, "starts_latest": s.iloc[-1],
                     "completions_latest": c.iloc[-1]})
    d.filter(regex="_(south|west|midwest|northeast)$").to_csv(
        out / "02_regional_starts_completions.csv")
    xs.save_exhibit(fig, out / "02_regional_starts_completions")
    return pd.DataFrame(rows)


def ex_permits_by_type(panel: pd.DataFrame, out: Path):
    p = panel[panel.cbsa_code.isin(ORDER)]
    last = int(p.year.max())
    mf_peak = {}
    for code in ORDER:
        m = p[(p.cbsa_code == code) & (p.year >= 2020)].set_index("year")["units_2p"]
        mf_peak[code] = (int(m.idxmax()), 100 * (m.loc[last] / m.max() - 1))
    down = [c for c in ORDER if mf_peak[c][1] <= -15]
    if len(down) == len(ORDER):
        title = "Apartment permitting is down from its post-2020 peak in all four big Texas metros"
    elif down:
        title = f"Apartment permitting has pulled back in {_join([NAMES[c] for c in down])}"
    else:
        title = "Housing permits by structure type, Texas' largest metros"
    fig, axes = xs.new_exhibit(
        title, f"Housing units authorized by building permits, {int(p.year.min())}–{last}",
        source=SRC_BPS, legend=[("1-unit (single-family)", xs.SUPPLY),
                                ("2+ units (multifamily)", xs.DEMAND)], legend_kind="patch",
        nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8, 5.8))
    for ax, code in zip(axes.flat, ORDER):
        m = p[p.cbsa_code == code]
        ax.bar(m.year, m.units_1u, width=0.78, color=xs.SUPPLY, edgecolor=xs.SURFACE, lw=0.6)
        ax.bar(m.year, m.units_2p, width=0.78, bottom=m.units_1u, color=xs.DEMAND,
               edgecolor=xs.SURFACE, lw=0.6)
        ax.set_title(NAMES[code], loc="left")
        peak_year, chg = mf_peak[code]
        ax.text(1, 1.04, f"Multifamily {_signed(chg)} vs. {peak_year} peak",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=xs.INK_2)
        ax.yaxis.set_major_formatter(xs.FuncFormatter(lambda v, _: xs.fmt_k(v)))
        ax.set_xlim(p.year.min() - 0.8, last + 0.8)
    p[["cbsa_code", "year", "units_1u", "units_2u", "units_3_4u", "units_5p", "units_total"]] \
        .assign(metro=lambda x: x.cbsa_code.map(NAMES)) \
        .to_csv(out / "03_tx_permits_by_type.csv", index=False)
    xs.save_exhibit(fig, out / "03_tx_permits_by_type")


def ex_permits_per_capita(panel: pd.DataFrame, out: Path):
    p = panel.assign(per_1k=1000 * panel.units_total / panel.population)
    w = p.pivot(index="year", columns="cbsa_code", values="per_1k")
    last = int(w.index.max())
    top = w.loc[last, ORDER].idxmax()
    above_us = [c for c in ORDER if w.loc[last, c] > w.loc[last, US]]
    if len(above_us) >= 3 and w.loc[last, above_us].min() > 1.8 * w.loc[last, US]:
        n = {3: "Three", 4: "All four"}[len(above_us)]
        title = f"{n} of Texas' big four permit housing at about twice the U.S. per-capita rate"
    else:
        title = f"{NAMES[top]} permits the most housing per resident of Texas' big four"
    fig, ax = xs.new_exhibit(
        title, f"Housing units permitted per 1,000 residents, {int(w.index.min())}–{last}",
        source=f"{SRC_BPS}; {SRC_PEP}",
        legend=[(NAMES[c], xs.METRO_COLORS[c]) for c in ALL], right=1.75)
    ax.plot(w.index, w[US], color=xs.BENCHMARK, lw=1.8)
    for code in ORDER:
        ax.plot(w.index, w[code], color=xs.METRO_COLORS[code])
    ax.set_ylim(0, w[ALL].max().max() * 1.08)
    ax.set_xlim(w.index.min(), last + 0.2)
    ax.set_xticks(range(2000, last + 1, 5))
    xs.end_labels(ax, [(last, w.loc[last, c], f"{NAMES[c]}  {w.loc[last, c]:.1f}",
                        xs.METRO_COLORS[c]) for c in ALL], x_pad_frac=0.008)
    w[ALL].rename(columns=NAMES).round(2).to_csv(out / "04_permits_per_1000_residents.csv")
    xs.save_exhibit(fig, out / "04_permits_per_1000_residents")
    return w.loc[last]


def ex_growth_rates(panel: pd.DataFrame, hu: pd.DataFrame, out: Path, base=2019, y1=2025):
    """Average annual growth since a pre-pandemic base, so the 2020 job dip and
    rebound don't distort the jobs bar. Population growth sums within-vintage PEP
    changes so the 2020 Census re-benchmark doesn't count as growth."""
    n = y1 - base
    rows = []
    for code in ALL:
        m = panel[panel.cbsa_code == code].set_index("year")
        stock = hu.set_index("cbsa_code").loc[code, "housing_units_2020"]
        pop_gain = m.loc[base + 1:y1, "pop_change"].sum()
        rows.append({
            "cbsa_code": code, "metro": NAMES[code],
            "supply_pct": 100 * m.loc[base + 1:y1, "units_total"].mean() / stock,
            "pop_pct": 100 * ((1 + pop_gain / m.loc[base, "population"]) ** (1 / n) - 1),
            "jobs_pct": 100 * ((m.loc[y1, "jobs_k"] / m.loc[base, "jobs_k"]) ** (1 / n) - 1),
        })
    g = pd.DataFrame(rows)
    tx = g[g.cbsa_code.isin(ORDER)]
    # require a clear margin before calling supply "faster" than population
    clear = tx.supply_pct > tx.pop_pct + 0.25
    faster = tx[clear].metro.tolist()
    if len(faster) == len(ORDER):
        title = "Housing supply has outgrown population in all four big Texas metros"
    elif len(faster) == len(ORDER) - 1:
        laggard = tx[~clear].metro.iloc[0]
        title = f"Housing supply has outgrown population in Texas' big metros, except {laggard}"
    else:
        title = "Housing supply vs. population and job growth, Texas' largest metros"
    fig, ax = xs.new_exhibit(
        title,
        f"Average annual growth, {base}–{y1}: units permitted as a share of the 2020 housing "
        f"stock, compared with population and payroll jobs",
        source=f"{SRC_BPS}; {SRC_PEP}; {SRC_CES}",
        note="Permits approximate gross additions; net stock growth is lower after "
             "demolitions and permits that are never built.",
        legend=[("Housing supply (permits ÷ 2020 stock)", xs.SUPPLY),
                ("Population", xs.DEMAND), ("Payroll jobs", xs.JOBS)], legend_kind="patch",
        figsize=(8, 5.6), left=1.45)
    _hbar_group(ax, g.metro.tolist(), [(g.supply_pct.tolist(), xs.SUPPLY),
                                       (g.pop_pct.tolist(), xs.DEMAND),
                                       (g.jobs_pct.tolist(), xs.JOBS)], height=0.22)
    ax.xaxis.set_major_formatter(xs.fmt_pct(1))
    ax.set_xlim(0, max(g[["supply_pct", "pop_pct", "jobs_pct"]].max()) * 1.12)
    g.round(3).to_csv(out / "05_supply_vs_population_jobs_growth.csv", index=False)
    xs.save_exhibit(fig, out / "05_supply_vs_population_jobs_growth")
    return g


def ex_permits_per_household(panel: pd.DataFrame, out: Path, h0=2021, h1=2024):
    rows = []
    for code in ALL:
        m = panel[panel.cbsa_code == code].set_index("year")
        d_hh = m.loc[h1, "households"] - m.loc[h0, "households"]
        permits = m.loc[h0:h1 - 1, "units_total"].sum()
        rows.append({"cbsa_code": code, "metro": NAMES[code], "households_added": d_hh,
                     "units_permitted": permits, "ratio": permits / d_hh})
    r = pd.DataFrame(rows)
    tx = r[r.cbsa_code.isin(ORDER)]
    if (tx.ratio > 1).all() and r.set_index("cbsa_code").loc[US, "ratio"] < 1:
        title = "Texas permitted more than one home per new household; the U.S. did not"
    elif (tx.ratio > 1).all():
        title = "Texas metros permitted more than one home per new household"
    else:
        title = "Homes permitted per new household, Texas' largest metros"
    fig, ax = xs.new_exhibit(
        title,
        f"Units permitted {h0}–{str(h1 - 1)[2:]} per household added {h0}→{str(h1)[2:]} "
        f"(permits lagged a year for construction)",
        source=f"{SRC_BPS}; {SRC_ACS}",
        note="Household counts are ACS 1-year survey estimates and carry sampling error. "
             "1.0 = one permitted unit per new household; some excess is needed to replace "
             "demolished units and keep normal vacancy.", figsize=(8, 5.2), left=1.45)
    y = _style_category_axis(ax, r.metro.tolist())
    ax.barh(y, r.ratio, height=0.42, color=[xs.METRO_COLORS[c] for c in r.cbsa_code],
            edgecolor=xs.SURFACE, lw=0.8, zorder=2)
    ax.axvline(1, color=xs.INK_2, lw=1, zorder=1)
    bg = dict(boxstyle="square,pad=0.15", fc=xs.SURFACE, ec="none")
    for yy, row in zip(y, r.itertuples()):
        ax.text(row.ratio + 0.03, yy, f"{row.ratio:.2f}", va="center", ha="left", fontsize=10.5,
                fontweight="bold", color=xs.INK, bbox=bg, zorder=3)
        ax.text(row.ratio + 0.2, yy, f"{row.units_permitted:,.0f} units ÷ "
                f"{row.households_added:,.0f} households", va="center", ha="left",
                fontsize=8, color=xs.INK_2, zorder=3)
    ax.text(1, -0.72, "1 unit per household", ha="center", va="center", fontsize=8,
            color=xs.INK_2, bbox=bg)
    ax.set_xlim(0, r.ratio.max() * 1.85)
    ax.set_ylim(len(r) - 0.45, -1.0)
    ax.xaxis.set_major_formatter(xs.fmt_thousands(1))
    r.round(3).to_csv(out / "06_permits_per_household_added.csv", index=False)
    xs.save_exhibit(fig, out / "06_permits_per_household_added")
    return r


def ex_ytd(ytd: pd.DataFrame, out: Path):
    month = int(ytd.through_month.iloc[0])
    y_cur = int(ytd.year.max())
    w = ytd.set_index(["cbsa_code", "year"])

    def pct(code, col):
        return 100 * (w.loc[(code, y_cur), col] / w.loc[(code, y_cur - 1), col] - 1)

    chg = pd.DataFrame({
        "metro": [NAMES[c] for c in ALL],
        "total_pct": [pct(c, "units_total") for c in ALL],
        "single_family_pct": [pct(c, "units_1u") for c in ALL],
        "multifamily_pct": [pct(c, "units_2p") for c in ALL],
    })
    n_down = int((chg.total_pct.iloc[:len(ORDER)] < 0).sum())
    if n_down == len(ORDER):
        title = f"{y_cur} permitting is running below {y_cur - 1} in all four big Texas metros"
    elif n_down >= len(ORDER) - 1:
        title = f"{y_cur} permitting is running below {y_cur - 1} in {n_down} of 4 big Texas metros"
    else:
        title = f"{y_cur} year-to-date permitting vs. {y_cur - 1}, Texas' largest metros"
    fig, ax = xs.new_exhibit(
        title, f"Change in units permitted, January–{MONTHS[month]} {y_cur} vs. the same "
        f"months of {y_cur - 1}", source=SRC_BPS,
        note="San Antonio's multifamily change is off a very small base.",
        legend=[("1-unit (single-family)", xs.SUPPLY), ("2+ units (multifamily)", xs.DEMAND)],
        legend_kind="patch", figsize=(8, 5.4), left=1.45)

    fmt = _signed

    _hbar_group(ax, chg.metro.tolist(), [(chg.single_family_pct.tolist(), xs.SUPPLY),
                                         (chg.multifamily_pct.tolist(), xs.DEMAND)],
                height=0.3, label_fmt=fmt)
    lim = max(abs(chg[["single_family_pct", "multifamily_pct"]]).max().max() * 1.25, 10)
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_formatter(xs.FuncFormatter(lambda v, _: fmt(v)))
    ytd.assign(metro=lambda x: x.cbsa_code.map(NAMES)).to_csv(
        out / "07_ytd_permits_change.csv", index=False)
    xs.save_exhibit(fig, out / "07_ytd_permits_change")
    return chg


def ex_scorecard(panel, ytd, ratio, out: Path):
    last = int(panel.year.max())
    month = int(ytd.through_month.iloc[0])
    rows = []
    for code in ALL:
        m = panel[panel.cbsa_code == code].set_index("year")
        y = ytd[ytd.cbsa_code == code].set_index("year")["units_total"]
        rr = ratio.set_index("cbsa_code").loc[code]
        rows.append([
            NAMES[code],
            f"{m.loc[last, 'units_total']:,.0f}",
            _signed(100 * (m.loc[last, 'units_total'] / m.loc[last - 1, 'units_total'] - 1)),
            _signed(100 * (y.loc[last + 1] / y.loc[last] - 1)),
            f"{1000 * m.loc[last, 'units_total'] / m.loc[last, 'population']:.1f}",
            f"{100 * m.loc[last, 'pop_change'] / m.loc[last - 1, 'population']:.1f}%",
            f"{100 * m.loc[last, 'job_change_k'] / m.loc[last - 1, 'jobs_k']:.1f}%",
            f"{m.loc[last, 'units_total'] / (1000 * m.loc[last, 'job_change_k']):.1f}",
            f"{rr.ratio:.2f}",
        ])
    cols = [("Metro", ""), (f"{last} permits", "units"), ("Change", f"vs. {last - 1}"),
            (f"{last + 1} YTD", f"Jan–{MONTHS[month][:3]} vs. {last}"),
            ("Permits", "per 1,000 res."), ("Population", f"growth {last}"),
            ("Jobs", f"growth {last}"), ("Permits", "per job added"),
            ("Permits", "per new HH*")]
    fig, ax = xs.new_exhibit(
        "Texas housing supply scorecard",
        f"Permits, population and payroll jobs in {last}, plus {last + 1} year-to-date "
        f"permitting",
        source=f"{SRC_BPS}; {SRC_PEP}; {SRC_CES}; {SRC_ACS}",
        note="*Units permitted 2021–2023 per household added 2021→2024. Jobs are "
             "annual-average total nonfarm payrolls.", figsize=(10, 4.3), left=0.62,
        panel_titles=False)
    ax.axis("off")
    right_edges = [None, 0.29, 0.385, 0.50, 0.60, 0.70, 0.785, 0.885, 1.0]
    for x, (h1, h2) in zip(right_edges, cols):
        xx, ha = (0.0, "left") if x is None else (x, "right")
        ax.text(xx, 1.0, h1, ha=ha, va="top", fontsize=9, fontweight="bold", color=xs.INK,
                transform=ax.transAxes)
        ax.text(xx, 0.91, h2, ha=ha, va="top", fontsize=8, color=xs.MUTED,
                transform=ax.transAxes)
    ax.plot([0, 1], [0.79, 0.79], color=xs.BASELINE, lw=1, transform=ax.transAxes)
    for i, row in enumerate(rows):
        yy = 0.67 - i * 0.165
        if code_is_us := (i == len(rows) - 1):
            ax.plot([0, 1], [yy + 0.083, yy + 0.083], color=xs.GRID, lw=0.8,
                    transform=ax.transAxes)
        ax.plot([0.006], [yy], "s", ms=7, color=xs.METRO_COLORS[ALL[i]], transform=ax.transAxes,
                clip_on=False)
        for j, (x, val) in enumerate(zip(right_edges, row)):
            ax.text(0.025 if x is None else x, yy, val, ha="left" if x is None else "right",
                    va="center", fontsize=10.5, color=xs.INK_2 if code_is_us else xs.INK,
                    fontweight="bold" if j == 0 else "normal", transform=ax.transAxes)
    pd.DataFrame(rows, columns=[f"{a} {b}".strip() for a, b in cols]).to_csv(
        out / "08_scorecard.csv", index=False)
    xs.save_exhibit(fig, out / "08_scorecard")


# ---------------------------------------------------------------------------
# Peer comparisons (Texas vs. fast-growing Sun Belt / Mountain West metros)
# ---------------------------------------------------------------------------

PEER_NOTE = "Texas metros in color; peer metros in gray."


def _ranked_bars(ax, values: pd.Series, fmt, ref=None, ref_label=None, height=0.56):
    """Horizontal bars ranked high -> low. Texas metros keep their entity color,
    peers recede to gray; `ref` draws a labeled vertical reference (e.g. U.S.)."""
    v = values.sort_values(ascending=False)
    y = _style_category_axis(ax, [NAMES[c] for c in v.index])
    colors = [xs.METRO_COLORS.get(c, xs.PEER) for c in v.index]
    ax.barh(y, v.values, height=height, color=colors, edgecolor=xs.SURFACE, lw=0.8, zorder=2)
    span = max(abs(v.max()), abs(v.min()))
    bg = dict(boxstyle="square,pad=0.1", fc=xs.SURFACE, ec="none")
    for yy, (code, val) in zip(y, v.items()):
        ax.text(val + (1 if val >= 0 else -1) * span * 0.015, yy, fmt(val), va="center",
                ha="left" if val >= 0 else "right", fontsize=9, color=xs.INK, bbox=bg,
                zorder=3, fontweight="bold" if code in ORDER else "normal")
    for lab, code in zip(ax.get_yticklabels(), v.index):
        if code in ORDER:
            lab.set_fontweight("bold")
    if ref is not None:
        ax.axvline(ref, color=xs.INK_2, lw=1, zorder=1)
        ax.text(ref, -0.75, ref_label, ha="center", va="center", fontsize=8, color=xs.INK_2,
                bbox=bg)
        ax.set_ylim(len(v) - 0.5, -1.05)
    return v


def ex_peer_permits_per_capita(panel, out: Path):
    last = int(panel.year.max())
    p = panel[panel.year == last].set_index("cbsa_code")
    per1k = 1000 * p.units_total / p.population
    us = per1k[US]
    v = per1k[CMP]
    tx_rank = [list(v.sort_values(ascending=False).index).index(c) + 1 for c in ORDER]
    top = NAMES[v.idxmax()]
    title = f"{top} permits the most housing per resident of {len(CMP)} fast-growing metros"
    fig, ax = xs.new_exhibit(
        title, f"Housing units permitted per 1,000 residents, {last}",
        source=f"{SRC_BPS}; {SRC_PEP}", note=PEER_NOTE, figsize=(8, 6.2), left=1.45)
    _ranked_bars(ax, v, lambda x: f"{x:.1f}", ref=us, ref_label=f"U.S. {us:.1f}")
    ax.set_xlim(0, v.max() * 1.12)
    ax.xaxis.set_major_formatter(xs.fmt_thousands())
    per1k[[*CMP, US]].rename(NAMES).round(2).to_csv(out / "09_peers_permits_per_1000.csv")
    xs.save_exhibit(fig, out / "09_peers_permits_per_1000")
    return dict(zip([NAMES[c] for c in ORDER], tx_rank))


def ex_peer_permits_per_household(panel, out: Path, h0=2021, h1=2024):
    m = panel.set_index(["cbsa_code", "year"])
    ratio = pd.Series({c: m.loc[(c, slice(h0, h1 - 1)), "units_total"].sum()
                       / (m.loc[(c, h1), "households"] - m.loc[(c, h0), "households"])
                       for c in [*CMP, US]})
    under = [NAMES[c] for c in CMP if ratio[c] < 1]
    if under:
        title = f"Only {_join(under)} permitted less than one home per new household"
        if len(under) > 3:
            title = "Homes permitted per new household, Texas vs. peer metros"
    else:
        title = "Every Texas and peer metro permitted more than one home per new household"
    fig, ax = xs.new_exhibit(
        title,
        f"Units permitted {h0}–{str(h1 - 1)[2:]} per household added {h0}→{str(h1)[2:]} "
        f"(permits lagged a year for construction)",
        source=f"{SRC_BPS}; {SRC_ACS}",
        note=f"{PEER_NOTE} Household counts are ACS 1-year survey estimates and carry "
             f"sampling error. U.S. = {ratio[US]:.2f}.", figsize=(8, 6.2), left=1.45)
    _ranked_bars(ax, ratio[CMP], lambda x: f"{x:.2f}", ref=1.0, ref_label="1 unit per household")
    ax.set_xlim(0, ratio[CMP].max() * 1.12)
    ax.xaxis.set_major_formatter(xs.fmt_thousands(1))
    ratio.rename(NAMES).round(3).to_csv(out / "10_peers_permits_per_household.csv",
                                        header=["units_per_household"])
    xs.save_exhibit(fig, out / "10_peers_permits_per_household")
    return ratio


def _place_point_labels(fig, ax, pts):
    """Greedy, collision-aware labels for a scatter. pts: list of (x, y, text, bold)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # seed with every marker's footprint so no label covers another point
    placed = []
    for x, y, *_ in pts:
        px, py = ax.transData.transform((x, y))
        placed.append(Bbox.from_extents(px - 7, py - 7, px + 7, py + 7))
    offsets = [(7, 0, "left", "center"), (-7, 0, "right", "center"), (0, 8, "center", "bottom"),
               (0, -8, "center", "top"), (7, 7, "left", "bottom"), (-7, 7, "right", "bottom"),
               (7, -7, "left", "top"), (-7, -7, "right", "top")]
    for x, y, text, bold in pts:
        px, py = ax.transData.transform((x, y))
        own = Bbox.from_extents(px - 7, py - 7, px + 7, py + 7)
        others = [b for b in placed if not (b.x0 == own.x0 and b.y0 == own.y0)]
        for dx, dy, ha, va in offsets:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                            va=va, fontsize=8.5, color=xs.INK,
                            fontweight="bold" if bold else "normal")
            bb = t.get_window_extent(r).expanded(1.05, 1.15)
            if not any(bb.overlaps(o) for o in others):
                placed.append(bb)
                others.append(bb)
                break
            t.remove()
        else:
            ax.annotate(text, (x, y), xytext=(7, 0), textcoords="offset points", ha="left",
                        va="center", fontsize=8.5, color=xs.INK)


def ex_peer_supply_vs_population(panel, hu, out: Path, base=2019, y1=2025):
    n = y1 - base
    rows = {}
    for code in [*CMP, US]:
        m = panel[panel.cbsa_code == code].set_index("year")
        stock = hu.set_index("cbsa_code").loc[code, "housing_units_2020"]
        rows[code] = {
            "supply_pct": 100 * m.loc[base + 1:y1, "units_total"].mean() / stock,
            "pop_pct": 100 * ((1 + m.loc[base + 1:y1, "pop_change"].sum()
                               / m.loc[base, "population"]) ** (1 / n) - 1),
        }
    g = pd.DataFrame(rows).T
    fig, ax = xs.new_exhibit(
        "Housing supply vs. population growth: Texas and its fast-growing peers",
        f"Average annual growth, {base}–{y1}. Above the line, permits added to the housing "
        f"stock faster than population grew.",
        source=f"{SRC_BPS}; {SRC_PEP}",
        note=f"{PEER_NOTE} Supply = units permitted ÷ 2020 housing stock (gross additions, "
             f"before demolitions).", figsize=(8, 6.4), panel_titles=False)
    lim = max(g.max()) * 1.12
    ax.plot([0, lim], [0, lim], color=xs.BASELINE, lw=1, zorder=1)
    ax.text(lim * 0.97, lim * 0.97, "supply = population growth ", ha="right", va="bottom",
            fontsize=8, color=xs.MUTED, rotation=0)
    for code, r in g.iterrows():
        color = xs.METRO_COLORS.get(code, xs.PEER)
        ax.plot(r.pop_pct, r.supply_pct, "o", ms=9 if code in ORDER else 7.5, color=color,
                mec=xs.SURFACE, mew=1.5, zorder=3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(axis="x", visible=True)
    ax.xaxis.set_major_formatter(xs.fmt_pct(1))
    ax.yaxis.set_major_formatter(xs.fmt_pct(1))
    ax.set_xlabel("Population growth, % per year", color=xs.INK_2, fontsize=9)
    ax.set_ylabel("Housing supply growth, % per year", color=xs.INK_2, fontsize=9)
    ax.spines["left"].set_visible(True)
    order = sorted(g.index, key=lambda c: (c not in ORDER, c))
    _place_point_labels(fig, ax, [(g.loc[c, "pop_pct"], g.loc[c, "supply_pct"], NAMES[c],
                                   c in ORDER) for c in order])
    g.rename(NAMES).round(3).to_csv(out / "11_peers_supply_vs_population.csv")
    xs.save_exhibit(fig, out / "11_peers_supply_vs_population")
    return g


def ex_peer_ytd(ytd, out: Path):
    month = int(ytd.through_month.iloc[0])
    y_cur = int(ytd.year.max())
    w = ytd.set_index(["cbsa_code", "year"])["units_total"]
    chg = pd.Series({c: 100 * (w[(c, y_cur)] / w[(c, y_cur - 1)] - 1) for c in [*CMP, US]})
    n_down = int((chg[CMP] <= -0.5).sum())  # matches the rounded labels (−0.4% shows as 0%)
    n_up = int((chg[CMP] >= 0.5).sum())
    if n_down >= len(CMP) / 2:
        title = f"{n_down} of {len(CMP)} fast-growing metros are permitting less housing in {y_cur}"
    else:
        title = f"{n_up} of {len(CMP)} fast-growing metros are permitting more housing in {y_cur}"
    fig, ax = xs.new_exhibit(
        title, f"Change in total units permitted, January–{MONTHS[month]} {y_cur} vs. the same "
        f"months of {y_cur - 1}", source=SRC_BPS,
        note=f"{PEER_NOTE} U.S. {_signed(chg[US])}.", figsize=(8, 6.2), left=1.45)
    show_us = abs(chg[US]) >= 1  # otherwise it sits on the zero line; the note carries it
    _ranked_bars(ax, chg[CMP], _signed, ref=chg[US] if show_us else None,
                 ref_label=f"U.S. {_signed(chg[US])}")
    ax.axvline(0, color=xs.BASELINE, lw=1)
    ax.set_xlim(min(chg[CMP].min() * 1.2, -10), max(chg[CMP].max() * 1.5, 10))
    ax.xaxis.set_major_formatter(xs.FuncFormatter(lambda v, _: _signed(v)))
    chg.rename(NAMES).round(2).to_csv(out / "12_peers_ytd_permits.csv", header=["pct_change"])
    xs.save_exhibit(fig, out / "12_peers_ytd_permits")
    return chg


def ex_peer_multifamily_retreat(panel, out: Path, peak=(2021, 2022)):
    last = int(panel.year.max())
    p = panel.assign(mf_1k=1000 * panel.units_2p / panel.population)
    w = p.pivot(index="cbsa_code", columns="year", values="mf_1k")
    d = pd.DataFrame({"peak": w[list(peak)].mean(axis=1), "now": w[last]}).loc[[*CMP, US]]
    d["chg_pct"] = 100 * (d.now / d.peak - 1)
    v = d.loc[CMP].sort_values("now", ascending=False)
    n_down = int((v.chg_pct <= -10).sum())
    if n_down == len(v):
        title = "Apartment permitting has retreated in every peer metro since the 2021–22 boom"
    elif n_down >= len(v) / 2:
        title = (f"Apartment permitting has retreated in {n_down} of {len(v)} fast-growing "
                 f"metros since the 2021–22 boom")
    else:
        title = "Apartment permitting per resident, Texas vs. peer metros"
    fig, ax = xs.new_exhibit(
        title,
        f"Multifamily (2+ unit) units permitted per 1,000 residents: {peak[0]}–{str(peak[1])[2:]} "
        f"average vs. {last}",
        source=f"{SRC_BPS}; {SRC_PEP}",
        legend=[(f"{peak[0]}–{str(peak[1])[2:]} average", xs.PEER),
                (f"{last} (Texas metros in color)", xs.INK_2)],
        legend_kind="patch", note=f"U.S.: {d.loc[US, 'peak']:.1f} → "
        f"{d.loc[US, 'now']:.1f}.", figsize=(8, 6.4), left=1.45)
    y = _style_category_axis(ax, [NAMES[c] for c in v.index])
    for lab, code in zip(ax.get_yticklabels(), v.index):
        if code in ORDER:
            lab.set_fontweight("bold")
    for yy, (code, r) in zip(y, v.iterrows()):
        color = xs.METRO_COLORS.get(code, xs.INK_2)
        ax.plot([r.now, r.peak], [yy, yy], color=xs.GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.plot(r.peak, yy, "o", ms=8, color=xs.PEER, mec=xs.SURFACE, mew=1.5, zorder=2)
        ax.plot(r.now, yy, "o", ms=9, color=color, mec=xs.SURFACE, mew=1.5, zorder=3)
        ax.text(max(r.peak, r.now) + 0.25, yy, _signed(r.chg_pct), va="center", ha="left",
                fontsize=8.5, color=xs.INK_2)
    ax.set_xlim(0, v[["peak", "now"]].max().max() * 1.15)
    ax.xaxis.set_major_formatter(xs.fmt_thousands())
    d.rename(NAMES).round(2).to_csv(out / "13_peers_multifamily_retreat.csv")
    xs.save_exhibit(fig, out / "13_peers_multifamily_retreat")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--brand", default=None, help="Footer credit (default: none)")
    ap.add_argument("--end", type=int, default=2025, help="Last full year of annual data")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.brand is not None:
        xs.BRAND = args.brand

    out = args.out or ROOT / "outputs" / "supply_exhibits" / date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)

    panel = metro_panel(CMP, 2000, args.end)
    panel.assign(metro=panel.cbsa_code.map(NAMES)).to_csv(out / "panel_texas_and_peers.csv",
                                                          index=False)
    hu = cbsa_housing_units_2020()
    ytd = cbsa_permits_ytd(CMP, args.end + 1)
    cons = construction_monthly()

    facts = ex_us_multifamily(cons, out)
    regions = ex_regions(cons, out)
    ex_permits_by_type(panel, out)
    per1k = ex_permits_per_capita(panel, out)
    growth = ex_growth_rates(panel, hu, out, y1=args.end)
    ratio = ex_permits_per_household(panel, out)
    ytd_chg = ex_ytd(ytd, out)
    ex_scorecard(panel, ytd, ratio, out)

    tx_ranks = ex_peer_permits_per_capita(panel, out)
    peer_ratio = ex_peer_permits_per_household(panel, out)
    peer_growth = ex_peer_supply_vs_population(panel, hu, out, y1=args.end)
    peer_ytd = ex_peer_ytd(ytd, out)
    peer_mf = ex_peer_multifamily_retreat(panel, out)

    print(f"Wrote exhibits to {out}")
    print(facts)
    print(regions.round(0).to_string(index=False))
    print(per1k.round(1).to_string())
    print(growth.round(2).to_string(index=False))
    print(ratio.round(2).to_string(index=False))
    print(ytd_chg.round(1).to_string(index=False))
    print("Texas rank in permits per 1k among", len(CMP), tx_ranks)
    print(peer_ratio.rename(NAMES).round(2).to_string())
    print(peer_growth.rename(NAMES).round(2).to_string())
    print(peer_ytd.rename(NAMES).round(1).to_string())
    print(peer_mf.rename(NAMES).round(2).to_string())


if __name__ == "__main__":
    main()
