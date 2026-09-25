"""Composite location quality score for workforce housing.

Methodology basis:
  - HUD Opportunity Index (HUD USER, 2019)
  - Enterprise Community Partners Opportunity360 framework
  - NYU Furman Center CoreData scoring methodology

Usage:
    from geostack.location_score import location_score, plot_location_score
    result = location_score(affordability_result=..., supply_result=..., ...)
    fig = plot_location_score(result["components"], result["composite"], property_name="The Republics")
"""

from __future__ import annotations

import numpy as np
from rich.console import Console

console = Console()

DEFAULT_WEIGHTS = {
    "income_depth": 0.25, "employment_access": 0.20, "supply_pressure": 0.15,
    "school_quality": 0.10, "transit_access": 0.10, "rent_burden": 0.10,
    "migration_growth": 0.10,
}

_NORM_RANGES = {
    "income_depth":      (0.20, 0.80, False),
    "employment_access": (0.30, 0.80, False),
    "supply_pressure":   (0, 5000, True),
    "school_quality":    (0, 4, False),
    "transit_access":    (0, 80, False),
    "rent_burden":       (0.30, 0.70, True),
    "migration_growth":  (-5.0, 10.0, False),
}

_TEA_RATING_MAP = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4}


def _normalize(value: float, low: float, high: float, invert: bool = False) -> float:
    if high == low:
        return 50.0
    scaled = (value - low) / (high - low)
    scaled = max(0.0, min(1.0, scaled))
    if invert:
        scaled = 1.0 - scaled
    return scaled * 100.0


def _compute_composite(components: dict[str, float], weights: dict[str, float]) -> float:
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum:.3f}")
    total = 0.0
    for dim, weight in weights.items():
        total += components.get(dim, 0.0) * weight
    return total


def location_score(
    affordability_result: dict | None = None,
    supply_result: dict | None = None,
    school_result: dict | None = None,
    transit_result: dict | None = None,
    migration_result: dict | None = None,
    target_rent: int | None = None,
    weights: dict[str, float] | None = None,
) -> dict:
    w = weights or DEFAULT_WEIGHTS
    raw = {}
    components = {}

    if affordability_result and target_rent:
        from geostack.affordability import _affordability_at_rent
        prop_dist = affordability_result.get("distributions", {}).get("property_renter")
        if prop_dist is not None and not prop_dist.empty:
            raw["income_depth"] = _affordability_at_rent(prop_dist, target_rent)
        else:
            raw["income_depth"] = 0.5
    else:
        raw["income_depth"] = 0.5

    if affordability_result and affordability_result.get("commute"):
        earn = affordability_result["commute"].get("earn_distribution", {})
        raw["employment_access"] = earn.get("earn_high_pct", 0.5)
    else:
        raw["employment_access"] = 0.5

    if supply_result:
        raw["supply_pressure"] = supply_result.get("permitted_units_5plus", 0)
    else:
        raw["supply_pressure"] = 2500

    if school_result and school_result.get("tea_rating"):
        raw["school_quality"] = _TEA_RATING_MAP.get(school_result["tea_rating"], 2)
    else:
        raw["school_quality"] = 2

    if transit_result and transit_result.get("walkscore") is not None:
        raw["transit_access"] = transit_result["walkscore"]
    else:
        raw["transit_access"] = 40

    if affordability_result and affordability_result.get("rent_burden"):
        burden = affordability_result["rent_burden"]
        raw["rent_burden"] = burden.get("burdened_pct", 0.35) + burden.get("severely_burdened_pct", 0.15)
    else:
        raw["rent_burden"] = 0.50

    if migration_result and migration_result.get("soi"):
        raw["migration_growth"] = migration_result["soi"].get("income_delta", 0)
    else:
        raw["migration_growth"] = 2.5

    for dim, (low, high, invert) in _NORM_RANGES.items():
        components[dim] = _normalize(raw.get(dim, (low + high) / 2), low, high, invert)

    composite = _compute_composite(components, w)

    console.print(f"\n[bold]Location Score: {composite:.1f} / 100[/bold]")
    for dim, score in sorted(components.items(), key=lambda x: -x[1]):
        console.print(f"  {dim:.<30s} {score:5.1f}")

    return {"composite": composite, "components": components, "weights": w, "raw_values": raw}


def plot_location_score(components: dict[str, float], composite: float,
                         property_name: str | None = None,
                         save_path: str | None = None) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt

    labels = list(components.keys())
    values = [components[k] for k in labels]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, values_closed, "o-", linewidth=2, color="#c0392b")
    ax.fill(angles_closed, values_closed, alpha=0.15, color="#c0392b")

    neutral = [50] * (len(labels) + 1)
    ax.plot(angles_closed, neutral, "--", linewidth=1, color="#999", alpha=0.5)

    ax.set_thetagrids([a * 180 / np.pi for a in angles],
                       [l.replace("_", " ").title() for l in labels], fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], fontsize=8, color="gray")

    title = f"Location Score: {composite:.0f}/100"
    if property_name:
        title = f"{property_name}\n{title}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    fig.text(0.99, 0.01, "Source: geostack composite index | HUD/Enterprise methodology",
             ha="right", fontsize=7, color="gray", style="italic")

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        console.print(f"[green]Saved → {save_path}[/green]")

    return fig
