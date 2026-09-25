"""MSA Population Center of Gravity — CLI entry point.

Computes how a metro's weighted geographic center has shifted over time,
supporting multiple weighting modes: population, workplace jobs (LODES WAC),
resident workers (LODES RAC), and income mass (ACS HH × MHI).

Example — single mode (population):
    python scripts/msa_population_gravity.py \\
        --msa "Dallas-Fort Worth-Arlington, TX" \\
        --years 2010,2013,2018,2020,2023 \\
        --output-dir outputs/msa_gravity/dfw \\
        --make-map --make-chart

Example — multi-weight comparison:
    python scripts/msa_population_gravity.py \\
        --msa "Dallas" \\
        --years 2010,2013,2018,2020,2021 \\
        --weights population,jobs,workers,income_mass \\
        --output-dir outputs/msa_gravity/dfw_multi \\
        --make-map --make-chart

    # List available MSAs:
    python scripts/msa_population_gravity.py --list-msas

Data Sources:
    - Decennial Census (SF1/DHC): 2000, 2010, 2020
    - ACS 5-Year Estimates: 2009+ (tract-level B01003_001E, B19013_001E)
    - LODES WAC/RAC: 2002–2021 (block-level, aggregated to tract)
    - TIGER/Line CB 2020: tract polygon boundaries (all years)
    - Census 2010→2020 tract relationship file (crosswalk normalization)

Limitations:
    - Pre-2020 data crosswalked via areal weighting; tracts with complex
      boundary changes may have slightly inaccurate weight allocation
    - LODES only available through 2021
    - Income mass is a proxy (HH × median HHI), not true aggregate income
    - ACS 5-year estimates represent a period, not a single year
"""

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import logging

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="MSA Population Center of Gravity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--msa", type=str, default="Dallas-Fort Worth-Arlington, TX",
        help="MSA name or partial match (default: DFW)",
    )
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument(
        "--years", type=str, default=None,
        help="Explicit comma-separated year list (overrides start/end)",
    )
    parser.add_argument(
        "--weights", type=str, default="population",
        help="Comma-separated weight modes: population,jobs,workers,income_mass (default: population)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/msa_gravity",
        help="Output directory for results",
    )
    parser.add_argument("--make-map", action="store_true", help="Generate interactive Folium map")
    parser.add_argument("--make-chart", action="store_true", help="Generate static Matplotlib chart")
    parser.add_argument("--no-crosswalk", action="store_true", help="Disable crosswalk normalization")
    parser.add_argument("--list-msas", action="store_true", help="Print available MSA definitions")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    from geostack.gravity import (
        WEIGHT_LABELS,
        list_msas,
        resolve_msa,
        gravity_time_series,
        multi_weight_time_series,
        compute_shift_metrics,
        export_results,
    )

    if args.list_msas:
        console.print("[bold]Available MSA Definitions:[/bold]\n")
        for name in list_msas():
            msa = resolve_msa(name)
            console.print(f"  • {name}  ({len(msa['counties'])} counties)")
        return

    msa = resolve_msa(args.msa)
    console.print(f"\n[bold]MSA:[/bold] {msa['name']}")
    console.print(f"[bold]Counties:[/bold] {len(msa['counties'])} ({', '.join(msa['counties'].values())})")

    years = None
    if args.years:
        years = [int(y.strip()) for y in args.years.split(",")]

    weight_modes = [w.strip() for w in args.weights.split(",")]
    use_crosswalk = not args.no_crosswalk

    console.print(f"[bold]Weights:[/bold] {', '.join(WEIGHT_LABELS.get(w, w) for w in weight_modes)}")
    console.print(f"[bold]Crosswalk:[/bold] {'enabled' if use_crosswalk else 'disabled'}")
    console.print()

    import pathlib
    out = pathlib.Path(args.output_dir)

    # Always compute the full multi-weight series so the hero map and
    # divergence chart have every center; single-weight requests still get
    # their own contour map + table.
    series_modes = weight_modes if len(weight_modes) > 1 else weight_modes
    combined_df = multi_weight_time_series(
        args.msa,
        start_year=args.start_year,
        end_year=args.end_year,
        years=years,
        weight_modes=weight_modes if len(weight_modes) > 1 else (["population"] + weight_modes
                                                                 if "population" not in weight_modes
                                                                 else weight_modes),
        use_crosswalk=use_crosswalk,
    )

    # Tables per requested mode
    for mode in weight_modes:
        mode_df = combined_df[combined_df["weight_mode"] == mode].sort_values("year")
        if not mode_df.empty:
            _print_table(mode_df, WEIGHT_LABELS.get(mode, mode))

    console.print("\n[bold]Exporting results...[/bold]")
    export_results(combined_df, args.output_dir, prefix="gravity_centers")

    latest = int(combined_df["year"].max())

    # ── Static maps ──
    if args.make_map:
        from geostack.gravity_viz import (
            gravity_contour_map, gravity_contour_panels, gravity_hero_interactive,
        )

        # One topographic structure map per requested weight (the IC money shot)
        first_yr = int(combined_df["year"].min())
        for mode in weight_modes:
            mdf = combined_df[combined_df["weight_mode"] == mode]
            if mdf.empty:
                continue
            yr = int(mdf["year"].max())
            console.print(f"\n[bold]Contour map — {mode} ({yr})...[/bold]")
            gravity_contour_map(
                args.msa, weight_mode=mode, display_year=yr,
                drift_from_year=first_yr, use_crosswalk=use_crosswalk,
                output_path=out / f"contour_{mode}_{yr}.png",
            )
            # Side-by-side year panels — makes the center-of-gravity drift stark
            console.print(f"[bold]Contour panels — {mode}...[/bold]")
            try:
                gravity_contour_panels(
                    args.msa, weight_mode=mode, use_crosswalk=use_crosswalk,
                    output_path=out / f"panels_{mode}.png",
                )
            except Exception as e:
                console.print(f"  [yellow]panels skipped: {e}[/yellow]")

        # Interactive hero companion (where + divergence + heading)
        console.print(f"\n[bold]Interactive hero map...[/bold]")
        gravity_hero_interactive(
            combined_df, display_year=latest,
            output_path=out / "hero_interactive.html",
        )

    # ── Static charts ──
    if args.make_chart:
        from geostack.gravity_viz import gravity_hero_map, gravity_trajectory_map
        console.print(f"\n[bold]Static hero map...[/bold]")
        gravity_hero_map(combined_df, display_year=latest,
                         output_path=out / "hero.png")

        # Trajectory + growth-share (per requested mode)
        first_yr = int(combined_df["year"].min())
        for mode in weight_modes:
            mdf = combined_df[combined_df["weight_mode"] == mode]
            if mdf.empty:
                continue
            yr = int(mdf["year"].max())
            console.print(f"\n[bold]Trajectory + growth-share — {mode}...[/bold]")
            try:
                gravity_trajectory_map(
                    args.msa, start_year=first_yr, end_year=yr, target_year=yr + 10,
                    weight_mode=mode, use_crosswalk=use_crosswalk,
                    output_path=out / f"trajectory_{mode}.png",
                )
            except Exception as e:
                console.print(f"  [yellow]trajectory skipped: {e}[/yellow]")

        if len(weight_modes) > 1:
            from geostack.gravity_viz import divergence_chart
            console.print(f"\n[bold]Divergence chart...[/bold]")
            divergence_chart(combined_df, output_path=out / "divergence_chart.png")

    # Net-shift summary for the population series
    pop_df = combined_df[combined_df["weight_mode"] == "population"].sort_values("year")
    if len(pop_df) >= 2:
        _print_summary(pop_df)

    console.print(f"\n[bold green]Done![/bold green] Results in: {args.output_dir}/")


def _print_table(df: pd.DataFrame, mode_label: str) -> None:
    """Print a Rich table for a single weight mode's results."""
    table = Table(title=f"{mode_label} — Center of Gravity")
    table.add_column("Year", style="bold")
    table.add_column("Weight", justify="right")
    table.add_column("Lat", justify="right")
    table.add_column("Lon", justify="right")
    table.add_column("Shift (mi)", justify="right")
    table.add_column("Bearing", justify="right")
    table.add_column("N/S (mi)", justify="right")
    table.add_column("E/W (mi)", justify="right")
    table.add_column("Source")

    has_shift = "shift_miles" in df.columns

    for _, row in df.iterrows():
        weight_val = row.get("total_weight", row.get("total_population", 0))
        table.add_row(
            str(int(row["year"])),
            f"{weight_val:,.0f}",
            f"{row['center_lat']:.4f}",
            f"{row['center_lon']:.4f}",
            f"{row['shift_miles']:.3f}" if has_shift else "—",
            f"{row['shift_bearing_deg']:.0f}°" if has_shift else "—",
            f"{row['shift_north_miles']:+.3f}" if has_shift else "—",
            f"{row['shift_east_miles']:+.3f}" if has_shift else "—",
            str(row.get("data_source", "")),
        )

    console.print()
    console.print(table)


def _print_summary(df: pd.DataFrame) -> None:
    """Print net shift summary."""
    total_shift = df.iloc[-1].get("shift_miles", 0)
    bearing = df.iloc[-1].get("shift_bearing_deg", 0)
    direction = _bearing_to_compass(bearing)
    console.print(
        f"  Net shift: {total_shift:.2f} miles {direction} "
        f"({int(df.iloc[0]['year'])}→{int(df.iloc[-1]['year'])})"
    )


def _bearing_to_compass(bearing: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((bearing + 11.25) / 22.5) % 16
    return dirs[idx]


if __name__ == "__main__":
    main()
