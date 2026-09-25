"""Geostack CLI — common commands for geospatial data management.

Usage:
    geostack download-tracts --state 48 --year 2023
    geostack db-status
"""

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def cli():
    """Geostack — geospatial analytics for CRE."""
    pass


@cli.command()
def db_status():
    """Check PostGIS database connection and extensions."""
    from geostack.db import get_engine, read_sql

    try:
        engine = get_engine()
        row = read_sql("SELECT PostGIS_Full_Version() as v")
        console.print(f"[green]Connected to PostGIS[/green]")
        console.print(f"[dim]{row['v'].iloc[0]}[/dim]")

        # Show tables with geometry columns
        geo_tables = read_sql("""
            SELECT f_table_name, f_geometry_column, srid, type
            FROM geometry_columns
            ORDER BY f_table_name
        """)
        if not geo_tables.empty:
            table = Table(title="Spatial Tables")
            table.add_column("Table")
            table.add_column("Geometry Column")
            table.add_column("SRID")
            table.add_column("Type")
            for _, row in geo_tables.iterrows():
                table.add_row(
                    row["f_table_name"],
                    row["f_geometry_column"],
                    str(row["srid"]),
                    row["type"],
                )
            console.print(table)
        else:
            console.print("[dim]No spatial tables found yet.[/dim]")
    except Exception as e:
        console.print(f"[red]Connection failed: {e}[/red]")


@cli.command()
@click.option("--state", required=True, help="State FIPS code (e.g., 48 for Texas)")
@click.option("--year", default=2023, help="TIGER vintage year")
@click.option("--load-db", is_flag=True, help="Also load into PostGIS")
def download_tracts(state, year, load_db):
    """Download census tract boundaries from TIGER/Line."""
    from geostack.tiger import download_tracts as _download

    gdf = _download(state_fips=state, year=year)
    console.print(f"Downloaded {len(gdf)} tracts")

    if load_db:
        from geostack.db import get_engine

        engine = get_engine()
        gdf.to_postgis(f"tiger_tracts_{state}_{year}", engine, if_exists="replace")
        console.print(f"[green]Loaded to PostGIS: tiger_tracts_{state}_{year}[/green]")


@cli.command()
@click.option("--state", required=True, help="State FIPS code")
@click.option("--county", required=True, help="County FIPS code")
@click.option("--year", default=2023, help="TIGER vintage year")
@click.option("--load-db", is_flag=True, help="Also load into PostGIS")
def download_roads(state, county, year, load_db):
    """Download road centerlines from TIGER/Line for a county."""
    from geostack.tiger import download_roads as _download

    gdf = _download(state_fips=state, county_fips=county, year=year)
    console.print(f"Downloaded {len(gdf)} road segments")

    if load_db:
        from geostack.db import get_engine

        engine = get_engine()
        table_name = f"tiger_roads_{state}{county}_{year}"
        gdf.to_postgis(table_name, engine, if_exists="replace")
        console.print(f"[green]Loaded to PostGIS: {table_name}[/green]")


@cli.command()
@click.option("--state", required=True, help="State FIPS code (e.g., 48 for Texas)")
@click.option("--county", required=True, help="3-digit county FIPS (e.g., 085 for Collin)")
@click.option("--year", default=2023, show_default=True, help="ACS 5-year vintage year")
def load_acs(state, county, year):
    """Load ACS 5-year demographic estimates for a county into PostGIS.

    Fetches all CRE-relevant variables from census.ACS_VARIABLES and upserts
    into acs_tract_data (one row per tract × variable × year).

    Requires tracts to be loaded first: `geostack load-county --state X --county Y`

    Idempotent — skips tracts already loaded for this year.

    Examples:
        geostack load-acs --state 48 --county 085 --year 2023
        geostack load-acs --state 48 --county 113 --year 2022
    """
    from geostack.census import ACS_VARIABLES, CensusClient
    from geostack.db import get_engine, read_sql

    state_fips = str(state).zfill(2)
    county_fips = str(county).zfill(3)

    engine = get_engine()

    # Require tracts to be present (FK constraint)
    tract_count = read_sql(
        "SELECT COUNT(*) AS n FROM census_tracts WHERE state_fips = :s AND county_fips = :c",
        params={"s": state_fips, "c": county_fips},
    )["n"].iloc[0]

    if tract_count == 0:
        console.print(
            f"[red]No tracts loaded for {state_fips}{county_fips}. "
            f"Run: geostack load-county --state {state} --county {county}[/red]"
        )
        return

    console.print(f"\n[bold]ACS {year} — County {state_fips}{county_fips} ({tract_count} tracts)[/bold]")

    # GEOIDs already in acs_tract_data for this county+year — skip them
    already_loaded = set(
        read_sql(
            """
            SELECT DISTINCT a.geoid FROM acs_tract_data a
            JOIN census_tracts t ON a.geoid = t.geoid
            WHERE t.state_fips = :s AND t.county_fips = :c AND a.year = :y
            """,
            params={"s": state_fips, "c": county_fips, "y": year},
        )["geoid"].tolist()
    )

    if len(already_loaded) >= tract_count:
        console.print(f"[dim]ACS {year} already fully loaded for this county — skipping.[/dim]")
        return

    # Valid GEOIDs in census_tracts for this county
    valid_geoids = set(
        read_sql(
            "SELECT geoid FROM census_tracts WHERE state_fips = :s AND county_fips = :c",
            params={"s": state_fips, "c": county_fips},
        )["geoid"].tolist()
    )
    new_geoids = valid_geoids - already_loaded

    # Fetch ACS data from Census Bureau API
    client = CensusClient()
    variables = list(ACS_VARIABLES.values())
    console.print(f"[dim]Fetching {len(variables)} variables for {len(new_geoids)} tracts...[/dim]")

    try:
        df = client.get_acs(variables, geography="tract", state=state_fips, county=county_fips, year=year)
    except Exception as e:
        console.print(f"[red]Census API error: {e}[/red]")
        return

    df = client.build_geoid(df, "tract")

    # Filter to only new tracts with valid GEOIDs
    df = df[df["GEOID"].isin(new_geoids)]

    if df.empty:
        console.print("[dim]No new tracts to load.[/dim]")
        return

    # Melt to long (EAV) format: one row per (geoid, year, variable, value)
    available_vars = [v for v in variables if v in df.columns]
    long_df = df[["GEOID"] + available_vars].melt(
        id_vars="GEOID",
        value_vars=available_vars,
        var_name="variable",
        value_name="value",
    )
    long_df = long_df.rename(columns={"GEOID": "geoid"})
    long_df["year"] = year
    long_df = long_df[["geoid", "year", "variable", "value"]]

    # Drop Census missing-value sentinel (-666666666) and NaNs
    CENSUS_MISSING = -666666666
    long_df = long_df[long_df["value"].notna() & (long_df["value"] != CENSUS_MISSING)]

    long_df.to_sql("acs_tract_data", engine, if_exists="append", index=False)

    tracts_loaded = long_df["geoid"].nunique()
    vars_loaded = long_df["variable"].nunique()

    console.print(f"\n[bold green]Done![/bold green]")
    table = Table(title=f"ACS {year} — County {state_fips}{county_fips}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Tracts loaded", f"{tracts_loaded:,}")
    table.add_row("Variables per tract", f"{vars_loaded:,}")
    table.add_row("Total rows inserted", f"{len(long_df):,}")
    console.print(table)


@cli.command()
@click.option("--state", required=True, help="State FIPS code (e.g., 48 for Texas)")
@click.option("--county", required=True, help="County FIPS code (e.g., 085 for Collin)")
@click.option("--year", default=2023, help="TIGER vintage year")
def load_county(state, county, year):
    """Load all reference data for a county into PostGIS.

    Downloads (if not cached) and loads:
    - Census tract boundaries for the state (shared, loaded once)
    - Road centerlines for the county

    Safe to run repeatedly — skips data already loaded.

    Example: geostack load-county --state 48 --county 085
    """
    from geostack.db import get_engine
    from geostack.tiger import (
        download_roads,
        download_tracts,
        load_roads_to_postgis,
        load_tracts_to_postgis,
    )

    engine = get_engine()

    # 1. Tracts (state-level, deduplicated on insert)
    console.print(f"\n[bold]Step 1:[/bold] Census tracts for state {state}")
    tracts_gdf = download_tracts(state_fips=state, year=year)
    load_tracts_to_postgis(tracts_gdf, engine)

    # 2. Roads (county-level, replaces existing for this county)
    console.print(f"\n[bold]Step 2:[/bold] Roads for county {state}{county}")
    roads_gdf = download_roads(state_fips=state, county_fips=county, year=year)
    load_roads_to_postgis(roads_gdf, state, county, engine)

    # Summary
    from geostack.db import read_sql

    counts = read_sql("""
        SELECT
            (SELECT COUNT(*) FROM census_tracts) as total_tracts,
            (SELECT COUNT(*) FROM census_tracts WHERE state_fips = :s AND county_fips = :c) as county_tracts,
            (SELECT COUNT(*) FROM roads WHERE state_fips = :s AND county_fips = :c) as county_roads
    """, params={"s": state, "c": county})

    console.print(f"\n[bold green]Done![/bold green]")
    table = Table(title=f"Loaded Data — County {state}{county}")
    table.add_column("Dataset")
    table.add_column("Count", justify="right")
    table.add_row("Total tracts (statewide)", f"{counts['total_tracts'].iloc[0]:,}")
    table.add_row("Tracts in this county", f"{counts['county_tracts'].iloc[0]:,}")
    table.add_row("Road segments", f"{counts['county_roads'].iloc[0]:,}")
    console.print(table)


@cli.command()
@click.option("--files", is_flag=True, help="Clear downloaded shapefile cache")
@click.option("--county", multiple=True, help="Remove a county's roads from PostGIS (e.g., 48085)")
@click.option("--all-roads", is_flag=True, help="Remove ALL road data from PostGIS")
@click.option("--all-tracts", is_flag=True, help="Remove ALL tract data from PostGIS")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def clean(files, county, all_roads, all_tracts, yes):
    """Show cache usage and optionally clean up disk/database.

    Run with no flags to see current usage. Add flags to clean specific data.
    PostGIS data can always be reloaded with `geostack load-county`.

    Examples:
        geostack clean                          # show usage
        geostack clean --files                  # delete shapefile cache
        geostack clean --county 48085           # remove Collin roads from DB
        geostack clean --files --all-roads      # full cleanup
    """
    import os
    import shutil

    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "shapefiles")
    cache_dir = os.path.normpath(cache_dir)

    # Always show current usage first
    console.print("\n[bold]Disk Cache[/bold] (data/shapefiles/)")
    total_cache_bytes = 0
    cache_items = []
    if os.path.exists(cache_dir):
        for entry in sorted(os.listdir(cache_dir)):
            entry_path = os.path.join(cache_dir, entry)
            if entry.startswith("."):
                continue
            if os.path.isdir(entry_path):
                size = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, filenames in os.walk(entry_path)
                    for f in filenames
                )
                total_cache_bytes += size
                cache_items.append((entry, size))

    if cache_items:
        table = Table()
        table.add_column("Cached File")
        table.add_column("Size", justify="right")
        for name, size in cache_items:
            table.add_row(name, _human_size(size))
        table.add_row("[bold]Total[/bold]", f"[bold]{_human_size(total_cache_bytes)}[/bold]")
        console.print(table)
    else:
        console.print("  [dim]Empty — nothing cached.[/dim]")

    # Show PostGIS usage
    console.print("\n[bold]PostGIS Tables[/bold]")
    try:
        from geostack.db import read_sql

        db_usage = read_sql("""
            SELECT
                state_fips || county_fips as county_fips,
                COUNT(*) as road_segments,
                pg_size_pretty(
                    SUM(ST_MemSize(geometry::geometry))::bigint
                ) as geom_size
            FROM roads
            GROUP BY state_fips, county_fips
            ORDER BY state_fips, county_fips
        """)
        tract_count = read_sql("SELECT COUNT(*) as n FROM census_tracts")

        console.print(f"  Census tracts: {tract_count['n'].iloc[0]:,}")
        if not db_usage.empty:
            table = Table(title="Roads by County")
            table.add_column("County FIPS")
            table.add_column("Segments", justify="right")
            table.add_column("Geometry Size", justify="right")
            for _, row in db_usage.iterrows():
                table.add_row(row["county_fips"], f"{row['road_segments']:,}", row["geom_size"])
            console.print(table)
        else:
            console.print("  [dim]No roads loaded.[/dim]")
    except Exception as e:
        console.print(f"  [yellow]Could not query PostGIS: {e}[/yellow]")

    # If no action flags, we're done (just showing status)
    if not files and not county and not all_roads and not all_tracts:
        console.print(
            "\n[dim]Add --files, --county, --all-roads, or --all-tracts to clean. "
            "All data is re-downloadable via `geostack load-county`.[/dim]"
        )
        return

    # Confirm before deleting
    actions = []
    if files:
        actions.append(f"Delete shapefile cache ({_human_size(total_cache_bytes)})")
    if county:
        actions.append(f"Remove roads for counties: {', '.join(county)}")
    if all_roads:
        actions.append("Remove ALL roads from PostGIS")
    if all_tracts:
        actions.append("Remove ALL tracts from PostGIS")

    console.print("\n[bold yellow]Planned actions:[/bold yellow]")
    for a in actions:
        console.print(f"  - {a}")

    if not yes:
        if not click.confirm("\nProceed?"):
            console.print("[dim]Cancelled.[/dim]")
            return

    # Execute
    if files and os.path.exists(cache_dir):
        for entry in os.listdir(cache_dir):
            entry_path = os.path.join(cache_dir, entry)
            if entry.startswith("."):
                continue
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
        console.print(f"[green]Cleared shapefile cache ({_human_size(total_cache_bytes)})[/green]")

    if county or all_roads or all_tracts:
        from geostack.db import execute_sql

        if all_roads:
            execute_sql("DELETE FROM roads")
            console.print("[green]Removed all roads from PostGIS[/green]")
        elif county:
            for c in county:
                state_fips = c[:2]
                county_fips = c[2:]
                execute_sql(
                    "DELETE FROM roads WHERE state_fips = :s AND county_fips = :c",
                    params={"s": state_fips, "c": county_fips},
                )
                console.print(f"[green]Removed roads for {c}[/green]")

        if all_tracts:
            execute_sql("DELETE FROM acs_tract_data")
            execute_sql("DELETE FROM census_tracts")
            console.print("[green]Removed all tracts and ACS data from PostGIS[/green]")


@cli.command()
@click.option("--state", required=True, help="State FIPS (e.g., 48) or abbreviation (tx)")
@click.option("--county", required=True, help="3-digit county FIPS (e.g., 085)")
@click.option("--year", default=2021, show_default=True, help="LODES data year (roughly 2002–2021)")
@click.option("--job-type", default="JT01", show_default=True,
              help="JT01=primary jobs (recommended), JT00=all jobs")
@click.option("--od", is_flag=True, help="Also load Origin-Destination flow data (large file)")
@click.option("--no-cache", is_flag=True, help="Skip local cache, always re-download")
def load_lodes(state, county, year, job_type, od, no_cache):
    """Load LODES8 employment data for a county into PostGIS.

    Downloads WAC (workplace) and RAC (residential) data for the state,
    filters to the specified county, and loads into lodes_wac / lodes_rac.
    Optionally loads OD (origin-destination) flow data as well.

    LODES8 files are state-level (Texas WAC is ~180 MB compressed), so the
    first run for a state takes longer — subsequent counties reuse the cache.

    Run db/migrations/002_lodes.sql against your geostack DB first.

    Examples:
        geostack load-lodes --state 48 --county 085 --year 2021
        geostack load-lodes --state tx --county 113 --year 2021 --od
    """
    from geostack.db import get_engine
    from geostack.lodes import (
        download_wac, download_rac, download_od,
        load_wac_to_postgis, load_rac_to_postgis, load_od_to_postgis,
        _resolve_state_abbr, FIPS_TO_ABBR,
    )

    # Resolve state FIPS for the county_filter (need 5-digit geoid prefix)
    abbr = _resolve_state_abbr(state)
    # Find FIPS from abbr if user passed abbreviation
    state_fips = state if (len(str(state)) == 2 and str(state).isdigit()) else \
        next((k for k, v in FIPS_TO_ABBR.items() if v == abbr), state)
    county_fips = str(county).zfill(3)
    county_filter = f"{str(state_fips).zfill(2)}{county_fips}"

    engine = get_engine()
    cache = not no_cache

    console.print(f"\n[bold]LODES {year} — County {county_filter} ({job_type})[/bold]")

    # WAC
    console.print(f"\n[bold]Step 1:[/bold] Workplace Area Characteristics (WAC)")
    wac = download_wac(state=abbr, year=year, job_type=job_type, county_filter=county_filter, cache=cache)
    load_wac_to_postgis(wac, engine, year=year, job_type=job_type)

    # RAC
    console.print(f"\n[bold]Step 2:[/bold] Residential Area Characteristics (RAC)")
    rac = download_rac(state=abbr, year=year, job_type=job_type, county_filter=county_filter, cache=cache)
    load_rac_to_postgis(rac, engine, year=year, job_type=job_type)

    # OD (optional — large file)
    if od:
        console.print(f"\n[bold]Step 3:[/bold] Origin-Destination flows (main)")
        od_df = download_od(state=abbr, year=year, job_type=job_type, county_filter=county_filter, cache=cache)
        load_od_to_postgis(od_df, engine, year=year, job_type=job_type)

    # Summary
    from geostack.db import read_sql

    wac_count = read_sql(
        "SELECT SUM(total_jobs) as jobs FROM lodes_wac "
        "WHERE state_fips = :s AND county_fips = :c AND year = :y AND job_type = :jt",
        params={"s": str(state_fips).zfill(2), "c": county_fips, "y": year, "jt": job_type},
    )
    rac_count = read_sql(
        "SELECT SUM(total_workers) as workers FROM lodes_rac "
        "WHERE state_fips = :s AND county_fips = :c AND year = :y AND job_type = :jt",
        params={"s": str(state_fips).zfill(2), "c": county_fips, "y": year, "jt": job_type},
    )

    console.print(f"\n[bold green]Done![/bold green]")
    table = Table(title=f"LODES {year} — County {county_filter}")
    table.add_column("Dataset")
    table.add_column("Count", justify="right")
    table.add_row("Total jobs (WAC)", f"{int(wac_count['jobs'].iloc[0] or 0):,}")
    table.add_row("Total resident workers (RAC)", f"{int(rac_count['workers'].iloc[0] or 0):,}")
    if od:
        od_count = read_sql(
            "SELECT COUNT(*) as flows, SUM(total_jobs) as jobs FROM lodes_od "
            "WHERE w_state_fips = :s AND w_county_fips = :c AND year = :y",
            params={"s": str(state_fips).zfill(2), "c": county_fips, "y": year},
        )
        table.add_row("OD flows (into county)", f"{int(od_count['flows'].iloc[0] or 0):,}")
    console.print(table)


@cli.command("load-parcels")
@click.option("--county", required=True,
              help="County name: collin, dallas, cooke, travis, grayson")
@click.option("--year", required=True, type=int,
              help="Tax year (e.g., 2024)")
@click.option("--all-years", is_flag=True,
              help="Load all available years for this county (Collin only)")
@click.option("--with-geometry", is_flag=True,
              help="Also fetch parcel geometry from ArcGIS REST and attach before loading")
@click.option("--core-only/--full", default=True, show_default=True,
              help="Fetch only core columns for Collin (~75% smaller). Ignored for other counties.")
@click.option("--limit", type=int, default=None,
              help="Cap records per year (useful for testing)")
@click.option("--app-token", envvar="SOCRATA_APP_TOKEN", default=None,
              help="Socrata app token for higher rate limits (or set SOCRATA_APP_TOKEN env var)")
def load_parcels(county, year, all_years, with_geometry, core_only, limit, app_token):
    """Fetch parcel appraisal data and load into the PostGIS parcels table.

    Downloads from the county's Socrata API, normalizes to the canonical
    ParcelYear schema, computes ILR and age, then replaces existing rows for
    the county + year (DELETE + re-insert — safe to re-run).

    Requires: geostack DB migration 003_parcels.sql must have been run.

    Examples:
        geostack load-parcels --county collin --year 2024
        geostack load-parcels --county collin --year 2024 --with-geometry
        geostack load-parcels --county collin --year 2024 --all-years
        geostack load-parcels --county dallas --year 2024 --limit 1000
    """
    from geostack import cad
    from geostack.cad import (
        COLLIN_DATASET_IDS,
        COUNTY_REGISTRY,
        fetch_collin_parcel_geometry,
        fetch_parcels,
        load_parcels_to_postgis,
        normalize_parcels,
        _arcgis_geojson_fetch,
    )
    from geostack.db import get_engine

    county = county.lower()
    if county not in COUNTY_REGISTRY:
        raise click.UsageError(
            f"Unknown county {county!r}. Supported: {list(COUNTY_REGISTRY)}"
        )

    meta = COUNTY_REGISTRY[county]

    # Build list of years to process
    if all_years:
        if county != "collin":
            raise click.UsageError("--all-years is currently only supported for Collin county.")
        years = sorted(COLLIN_DATASET_IDS.keys())
    else:
        years = [year]

    engine = get_engine()
    total_loaded = 0

    for yr in years:
        console.print(f"\n[bold]Step: {county.title()} County parcels — {yr}[/bold]")

        # 1. Fetch raw data
        try:
            df = fetch_parcels(county, year=yr, core_only=core_only, limit=limit, app_token=app_token)
        except Exception as e:
            console.print(f"[red]Fetch failed for {yr}: {e}[/red]")
            continue

        if df.empty:
            console.print(f"[yellow]No records returned for {county}/{yr} — skipping.[/yellow]")
            continue

        console.print(f"[dim]Fetched {len(df):,} raw records[/dim]")

        # 2. Normalize to canonical ParcelYear schema
        try:
            df = normalize_parcels(df, county)
        except Exception as e:
            console.print(f"[red]Normalization failed: {e}[/red]")
            continue

        ilr_pct = df["ilr"].notna().mean() * 100 if "ilr" in df.columns else 0.0
        console.print(
            f"[dim]Normalized: {len(df):,} rows, ILR coverage {ilr_pct:.0f}%[/dim]"
        )

        # 3. Optionally attach geometry
        geom_pct = 0.0
        if with_geometry:
            console.print("[dim]Fetching parcel geometry (this may take a moment)...[/dim]")
            try:
                import geopandas as gpd
                import pandas as _pd

                id_field = meta["geom_id_field"]
                geo_frames = []

                if meta.get("geom_fetch") == "txgio_bbox_keyset":
                    fetch_geometry = getattr(cad, f"fetch_{county}_parcel_geometry")
                    geo_frames.append(fetch_geometry())
                    join_key = "geo_id" if id_field == "geo_id" and "geo_id" in df.columns else "parcel_id"
                else:
                    gis_url = meta["gis_url"]
                    parcel_ids = df["parcel_id"].dropna().astype(str).tolist()
                    join_key = "parcel_id"

                    # ArcGIS limits to 1000 records per request — batch the fetch
                    batch_size = 1000
                    for i in range(0, len(parcel_ids), batch_size):
                        batch = parcel_ids[i : i + batch_size]
                        ids_str = ",".join(f"'{pid}'" for pid in batch)
                        where = f"{id_field} IN ({ids_str})"
                        try:
                            geo_frames.append(
                                _arcgis_geojson_fetch(gis_url, where, batch_size, county.title())
                            )
                        except Exception as exc:
                            console.print(f"[yellow]Geometry batch {i}–{i + batch_size} failed: {exc}[/yellow]")

                if geo_frames:
                    geom_gdf = gpd.GeoDataFrame(
                        _pd.concat(geo_frames, ignore_index=True),
                        crs="EPSG:4326",
                    )
                    # Join geometry onto the normalized DataFrame
                    geom_col = next(
                        (c for c in geom_gdf.columns if c.lower() == id_field.lower()), None
                    )
                    if geom_col:
                        geom_slim = geom_gdf[[geom_col, "geometry"]].rename(
                            columns={geom_col: join_key}
                        )
                        df = gpd.GeoDataFrame(
                            df.merge(geom_slim, on=join_key, how="left"),
                            geometry="geometry",
                            crs="EPSG:4326",
                        )
                        geom_pct = df["geometry"].notna().mean() * 100
                        console.print(
                            f"[dim]Geometry attached: {geom_pct:.0f}% coverage[/dim]"
                        )
                        if geom_pct < 90:
                            console.print(
                                f"[yellow]Warning: only {geom_pct:.0f}% of parcels have geometry. "
                                f"ArcGIS service may be incomplete for this county/year.[/yellow]"
                            )
            except Exception as e:
                console.print(f"[yellow]Geometry fetch failed (continuing without): {e}[/yellow]")

        # 4. Load into PostGIS
        try:
            n = load_parcels_to_postgis(df, county, year=yr, engine=engine)
            total_loaded += n
        except Exception as e:
            console.print(f"[red]DB load failed for {yr}: {e}[/red]")
            continue

    # Summary
    console.print(f"\n[bold green]Done![/bold green]")
    table = Table(title=f"Parcels loaded — {county.title()}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("County", county.title())
    table.add_row("Year(s)", ", ".join(str(y) for y in years))
    table.add_row("Total rows loaded", f"{total_loaded:,}")
    if with_geometry:
        table.add_row("Geometry", "attached" if geom_pct > 0 else "not available")
    console.print(table)


def _human_size(nbytes):
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


if __name__ == "__main__":
    cli()
