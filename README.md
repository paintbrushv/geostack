# geostack

A shared geospatial analytics library for commercial real estate analysis:
PostGIS database management, Census/ACS ingestion, TIGER/Line shapefile
handling, H3 hexagonal grids, drive-time isochrones, county appraisal-district
(CAD) utilities, and static/interactive mapping — all as a pip-installable
Python package.

> **Validation status:** this package ships with a unit test suite that runs
> fully offline (`pytest -m "not integration"`). The integration-marked tests
> require a live PostGIS database and/or network access to public data APIs
> (Census Bureau, OSRM, county ArcGIS services) and are deselected by default.
> All unit-test fixtures are **synthetic**; no real property, deal, or client
> data is included in this repository.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # core + test tooling
pip install -e ".[all]"       # every optional extra (viz, h3, routing, ...)
```

PostGIS is required for the database features: create a database (e.g.
`createdb geostack && psql -d geostack -c "CREATE EXTENSION postgis"`), then
copy `.env.example` to `.env` and set `DATABASE_URL`.

## Usage

```bash
geostack db-status
geostack download-tracts --state 48 --year 2023 --load-db
geostack download-roads --state 48 --county 085 --year 2023 --load-db
```

```python
from geostack.db import get_engine
from geostack.census import get_tract_demographics
from geostack.h3grid import points_to_h3
```

## Repository layout

```
src/geostack/    # the pip-installable library
db/migrations/   # SQL schema files
scripts/         # example pipelines and exhibits (public data sources only)
notebooks/       # Jupyter exploration
tests/           # pytest suite (offline; integration tests are marked)
data/            # download cache (gitignored, populated by scripts)
```

## Tests

```bash
pytest -m "not integration"
```

Recommended: run with a private basetemp (`umask 077`) since pytest tmp dirs
default to a shared `/tmp`. Optional-dependency tests (h3, duckdb) are skipped
with a visible reason when the extra is not installed.

## License

Apache License 2.0 — see [LICENSE](LICENSE).