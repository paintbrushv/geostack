"""PostGIS database connection and session management.

Usage from any project:

    from geostack.db import get_engine, read_postgis, write_postgis

    engine = get_engine()  # reads DATABASE_URL from .env
    gdf = read_postgis("SELECT * FROM parcels WHERE ilr < 0.5", engine)
"""

import os

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

_engine_cache = {}


def get_engine(database_url=None):
    """Get or create a SQLAlchemy engine for PostGIS.

    Args:
        database_url: Override the DATABASE_URL env var. Useful when a project
                      needs to connect to a specific database (e.g., different databases on the same instance).
    """
    url = database_url or os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError(
            "DATABASE_URL not set. Copy .env.example to .env and configure it, "
            "or pass database_url explicitly."
        )
    if url not in _engine_cache:
        _engine_cache[url] = create_engine(url, pool_pre_ping=True)
    return _engine_cache[url]


def read_postgis(sql, engine=None, geom_col="geometry", **kwargs):
    """Read a PostGIS query into a GeoDataFrame.

    Thin wrapper around gpd.read_postgis that handles engine defaults.
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        return gpd.read_postgis(sql, conn, geom_col=geom_col, **kwargs)


def read_sql(sql, engine=None, params=None, **kwargs):
    """Read a SQL query into a plain DataFrame (no geometry)."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql) if params else sql, conn, params=params, **kwargs)


def write_postgis(gdf, table_name, engine=None, if_exists="replace", schema=None):
    """Write a GeoDataFrame to PostGIS.

    Args:
        gdf: GeoDataFrame to write.
        table_name: Target table name.
        engine: SQLAlchemy engine. Defaults to get_engine().
        if_exists: 'replace', 'append', or 'fail'.
        schema: Optional schema name.
    """
    engine = engine or get_engine()
    gdf.to_postgis(table_name, engine, if_exists=if_exists, schema=schema)


def execute_sql(sql, engine=None, params=None):
    """Execute raw SQL (for DDL, inserts, etc.)."""
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})
