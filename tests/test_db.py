"""Basic tests for geostack database connectivity."""

import pytest


def test_import():
    """Verify the package is importable."""
    import geostack
    assert geostack.__version__ == "0.1.0"


def test_engine_requires_url():
    """get_engine should fail clearly if no DATABASE_URL is set."""
    import os
    from geostack.db import get_engine, _engine_cache

    # Clear cache and env
    _engine_cache.clear()
    old = os.environ.pop("DATABASE_URL", None)
    try:
        with pytest.raises(ValueError, match="DATABASE_URL not set"):
            get_engine()
    finally:
        if old:
            os.environ["DATABASE_URL"] = old


def test_census_variables():
    """Verify ACS variable dict has expected keys."""
    from geostack.census import ACS_VARIABLES

    assert "total_population" in ACS_VARIABLES
    assert "median_rent" in ACS_VARIABLES
    assert "median_household_income" in ACS_VARIABLES
