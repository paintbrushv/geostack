-- Parcel appraisal data (loaded from County Appraisal Districts)
-- Run: psql -d geostack -f db/migrations/003_parcels.sql
-- Loaded via: geostack load-parcels --county collin --year 2024

CREATE TABLE IF NOT EXISTS parcels (
    id                  SERIAL PRIMARY KEY,

    -- Identification
    parcel_id           TEXT NOT NULL,
    county              TEXT NOT NULL,
    state               TEXT NOT NULL DEFAULT 'tx',
    tax_year            INTEGER NOT NULL,

    -- Valuation (dollars)
    land_value          NUMERIC,
    improvement_value   NUMERIC,
    total_value         NUMERIC,
    assessed_value      NUMERIC,
    appraised_value     NUMERIC,

    -- Previous year values (for YoY delta)
    prev_land_value     NUMERIC,
    prev_improvement_value NUMERIC,
    prev_total_value    NUMERIC,

    -- Building attributes
    year_built          INTEGER,
    effective_year      INTEGER,
    improvement_class   TEXT,
    main_area_sqft      NUMERIC,

    -- Land attributes
    land_use_code       TEXT,
    land_acres          NUMERIC,
    land_sqft           NUMERIC,

    -- Classification
    property_type       TEXT,
    property_subtype    TEXT,
    property_category   TEXT,

    -- Ownership
    owner_name          TEXT,

    -- Location
    situs_address       TEXT,
    city                TEXT,
    zip_code            TEXT,
    neighborhood_code   TEXT,
    market_area_code    TEXT,

    -- Secondary IDs
    geo_id              TEXT,

    -- Computed fields (populated by geostack.analysis during load)
    ilr                 NUMERIC,    -- improvement_value / land_value
    age                 INTEGER,    -- tax_year - effective_year

    -- Spatial (populated when --with-geometry flag is used)
    geometry            GEOMETRY(MULTIPOLYGON, 4326),

    UNIQUE (parcel_id, county, state, tax_year)
);

CREATE INDEX IF NOT EXISTS idx_parcels_county_year  ON parcels(county, tax_year);
CREATE INDEX IF NOT EXISTS idx_parcels_state_county ON parcels(state, county);
CREATE INDEX IF NOT EXISTS idx_parcels_parcel_id    ON parcels(parcel_id);
CREATE INDEX IF NOT EXISTS idx_parcels_ilr          ON parcels(ilr) WHERE ilr IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_parcels_geom
    ON parcels USING GIST(geometry) WHERE geometry IS NOT NULL;
