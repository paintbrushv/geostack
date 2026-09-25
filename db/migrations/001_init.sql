-- Geostack base schema
-- Run against the geostack database on PG17

CREATE EXTENSION IF NOT EXISTS postgis;

-- Census tract boundaries (loaded from TIGER/Line)
CREATE TABLE IF NOT EXISTS census_tracts (
    gid SERIAL PRIMARY KEY,
    geoid VARCHAR(11) UNIQUE NOT NULL,  -- state(2) + county(3) + tract(6)
    state_fips VARCHAR(2) NOT NULL,
    county_fips VARCHAR(3) NOT NULL,
    name VARCHAR(100),
    aland BIGINT,       -- land area (sq meters)
    awater BIGINT,      -- water area (sq meters)
    geometry GEOMETRY(MULTIPOLYGON, 4326)
);

CREATE INDEX IF NOT EXISTS idx_tracts_geoid ON census_tracts(geoid);
CREATE INDEX IF NOT EXISTS idx_tracts_geom ON census_tracts USING GIST(geometry);
CREATE INDEX IF NOT EXISTS idx_tracts_county ON census_tracts(state_fips, county_fips);

-- ACS demographic time series
CREATE TABLE IF NOT EXISTS acs_tract_data (
    id SERIAL PRIMARY KEY,
    geoid VARCHAR(11) NOT NULL REFERENCES census_tracts(geoid),
    year INTEGER NOT NULL,
    variable VARCHAR(20) NOT NULL,
    value NUMERIC,
    UNIQUE(geoid, year, variable)
);

CREATE INDEX IF NOT EXISTS idx_acs_geoid_year ON acs_tract_data(geoid, year);
CREATE INDEX IF NOT EXISTS idx_acs_variable ON acs_tract_data(variable);

-- Road centerlines (loaded from TIGER/Line)
CREATE TABLE IF NOT EXISTS roads (
    gid SERIAL PRIMARY KEY,
    linearid VARCHAR(22),
    fullname VARCHAR(100),
    rttyp VARCHAR(1),       -- route type (I=interstate, U=US, S=state, M=county, etc.)
    mtfcc VARCHAR(5),       -- MAF/TIGER feature class code
    state_fips VARCHAR(2),
    county_fips VARCHAR(3),
    geometry GEOMETRY(LINESTRING, 4326)
);

CREATE INDEX IF NOT EXISTS idx_roads_geom ON roads USING GIST(geometry);
CREATE INDEX IF NOT EXISTS idx_roads_county ON roads(state_fips, county_fips);
CREATE INDEX IF NOT EXISTS idx_roads_name ON roads(fullname);
