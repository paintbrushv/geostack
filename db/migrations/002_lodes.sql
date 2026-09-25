-- LODES8 employment data tables
-- Run against the geostack database after 001_init.sql

-- WAC: Workplace Area Characteristics — jobs at each census block
CREATE TABLE IF NOT EXISTS lodes_wac (
    id SERIAL PRIMARY KEY,
    w_geocode VARCHAR(15) NOT NULL,     -- 15-digit census block (work location)
    state_fips VARCHAR(2) NOT NULL,
    county_fips VARCHAR(3) NOT NULL,
    tract_geoid VARCHAR(11) NOT NULL,   -- for joining to census_tracts.geoid
    year SMALLINT NOT NULL,
    job_type VARCHAR(4) NOT NULL DEFAULT 'JT01',  -- JT00=all, JT01=primary
    segment VARCHAR(4) NOT NULL DEFAULT 'S000',   -- S000=all, SE01-SE03=by earnings, etc.

    -- Totals
    total_jobs INTEGER,

    -- By age
    jobs_age_29_under INTEGER,
    jobs_age_30_54 INTEGER,
    jobs_age_55_over INTEGER,

    -- By earnings (monthly)
    jobs_earn_low INTEGER,              -- ≤ $1,250/mo
    jobs_earn_mid INTEGER,              -- $1,251–$3,333/mo
    jobs_earn_high INTEGER,             -- > $3,333/mo

    -- By NAICS sector
    jobs_agriculture INTEGER,
    jobs_mining_oil_gas INTEGER,
    jobs_utilities INTEGER,
    jobs_construction INTEGER,
    jobs_manufacturing INTEGER,
    jobs_wholesale_trade INTEGER,
    jobs_retail INTEGER,
    jobs_transport_warehousing INTEGER,
    jobs_information INTEGER,
    jobs_finance_insurance INTEGER,
    jobs_real_estate INTEGER,
    jobs_professional_scientific INTEGER,
    jobs_management INTEGER,
    jobs_admin_support INTEGER,
    jobs_education INTEGER,
    jobs_healthcare INTEGER,
    jobs_arts_entertainment INTEGER,
    jobs_accommodation_food INTEGER,
    jobs_other_services INTEGER,
    jobs_public_admin INTEGER,

    UNIQUE(w_geocode, year, job_type, segment)
);

CREATE INDEX IF NOT EXISTS idx_lodes_wac_county ON lodes_wac(state_fips, county_fips);
CREATE INDEX IF NOT EXISTS idx_lodes_wac_tract ON lodes_wac(tract_geoid);
CREATE INDEX IF NOT EXISTS idx_lodes_wac_year ON lodes_wac(year, job_type, segment);


-- RAC: Residential Area Characteristics — workers living at each census block
CREATE TABLE IF NOT EXISTS lodes_rac (
    id SERIAL PRIMARY KEY,
    h_geocode VARCHAR(15) NOT NULL,     -- 15-digit census block (home location)
    state_fips VARCHAR(2) NOT NULL,
    county_fips VARCHAR(3) NOT NULL,
    tract_geoid VARCHAR(11) NOT NULL,
    year SMALLINT NOT NULL,
    job_type VARCHAR(4) NOT NULL DEFAULT 'JT01',
    segment VARCHAR(4) NOT NULL DEFAULT 'S000',

    -- Totals
    total_workers INTEGER,

    -- By age
    workers_age_29_under INTEGER,
    workers_age_30_54 INTEGER,
    workers_age_55_over INTEGER,

    -- By earnings
    workers_earn_low INTEGER,
    workers_earn_mid INTEGER,
    workers_earn_high INTEGER,

    -- By industry of employment (NAICS)
    workers_in_agriculture INTEGER,
    workers_in_mining_oil_gas INTEGER,
    workers_in_utilities INTEGER,
    workers_in_construction INTEGER,
    workers_in_manufacturing INTEGER,
    workers_in_wholesale_trade INTEGER,
    workers_in_retail INTEGER,
    workers_in_transport_warehousing INTEGER,
    workers_in_information INTEGER,
    workers_in_finance_insurance INTEGER,
    workers_in_real_estate INTEGER,
    workers_in_professional_scientific INTEGER,
    workers_in_management INTEGER,
    workers_in_admin_support INTEGER,
    workers_in_education INTEGER,
    workers_in_healthcare INTEGER,
    workers_in_arts_entertainment INTEGER,
    workers_in_accommodation_food INTEGER,
    workers_in_other_services INTEGER,
    workers_in_public_admin INTEGER,

    UNIQUE(h_geocode, year, job_type, segment)
);

CREATE INDEX IF NOT EXISTS idx_lodes_rac_county ON lodes_rac(state_fips, county_fips);
CREATE INDEX IF NOT EXISTS idx_lodes_rac_tract ON lodes_rac(tract_geoid);
CREATE INDEX IF NOT EXISTS idx_lodes_rac_year ON lodes_rac(year, job_type, segment);


-- OD: Origin-Destination — home block → work block flows
CREATE TABLE IF NOT EXISTS lodes_od (
    id SERIAL PRIMARY KEY,
    w_geocode VARCHAR(15) NOT NULL,     -- work block
    h_geocode VARCHAR(15) NOT NULL,     -- home block
    w_state_fips VARCHAR(2),
    w_county_fips VARCHAR(3),
    h_state_fips VARCHAR(2),
    h_county_fips VARCHAR(3),
    year SMALLINT NOT NULL,
    job_type VARCHAR(4) NOT NULL DEFAULT 'JT01',

    total_jobs INTEGER,
    jobs_age_29_under INTEGER,
    jobs_age_30_54 INTEGER,
    jobs_age_55_over INTEGER,
    jobs_earn_low INTEGER,
    jobs_earn_mid INTEGER,
    jobs_earn_high INTEGER,
    jobs_goods_producing INTEGER,
    jobs_trade_transport_util INTEGER,
    jobs_other_services INTEGER,

    UNIQUE(w_geocode, h_geocode, year, job_type)
);

CREATE INDEX IF NOT EXISTS idx_lodes_od_work_county ON lodes_od(w_state_fips, w_county_fips);
CREATE INDEX IF NOT EXISTS idx_lodes_od_home_county ON lodes_od(h_state_fips, h_county_fips);
CREATE INDEX IF NOT EXISTS idx_lodes_od_year ON lodes_od(year, job_type);
