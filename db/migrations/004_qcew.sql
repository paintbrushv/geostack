-- QCEW (Quarterly Census of Employment and Wages) annual data
-- County-level employment and wages by NAICS sector
-- Source: BLS QCEW program (https://www.bls.gov/cew/)

CREATE TABLE IF NOT EXISTS qcew_annual (
    id SERIAL PRIMARY KEY,
    state_fips VARCHAR(2) NOT NULL,
    county_fips VARCHAR(3) NOT NULL,
    year SMALLINT NOT NULL,
    naics_code VARCHAR(6) NOT NULL,
    naics_title TEXT,
    establishment_count INTEGER,
    annual_avg_employment INTEGER,
    total_annual_wages BIGINT,
    avg_weekly_wage INTEGER,
    avg_annual_pay INTEGER,
    UNIQUE(state_fips, county_fips, year, naics_code)
);

CREATE INDEX IF NOT EXISTS idx_qcew_county_year
    ON qcew_annual(state_fips, county_fips, year);

CREATE INDEX IF NOT EXISTS idx_qcew_naics
    ON qcew_annual(naics_code, year);
