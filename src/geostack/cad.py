"""County Appraisal District (CAD) utilities.

Helpers for fetching, normalizing, and loading parcel data from Texas county
appraisal districts. Each county has different data formats — this module
provides a common interface.

Supported counties:
  - Collin County   — Socrata (data.texas.gov) + ArcGIS REST
  - Dallas County   — DCAD multi-CSV bulk download (dallascad.org)
  - Tarrant County  — TAD pipe-delimited bulk download (tad.org)
  - Cooke County    — True Prodigy API (CCAD; full roll + lat/lon, no geocode)
  - Travis County   — Socrata (data.austintexas.gov) + ArcGIS REST
  - Grayson County  — Socrata (data.texas.gov) + ArcGIS REST
  - Wise County     — PACS fixed-width bulk download (wise-cad.com)
  - Denton County   — PACS fixed-width bulk download (dentoncad.net)

Usage:
    from geostack.cad import fetch_parcels, normalize_parcels, load_parcels_to_postgis

    # Fetch full dataset with pagination
    df = fetch_parcels("collin", year=2024)

    # Normalize to canonical ParcelYear schema + compute ILR/age
    df = normalize_parcels(df, "collin")

    # Load into PostGIS (DELETE + re-insert for county+year)
    load_parcels_to_postgis(df, "collin", year=2024, engine=engine)
"""

import csv
import json
import os
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from rich.console import Console

TXGIO_STRATMAP_URL = (
    "https://services1.arcgis.com/1mtXwieMId59thmg/arcgis/rest/services/"
    "2019_Texas_Parcels_StratMap/FeatureServer/0/query"
)
TIGER_COUNTY_URL = "https://www2.census.gov/geo/tiger/TIGER{year}/COUNTY/tl_{year}_us_county.zip"

console = Console()

# ---------------------------------------------------------------------------
# Collin County CAD — Socrata
# ---------------------------------------------------------------------------

# Year-specific Socrata dataset IDs for Collin CAD (data.texas.gov)
# Each year is a separate certified appraisal roll dataset.
COLLIN_DATASET_IDS: dict[int, str] = {
    2024: "6dqt-e958",
    2023: "khef-anha",
    2022: "vtby-uz4n",
    2021: "cwex-zmku",
    2020: "hkzz-rrn8",
}

# Core columns for ILR analysis — fetch subset to reduce download size ~75%
COLLIN_CORE_COLUMNS: list[str] = [
    # Identification
    "propid", "geoid", "propyear",
    # Current year values
    "currvalland", "currvalimprv", "currvalmarket",
    # Previous year values (for YoY delta)
    "prevvalland", "prevvalimprv", "prevvalmarket", "prevvalyear",
    # Classification
    "proptype", "propsubtype", "propcategorycode",
    # Ownership
    "ownername",
    # Owner mailing address (Socrata fields; distinct from situs/property address)
    "owneraddrline1", "owneraddrcity", "owneraddrstate", "owneraddrzip",
    # Location
    "situsconcat", "situscity", "situszip",
    # Building
    "imprvyearbuilt", "imprvmainarea",
    # Geographic
    "nbhdcode",
    # Land
    "landsizeacres",
    # Legal description (subdivision / lot / block — for parcel joins)
    "legalabssubname", "legalabssublot", "legalabssubblock",
    "legalabssubcode", "legaldescription",
]

# Column mapping: raw Collin CAD Socrata → canonical ParcelYear schema
_COLLIN_COLUMN_MAP: dict[str, str] = {
    "propid": "parcel_id",
    "geoid": "geo_id",
    "propyear": "tax_year",
    "currvalland": "land_value",
    "currvalimprv": "improvement_value",
    "currvalmarket": "total_value",
    "currvalassessed": "assessed_value",
    "currvalappraised": "appraised_value",
    "prevvalland": "prev_land_value",
    "prevvalimprv": "prev_improvement_value",
    "prevvalmarket": "prev_total_value",
    "imprvyearbuilt": "year_built",
    "imprvclasscd": "improvement_class",
    "imprvmainarea": "main_area_sqft",
    "landsizeacres": "land_acres",
    "landsizesqft": "land_sqft",
    "proptype": "property_type",
    "propsubtype": "property_subtype",
    # PTAD state category code (A, A1, B, C1, D1, D2, F1, M1, …) — the Texas
    # property-class code the redemption engine classifies from. Already present
    # in COLLIN_CORE_COLUMNS. Mapped to canonical state_use_code for parity with
    # Denton's imprv_state_cd → state_use_code (property_category was unused).
    "propcategorycode": "state_use_code",
    "propusecode": "land_use_code",
    "ownername": "owner_name",
    # Owner mailing address — Socrata field names (distinct from situs/property address)
    # Verified live: owneraddrline1 differs from situsconcat on ~30%+ of Collin parcels.
    "owneraddrline1": "mail_addr1",
    "owneraddrcity":  "mail_city",
    "owneraddrstate": "mail_state",
    "owneraddrzip":   "mail_zip",
    "situsconcat": "situs_address",
    "situscity": "city",
    "situszip": "zip_code",
    "nbhdcode": "neighborhood_code",
    "marketareacode": "market_area_code",
    "legalabssubname": "subdivision",
    "legalabssublot": "lot",
    "legalabssubblock": "block",
    "legalabssubcode": "legal_code",
    "legaldescription": "legal_description",
}

# ---------------------------------------------------------------------------
# Dallas County CAD (DCAD) — Socrata
# ---------------------------------------------------------------------------

# NOTE: This Socrata URL is Dallas COUNTY government, NOT Dallas CAD (DCAD).
# DCAD uses bulk CSV downloads from dallascad.org/dataproducts.aspx
# TODO: Replace with DCAD bulk downloader (see dallascad.org/dataproducts.aspx)
DALLAS_SOCRATA_DATASET = "https://data.dallascounty.org/resource/5jyq-vp7y.json"  # WRONG — county, not CAD
DALLAS_GIS_URL = (
    "https://services2.arcgis.com/rwnOSbfKSwyTBcwN/arcgis/rest/services/"
    "CurrentDcadParcels/FeatureServer/0/query"
)
# NOTE: Old URL (gis.dallascountytx.gov) is dead. This ArcGIS Online endpoint
# has geometry only (GIS_Acct field matches bulk account_num).

# Best-effort column mapping for Dallas CAD
_DALLAS_COLUMN_MAP: dict[str, str] = {
    "account_num": "parcel_id",
    "tax_year": "tax_year",
    "land_value": "land_value",
    "improvement_value": "improvement_value",
    "total_value": "total_value",
    "year_built": "year_built",
    "owner_name": "owner_name",
    "situs_address": "situs_address",
    "situs_city": "city",
    "situs_zip": "zip_code",
    "property_type": "property_type",
    "land_use_code": "land_use_code",
    "land_acres": "land_acres",
}

# ---------------------------------------------------------------------------
# Cooke County CAD — True Prodigy (primary; Socrata mgnk-vbdk dead as of 2025)
# ---------------------------------------------------------------------------

# NOTE: The original Socrata dataset https://data.texas.gov/resource/mgnk-vbdk.json
# returned 404 as of mid-2025. fetch_cooke_parcels() now routes to the True Prodigy
# path (fetch_cooke_parcels_trueprodigy), which returns ~30k parcels with lat/lon.
# The Socrata constant is retained for historical reference only.
COOKE_SOCRATA_DATASET = "https://data.texas.gov/resource/mgnk-vbdk.json"  # DEAD — 404 since 2025
COOKE_GIS_URL = (
    "https://gis.co.cooke.tx.us/arcgis/rest/services/"
    "Public/Parcels/MapServer/0/query"
)

# Column map for the True Prodigy path (raw API field names → canonical ParcelYear schema).
# These are the raw True Prodigy JSON field names returned by fetch_cooke_parcels(normalize=False).
_COOKE_COLUMN_MAP: dict[str, str] = {
    "pAccountID":       "parcel_id",
    "pYear":            "tax_year",
    "landValue":        "land_value",
    "improvementValue": "improvement_value",
    "appraisedValue":   "total_value",
    "marketValue":      "market_value",
    "name":             "owner_name",
    # Owner mailing address — True Prodigy JSON field names (distinct from property situs)
    # Verified live: addrDeliveryLine often differs from fullSitus on Cooke parcels.
    "addrDeliveryLine": "mail_addr1",
    "addrCity":         "mail_city",
    "addrState":        "mail_state",
    "addrZip":          "mail_zip",
    "fullSitus":        "situs_address",
    "city":             "city",
    "zip":              "zip_code",
    "propType":         "property_type",
    "useCd":            "land_use_code",
    "legalAcreage":     "land_acres",
    "latitude":         "latitude",
    "longitude":        "longitude",
    "geoID":            "geo_id",
}

# ---------------------------------------------------------------------------
# Travis County CAD (TCAD — Austin metro) — Socrata
# ---------------------------------------------------------------------------

TCAD_SOCRATA_DATASET = "https://data.austintexas.gov/resource/i7d6-pn5j.json"
TCAD_GIS_URL = (
    "https://maps.traviscountytx.gov/arcgis/rest/services/"
    "Parcels/MapServer/0/query"
)

_TCAD_COLUMN_MAP: dict[str, str] = {
    "prop_id": "parcel_id",
    "tax_year": "tax_year",
    "land_value": "land_value",
    "improvement_value": "improvement_value",
    "total_value": "total_value",
    "year_built": "year_built",
    "owner_name": "owner_name",
    "situs_address": "situs_address",
    "situs_city": "city",
    "situs_zip": "zip_code",
    "property_type": "property_type",
    "land_use_code": "land_use_code",
    "land_acres": "land_acres",
}

# ---------------------------------------------------------------------------
# Grayson County CAD (Sherman/Denison area) — PACS bulk download
# ---------------------------------------------------------------------------

# NOTE: The original Socrata dataset https://data.texas.gov/resource/e4rz-6rb7.json
# returned 404 as of mid-2025. Grayson CAD publishes its own PACS 8.0.31 export at
# graysonappraisal.org/public-information/ — same fixed-width format as Wise/Denton.
# The Socrata constant is retained for historical reference only.
GRAYSON_SOCRATA_DATASET = "https://data.texas.gov/resource/e4rz-6rb7.json"  # DEAD — 404 since 2025

# Live PACS bulk export from graysonappraisal.org/public-information/
# This is updated with each certification supplement; last seen: 2025 Sup 25 (Feb 2026)
GRAYSON_EXPORT_URL = "https://maps.graysonappraisal.org/ORR_Export/GCAD_Export.zip"

GRAYSON_GIS_URL = (
    "https://gis.co.grayson.tx.us/arcgis/rest/services/"
    "Public/Parcels/MapServer/0/query"
)

# Column mapping: PACS-derived field names → canonical ParcelYear schema
# (identical layout to Wise/Denton PACS 8.0.30/8.0.31; same field positions)
_GRAYSON_COLUMN_MAP: dict[str, str] = {
    "prop_id":           "parcel_id",
    "prop_val_yr":       "tax_year",
    "prop_type_cd":      "property_type",
    "geo_id":            "geo_id",
    "py_owner_name":     "owner_name",
    # Owner mailing address (PACS py_addr_* block; parsed in _parse_grayson_appraisal_info)
    # Empirically verified from live GCAD_Export.zip — situs often differs from mail.
    "mail_addr1":        "mail_addr1",
    "mail_city":         "mail_city",
    "mail_state":        "mail_state",
    "mail_zip":          "mail_zip",
    "situs_address":     "situs_address",
    "situs_city":        "city",
    "situs_zip":         "zip_code",
    "hood_cd":           "neighborhood_code",
    "land_value":        "land_value",
    "improvement_value": "improvement_value",
    "appraised_val":     "total_value",
    "assessed_val":      "assessed_value",
    "legal_acreage":     "land_acres",
    "yr_built":          "year_built",
    "main_area_sqft":    "main_area_sqft",
    "imprv_state_cd":    "state_use_code",  # PTAD state category code (already parsed; see _parse_grayson_appraisal_info)
    "land_state_cd":     "land_state_cd",  # fallback PTAD code when improvement state is blank
}

# ---------------------------------------------------------------------------
# Tarrant Appraisal District (TAD) — pipe-delimited bulk download
# ---------------------------------------------------------------------------

# TAD publishes pipe-delimited property data at tad.org/resources/data-downloads
# Commercial-only file is most useful for multifamily screening
TAD_DATASET_URLS: dict[int, str] = {
    2025: "https://www.tad.org/content/data-download/PropertyData_C_2025(Certified).ZIP",
    2024: "https://www.tad.org/content/data-download/PropertyData_C_2024(Certified).ZIP",
}
TAD_LOCATION_URL = "https://www.tad.org/content/data-download/PropertyLocation(Delimited).ZIP"
# All-property (residential + commercial + land + mineral) certified bulk file —
# the file NPL needs, since foreclosure collateral is mostly residential.
# NOTE 1: TAD serves these only to a browser User-Agent; a bare python-requests UA
#   gets HTTP 403 (verified 2026-06-29). Downloads MUST send _BULK_HEADERS.
# NOTE 2: the 2025 all-property file is published TRUNCATED at the source (no zip
#   central directory; the deflate stream cuts off after ~915k mineral rows, before
#   the residential rows), so it is omitted until TAD republishes a complete export.
#   2024 is the latest intact all-property file (78 MB, ~1.93M rows, 651k real-property).
TAD_FULL_URLS: dict[int, str] = {
    2024: "https://www.tad.org/content/data-download/PropertyData_2024(Certified).ZIP",
}

# Some CAD bulk hosts (TAD, DCAD) reject the default python-requests User-Agent
# with 403. Send a browser UA for all bulk ZIP downloads.
_BULK_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Column mapping: raw TAD pipe-delimited fields → canonical ParcelYear schema
_TARRANT_COLUMN_MAP: dict[str, str] = {
    "account_num": "parcel_id",
    "owner_name": "owner_name",
    "situs_address": "situs_address",
    "property_class": "property_category",
    "state_use_code": "property_type",
    "year_built": "year_built",
    "living_area": "main_area_sqft",
    "land_value": "land_value",
    "improvement_value": "improvement_value",
    "total_value": "total_value",
    "appraised_value": "assessed_value",
    "land_acres": "land_acres",
    "deed_date": "deed_date",
    "gis_link": "geo_id",
    "city": "city",
}

# ---------------------------------------------------------------------------
# Dallas Central Appraisal District (DCAD) — CSV bulk download
# ---------------------------------------------------------------------------

# DCAD publishes CSV ZIPs at dallascad.org/dataproducts.aspx
# NOTE: NOT the same as data.dallascounty.org (that's county government, not CAD)
DCAD_DATASET_URLS: dict[int, str] = {
    2025: "https://www.dallascad.org/ViewPDFs.aspx?type=3&id=\\\\DCAD.ORG\\WEB\\WEBDATA\\WEBFORMS\\DATA PRODUCTS\\DCAD2025_CERTIFIED_07242025.zip",
    2024: "https://www.dallascad.org/ViewPDFs.aspx?type=3&id=\\\\DCAD.ORG\\WEB\\WEBDATA\\WEBFORMS\\DATA PRODUCTS\\DCAD2024_CERTIFIED_07252024.zip",
}

# Column mapping: raw DCAD CSV fields → canonical ParcelYear schema
_DCAD_BULK_COLUMN_MAP: dict[str, str] = {
    "account_num": "parcel_id",
    "owner_name1": "owner_name",
    # Owner mailing address — DCAD CSV bulk field names (distinct from property situs)
    # Verified live: owner_address_line1 often differs from full_street_name+property_city.
    "owner_address_line1": "mail_addr1",
    "owner_address_line2": "mail_addr2",   # care-of / attn / suite (often blank)
    "owner_city":          "mail_city",
    "owner_state":         "mail_state",
    "owner_zipcode":       "mail_zip",
    "street_num": "street_num",
    "full_street_name": "situs_address",
    "property_city": "city",
    "property_zipcode": "zip_code",
    "sptd_code": "state_use_code",  # SPTD = State Property Tax Division use code; DCAD's name for the PTAD state_use_code (Collin: propcategorycode, Tarrant/Grayson: imprv_state_cd). The npl property_class classifier + backfill_cad_state_use read state_use_code; without this Dallas parcels stayed 0-coded. (Previously mis-mapped to property_type; DCAD BPP/commercial is filtered geostack-side via the B-prefix sptd_code filter, not npl's _bpp_mask == "P".)
    "division_cd": "property_category",
    "year_built": "year_built",
    "gross_bldg_area": "main_area_sqft",
    "num_units": "num_units",
    "property_name": "property_name",
    "land_val": "land_value",
    "impr_val": "improvement_value",
    "tot_val": "total_value",
    "gis_parcel_id": "geo_id",
    "nbhd_cd": "neighborhood_code",
}

# ---------------------------------------------------------------------------
# Wise County CAD — PACS fixed-width bulk download
# ---------------------------------------------------------------------------

# Year → direct ZIP URL (certified all-property roll, nested-ZIP PACS format)
WISE_DATASET_URLS: dict[int, str] = {
    2025: "https://wise-cad.com/wp-content/uploads/2025/07/Certified-Roll-0_2025_07_23.zip",
    2024: "https://wise-cad.com/wp-content/uploads/2024/08/2024-Certified-Data-All-Property.zip",
    2023: "https://wise-cad.com/wp-content/uploads/2023/08/2023-Certified-Data-All-Property.zip",
    2022: "https://wise-cad.com/wp-content/uploads/2022/08/2022-ALL-PROPERTY-ALL-FILES-.zip",
}

# APPRAISAL_INFO.TXT field positions (Python 0-indexed slices, record width 9,249)
# Source of truth: "Appraisal Export Layout - 8.0.31.xlsx" (inside GCAD_Export.zip).
# PACS spec uses 1-indexed end-inclusive positions; Python slices are [start-1:end].
_AI_PROP_ID        = slice(0, 12)
_AI_PROP_TYPE      = slice(12, 17)
_AI_PROP_VAL_YR    = slice(17, 22)
_AI_SUP_NUM        = slice(22, 34)
_AI_GEO_ID         = slice(546, 596)
_AI_OWNER_NAME     = slice(608, 678)
# Owner mailing address block (py_addr_* in the PACS 8.0.31 spec):
#   py_addr_line1 (694-753, 60 chars): optional attention / care-of / PO Box line
#   py_addr_line2 (754-813, 60 chars): street address (primary delivery line)
#   py_addr_city  (874-923, 50 chars): city
#   py_addr_state (924-973, 50 chars): state abbreviation
#   py_addr_zip   (979-983,  5 chars): ZIP code (5-digit only; +4 is at 984-987)
# Empirically verified against live Grayson 2025 export (GCAD_Export.zip).
# For assembly: prefer _AI_MAIL_ADDR2 (street); fall back to _AI_MAIL_ADDR1 (PO Box).
_AI_MAIL_ADDR1     = slice(693, 753)   # py_addr_line1: care-of / PO Box (often blank)
_AI_MAIL_ADDR2     = slice(753, 813)   # py_addr_line2: street address (primary)
_AI_MAIL_CITY      = slice(873, 923)   # py_addr_city
_AI_MAIL_STATE     = slice(923, 973)   # py_addr_state
_AI_MAIL_ZIP       = slice(978, 983)   # py_addr_zip (5-digit)
_AI_SITUS_PREFX    = slice(1039, 1049)
_AI_SITUS_STREET   = slice(1049, 1099)
_AI_SITUS_SUFFIX   = slice(1099, 1109)
_AI_SITUS_CITY     = slice(1109, 1139)
_AI_SITUS_ZIP      = slice(1139, 1149)
_AI_SITUS_NUM      = slice(4460, 4475)
_AI_LEGAL_ACREAGE  = slice(1659, 1675)
_AI_HOOD_CD        = slice(1685, 1695)
_AI_LAND_HSTD      = slice(1795, 1810)
_AI_LAND_NON_HSTD  = slice(1810, 1825)
_AI_IMPRV_HSTD     = slice(1825, 1840)
_AI_IMPRV_NON_HSTD = slice(1840, 1855)
_AI_APPRAISED_VAL  = slice(1915, 1930)
_AI_ASSESSED_VAL   = slice(1945, 1960)
_AI_IMPRV_STATE_CD = slice(2731, 2741)  # e.g. "B1" = apartments, "B2" = duplexes
_AI_LAND_STATE_CD  = slice(2741, 2751)  # land-only parcels often carry use code here
_GRAYSON_CACHE_REQUIRED_COLUMNS = frozenset({"land_state_cd"})

# APPRAISAL_IMPROVEMENT_DETAIL.TXT field positions (record width 624)
_AID_PROP_ID       = slice(0, 12)
_AID_PROP_VAL_YR   = slice(12, 16)
_AID_TYPE_CD       = slice(40, 50)
_AID_YR_BUILT      = slice(85, 89)
_AID_AREA          = slice(93, 108)

# Column mapping: raw PACS field names → canonical ParcelYear schema
_WISE_COLUMN_MAP: dict[str, str] = {
    "prop_id":           "parcel_id",
    "prop_val_yr":       "tax_year",
    "prop_type_cd":      "property_type",
    "geo_id":            "geo_id",
    "py_owner_name":     "owner_name",
    # Owner mailing address (PACS py_addr_* block; same layout as Grayson/Denton)
    "mail_addr1":        "mail_addr1",
    "mail_city":         "mail_city",
    "mail_state":        "mail_state",
    "mail_zip":          "mail_zip",
    "situs_address":     "situs_address",
    "situs_city":        "city",
    "situs_zip":         "zip_code",
    "hood_cd":           "neighborhood_code",
    "land_value":        "land_value",
    "improvement_value": "improvement_value",
    "appraised_val":     "total_value",
    "assessed_val":      "assessed_value",
    "legal_acreage":     "land_acres",
    "yr_built":          "year_built",
    "main_area_sqft":    "main_area_sqft",
}

# ---------------------------------------------------------------------------
# Denton County CAD — PACS fixed-width bulk download (dentoncad.net)
# ---------------------------------------------------------------------------

# Denton CAD publishes PACS fixed-width data at dentoncad.net.
# Individual TXT files are available directly; a combined ZIP is also provided.
# Record layouts are identical to Wise County (PACS v8.0.30): APPRAISAL_INFO.TXT
# (9,249 chars/record), APPRAISAL_IMPROVEMENT_DETAIL.TXT (624 chars/record).
#
# Base URL: https://dentoncad.net/data/_uploaded/files/datafiles/{year}/CertifiedDataAllProperty/

# Year → direct ZIP URL (certified all-property roll, PACS format)
DENTON_DATASET_URLS: dict[int, str] = {
    2025: "https://dentoncad.net/data/_uploaded/files/datafiles/2025/CertifiedDataAllProperty/Denton_AppraisalDataExtracts_ALL(CERTIFIED).zip",
    2024: "https://dentoncad.net/data/_uploaded/files/datafiles/2024/CertifiedDataAllProperty/CertifiedDataAll.zip",
}

# Year → individual APPRAISAL_INFO.TXT URL (alternative to ZIP download)
DENTON_APPRAISAL_INFO_URLS: dict[int, str] = {
    2024: "https://dentoncad.net/data/_uploaded/files/datafiles/2024/CertifiedDataAllProperty/2024-07-22_008084_APPRAISAL_INFO.TXT",
}

# Year → individual APPRAISAL_IMPROVEMENT_DETAIL.TXT URL
DENTON_IMPROVEMENT_DETAIL_URLS: dict[int, str] = {
    2024: "https://dentoncad.net/data/_uploaded/files/datafiles/2024/CertifiedDataAllProperty/2024-07-22_008084_APPRAISAL_IMPROVEMENT_DETAIL.TXT",
}

# Column mapping: identical to Wise County (same PACS layout)
_DENTON_COLUMN_MAP: dict[str, str] = {
    "prop_id":           "parcel_id",
    "prop_val_yr":       "tax_year",
    "prop_type_cd":      "property_type",
    "geo_id":            "geo_id",
    "py_owner_name":     "owner_name",
    # Owner mailing address (PACS py_addr_* block; same layout as Grayson/Wise)
    "mail_addr1":        "mail_addr1",
    "mail_city":         "mail_city",
    "mail_state":        "mail_state",
    "mail_zip":          "mail_zip",
    "situs_address":     "situs_address",
    "situs_city":        "city",
    "situs_zip":         "zip_code",
    "hood_cd":           "neighborhood_code",
    "land_value":        "land_value",
    "improvement_value": "improvement_value",
    "appraised_val":     "total_value",
    "assessed_val":      "assessed_value",
    "legal_acreage":     "land_acres",
    "yr_built":          "year_built",
    "main_area_sqft":    "main_area_sqft",
    "imprv_state_cd":    "state_use_code",
}

# ---------------------------------------------------------------------------
# County registry — maps county name to metadata and fetch/geometry functions
# ---------------------------------------------------------------------------

# Lookup table: county_name → FIPS codes, Socrata URL, sort key, GIS URL, column map
COUNTY_REGISTRY: dict[str, dict] = {
    "collin": {
        "state_fips": "48",
        "county_fips": "085",
        "state": "tx",
        "socrata_url": None,  # uses COLLIN_DATASET_IDS[year] dynamically
        "dataset_ids": COLLIN_DATASET_IDS,
        "sort_key": "prop_id",
        "gis_url": "https://gis.collincountytx.gov/arcgis/rest/services/Public/Parcels/MapServer/0/query",
        "geom_id_field": "PROP_ID",
        "column_map": _COLLIN_COLUMN_MAP,
        "core_columns": COLLIN_CORE_COLUMNS,
    },
    "dallas": {
        "state_fips": "48",
        "county_fips": "113",
        "state": "tx",
        "socrata_url": DALLAS_SOCRATA_DATASET,  # DEPRECATED — this is county gov, not DCAD
        "dataset_ids": DCAD_DATASET_URLS,  # Correct: bulk CSV from dallascad.org
        "sort_key": "account_num",
        "gis_url": DALLAS_GIS_URL,
        "geom_id_field": "ACCOUNT_NUM",
        "column_map": _DCAD_BULK_COLUMN_MAP,
        "core_columns": None,
        "bulk_download": True,
        "format": "csv",  # DCAD's own CSV layout (not PACS, not Socrata)
    },
    "cooke": {
        "state_fips": "48",
        "county_fips": "097",
        "state": "tx",
        "socrata_url": None,       # dead — uses fetch_cooke_parcels_trueprodigy via fetch_parcels
        "dataset_ids": None,
        "sort_key": "parcel_id",
        "gis_url": COOKE_GIS_URL,
        "geom_id_field": "PROP_ID",
        "column_map": _COOKE_COLUMN_MAP,
        "core_columns": None,
        "trueprodigy": True,       # True Prodigy PACS portal (requires Playwright headless)
    },
    "travis": {
        "state_fips": "48",
        "county_fips": "453",
        "state": "tx",
        "socrata_url": TCAD_SOCRATA_DATASET,
        "dataset_ids": None,
        "sort_key": "prop_id",
        "gis_url": TCAD_GIS_URL,
        "geom_id_field": "PROP_ID",
        "column_map": _TCAD_COLUMN_MAP,
        "core_columns": None,
    },
    "grayson": {
        "state_fips": "48",
        "county_fips": "181",
        "state": "tx",
        "socrata_url": None,       # dead — uses GRAYSON_EXPORT_URL via fetch_grayson_parcels
        "dataset_ids": {"current": GRAYSON_EXPORT_URL},
        "sort_key": "prop_id",
        "gis_url": GRAYSON_GIS_URL,
        "geom_id_field": "PROP_ID",
        "column_map": _GRAYSON_COLUMN_MAP,
        "core_columns": None,
        "pacs": True,              # PACS fixed-width format; property_type needs code → label mapping
    },
    "tarrant": {
        "state_fips": "48",
        "county_fips": "439",
        "state": "tx",
        "socrata_url": None,
        "dataset_ids": TAD_FULL_URLS,   # all-property (residential incl.); commercial-only is TAD_DATASET_URLS
        "sort_key": "account_num",
        "gis_url": "https://gis-tad.opendata.arcgis.com/datasets",  # ArcGIS Hub
        "geom_id_field": "ACCOUNT_NUM",
        "column_map": _TARRANT_COLUMN_MAP,
        "core_columns": None,
        "bulk_download": True,
        "format": "pipe_delimited",
    },
    # NOTE (2026-07-06): feature.tnris.org is DEAD (NXDOMAIN) — TNRIS rebranded to
    # TxGIO (~2023). The canonical successor set below is the TxGIO StratMap 2025
    # parcels MapServer (has fips/geo_id). CAVEAT: that MapServer's REST /query
    # returns "operation not supported" from a standard client, and the queryable
    # ArcGIS-Online mirror
    # services1.arcgis.com/1mtXwieMId59thmg/.../2019_Texas_Parcels_StratMap/FeatureServer/0
    # (layer Stratmap25_landparcels_48) has NO fips field. So geostack's
    # _arcgis_geojson_fetch (offset pagination + FIPS where) needs a bbox/keyset
    # follow-up before wise/denton geometry fetch works against TxGIO.
    "wise": {
        "state_fips": "48",
        "county_fips": "497",
        "state": "tx",
        "socrata_url": None,       # uses WISE_DATASET_URLS[year] via fetch_wise_parcels
        "dataset_ids": WISE_DATASET_URLS,
        "sort_key": "prop_id",
        # TxGIO StratMap statewide parcel service. The services1 mirror is
        # queryable but has no FIPS column, so geometry fetch uses TIGER bbox
        # + OBJECTID keyset paging, then county-boundary centroid filtering.
        "gis_url": TXGIO_STRATMAP_URL,
        "geom_id_field": "geo_id",   # TxGIO geo_id → parcel_geo.geo_id join key
        "geom_fetch": "txgio_bbox_keyset",
        "column_map": _WISE_COLUMN_MAP,
        "core_columns": None,
        "pacs": True,              # PACS fixed-width format; property_type needs code → label mapping
    },
    "denton": {
        "state_fips": "48",
        "county_fips": "121",
        "state": "tx",
        "socrata_url": None,       # uses DENTON_DATASET_URLS[year] via fetch_denton_parcels
        "dataset_ids": DENTON_DATASET_URLS,
        "sort_key": "prop_id",
        # TxGIO StratMap statewide parcel service. The services1 mirror is
        # queryable but has no FIPS column, so geometry fetch uses TIGER bbox
        # + OBJECTID keyset paging, then county-boundary centroid filtering.
        "gis_url": TXGIO_STRATMAP_URL,
        "geom_id_field": "geo_id",
        "geom_fetch": "txgio_bbox_keyset",
        "column_map": _DENTON_COLUMN_MAP,
        "core_columns": None,
        "pacs": True,
    },
}

SOCRATA_BATCH_SIZE = 50_000


# ---------------------------------------------------------------------------
# Shared internal helpers
# ---------------------------------------------------------------------------

def _socrata_paginate(
    url: str,
    where_clause: str,
    sort_key: str = ":id",
    select_cols: list[str] | None = None,
    batch_size: int = SOCRATA_BATCH_SIZE,
    limit: int | None = None,
    app_token: str | None = None,
) -> pd.DataFrame:
    """Paginate a Socrata JSON endpoint and return a single concatenated DataFrame.

    Args:
        url: Full Socrata resource URL ending in .json.
        where_clause: SoQL $where filter (e.g., "tax_year='2024'").
        sort_key: SoQL $order column for stable pagination (default ``:id``).
        select_cols: Optional list of column names for $select projection.
        batch_size: Rows per request (Socrata max is 50,000).
        limit: Optional ceiling on total rows fetched (for testing).
        app_token: Optional Socrata app token for higher rate limits.

    Returns:
        DataFrame with all matching rows.
    """
    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token

    all_records: list[dict] = []
    offset = 0

    while True:
        batch_limit = batch_size
        if limit is not None:
            remaining = limit - len(all_records)
            if remaining <= 0:
                break
            batch_limit = min(batch_size, remaining)

        params: dict = {
            "$where": where_clause,
            "$limit": batch_limit,
            "$offset": offset,
            "$order": sort_key,
        }
        if select_cols:
            params["$select"] = ",".join(select_cols)

        resp = requests.get(url, params=params, headers=headers)
        resp.raise_for_status()
        records = resp.json()

        if not records:
            break

        all_records.extend(records)
        offset += len(records)

        if len(records) < batch_limit:
            break  # partial page = last page

    console.print(f"[dim]Fetched {len(all_records):,} records from {url}[/dim]")
    return pd.DataFrame(all_records)


def _arcgis_geojson_fetch(url: str, where_clause: str, max_records: int, label: str) -> gpd.GeoDataFrame:
    """Shared ArcGIS REST GeoJSON fetch used by all county geometry helpers."""
    params = {
        "where": where_clause,
        "outFields": "*",
        "f": "geojson",
        "resultRecordCount": max_records,
    }
    console.print(f"[dim]Fetching geometry from {label} GIS...[/dim]")
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    gdf = gpd.GeoDataFrame.from_features(resp.json()["features"], crs="EPSG:4326")
    console.print(f"[green]Retrieved {len(gdf)} parcel geometries from {label}[/green]")
    return gdf


def _keyset_feature_pages(
    url: str,
    *,
    bbox: str,
    oid_field: str = "OBJECTID",
    page_guard: int = 2000,
    timeout: int = 120,
    sleep: float = 0.5,
    max_records: int | None = None,
) -> list:
    """Return raw GeoJSON features for one bbox query paged by OBJECTID keyset."""
    features_out: list = []
    last_oid = 0
    while True:
        params = {
            "geometry": bbox,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "where": f"{oid_field} > {last_oid}",
            "orderByFields": f"{oid_field} ASC",
            "outFields": "*",
            "outSR": "4326",
            "f": "geojson",
            "resultRecordCount": page_guard,
        }
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if not features:
            break

        if max_records is not None:
            remaining = max_records - len(features_out)
            if remaining <= 0:
                break
            features = features[:remaining]

        page_max = max(
            (feature.get("properties", {}).get(oid_field, 0) or 0)
            for feature in features
        )
        if page_max <= last_oid:
            break

        features_out.extend(features)
        last_oid = page_max
        if max_records is not None and len(features_out) >= max_records:
            break
        if len(features) < page_guard:
            break
        time.sleep(sleep)
    return features_out


def _arcgis_bbox_keyset_fetch(
    url: str,
    *,
    bbox: str,
    oid_field: str = "OBJECTID",
    page_guard: int = 2000,
    timeout: int = 120,
    sleep: float = 0.5,
    max_records: int | None = None,
) -> gpd.GeoDataFrame:
    """Fetch an ArcGIS envelope query without resultOffset pagination."""
    features = _keyset_feature_pages(
        url,
        bbox=bbox,
        oid_field=oid_field,
        page_guard=page_guard,
        timeout=timeout,
        sleep=sleep,
        max_records=max_records,
    )
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


def _arcgis_bbox_tiled_keyset_fetch(
    url: str,
    *,
    bbox: str,
    tile_deg: float = 0.1,
    oid_field: str = "OBJECTID",
    page_guard: int = 2000,
    timeout: int = 120,
    sleep: float = 0.0,
    max_records: int | None = None,
) -> gpd.GeoDataFrame:
    """Fetch a large ArcGIS bbox as smaller keyset-paged tiles."""
    if tile_deg <= 0:
        raise ValueError("tile_deg must be positive")

    minx, miny, maxx, maxy = (float(value) for value in bbox.split(","))
    seen_oids: set = set()
    merged: list = []
    x = minx
    while x < maxx:
        x2 = min(x + tile_deg, maxx)
        y = miny
        while y < maxy:
            y2 = min(y + tile_deg, maxy)
            tile = f"{x},{y},{x2},{y2}"
            for feature in _keyset_feature_pages(
                url,
                bbox=tile,
                oid_field=oid_field,
                page_guard=page_guard,
                timeout=timeout,
                sleep=sleep,
                max_records=None
                if max_records is None
                else max_records - len(merged),
            ):
                oid = feature.get("properties", {}).get(oid_field)
                if oid is not None:
                    if oid in seen_oids:
                        continue
                    seen_oids.add(oid)
                merged.append(feature)
                if max_records is not None and len(merged) >= max_records:
                    return gpd.GeoDataFrame.from_features(merged, crs="EPSG:4326")
            if y2 <= y:
                break
            y = y2
        if x2 <= x:
            break
        x = x2

    if not merged:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(merged, crs="EPSG:4326")


def _tiger_cache_dir() -> Path:
    """Cache TIGER county boundaries under the geostack repo data directory."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "tiger"


def _txgio_county_boundary(
    county_fips: str,
    *,
    state_fips: str = "48",
    year: int = 2024,
):
    """Return a TIGER county boundary geometry in EPSG:4326."""
    cache_dir = _tiger_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"tl_{year}_us_county.zip"
    if not cache_path.exists():
        url = TIGER_COUNTY_URL.format(year=year)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)

    counties = gpd.read_file(f"zip://{cache_path}")
    county = counties[
        (counties["STATEFP"].astype(str) == state_fips)
        & (counties["COUNTYFP"].astype(str) == county_fips)
    ]
    if county.empty:
        raise ValueError(f"County boundary not found for {state_fips}{county_fips}")
    return county.to_crs("EPSG:4326").geometry.iloc[0]


def _txgio_county_bbox(county_fips: str, *, state_fips: str = "48") -> tuple[str, object]:
    """Return ``minx,miny,maxx,maxy`` plus boundary geometry for a county."""
    boundary = _txgio_county_boundary(county_fips, state_fips=state_fips)
    minx, miny, maxx, maxy = boundary.bounds
    return f"{minx},{miny},{maxx},{maxy}", boundary


def _filter_txgio_to_county(gdf: gpd.GeoDataFrame, boundary) -> gpd.GeoDataFrame:
    """Keep TxGIO parcels whose centroids fall inside the county boundary."""
    if gdf.empty:
        return gdf
    source = gdf if gdf.crs is not None else gdf.set_crs("EPSG:4326")
    source = source.to_crs("EPSG:4326")
    projected = source.to_crs(source.estimate_utm_crs())
    centroids = projected.geometry.centroid
    centroids = gpd.GeoSeries(centroids, crs=projected.crs).to_crs("EPSG:4326")
    mask = centroids.within(boundary)
    return source[mask].copy()


def _fetch_txgio_county_parcel_geometry(
    county: str,
    *,
    max_records: int | None = None,
) -> gpd.GeoDataFrame:
    """Fetch TxGIO statewide parcels for a county using bbox/keyset + centroid filtering."""
    meta = COUNTY_REGISTRY[county]
    bbox, boundary = _txgio_county_bbox(
        meta["county_fips"],
        state_fips=meta["state_fips"],
    )
    console.print(f"[dim]Fetching {county.title()} geometry from TxGIO bbox...[/dim]")
    frames: list[gpd.GeoDataFrame] = []
    kept = 0
    last_oid = 0
    page_guard = 2000
    while True:
        params = {
            "geometry": bbox,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "where": f"OBJECTID > {last_oid}",
            "orderByFields": "OBJECTID ASC",
            "outFields": "*",
            "outSR": "4326",
            "f": "geojson",
            "resultRecordCount": page_guard,
        }
        resp = requests.get(meta["gis_url"], params=params, timeout=120)
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if not features:
            break

        page_max = max(
            (feature.get("properties", {}).get("OBJECTID", 0) or 0)
            for feature in features
        )
        if page_max <= last_oid:
            break

        page_gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        filtered_page = _filter_txgio_to_county(page_gdf, boundary)
        if not filtered_page.empty:
            frames.append(filtered_page)
            kept += len(filtered_page)
            if max_records is not None and kept >= max_records:
                break

        last_oid = page_max
        if len(features) < page_guard:
            break

    if frames:
        filtered = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True),
            crs="EPSG:4326",
        )
        if max_records is not None:
            filtered = filtered.head(max_records).copy()
    else:
        filtered = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    console.print(
        f"[green]Retrieved {len(filtered)} {county.title()} parcel geometries from TxGIO[/green]"
    )
    return filtered


# ---------------------------------------------------------------------------
# Generic fetch / normalize / load API
# ---------------------------------------------------------------------------

def fetch_parcels(
    county: str,
    year: int | None = None,
    core_only: bool = True,
    limit: int | None = None,
    app_token: str | None = None,
) -> pd.DataFrame:
    """Fetch parcel appraisal data for any supported county.

    Routes to the county-specific fetch path and returns a raw
    (un-normalized) DataFrame. Use :func:`normalize_parcels` to map columns
    to the canonical ParcelYear schema.

    Args:
        county: County name — one of ``collin``, ``dallas``, ``cooke``,
            ``travis``, ``grayson``, ``tarrant``, ``wise``, ``denton``.
        year: Appraisal tax year. ``None`` (default) means "current / best
            available" — each county resolves this to its newest dataset
            (Cooke: portal's current year; Grayson: the year in the live
            certified export; bulk counties: newest registered URL).
            Explicit years are passed through unchanged.
        core_only: For Collin, fetch only CORE_COLUMNS (~75% smaller). Ignored
            for other counties (no column projection defined yet).
        limit: Cap on total rows fetched (useful for testing).
        app_token: Optional Socrata app token.

    Returns:
        Raw DataFrame with all columns as returned by the county source.

    Raises:
        ValueError: If county is not in COUNTY_REGISTRY.
    """
    county = county.lower()
    if county not in COUNTY_REGISTRY:
        raise ValueError(
            f"Unknown county {county!r}. Supported: {list(COUNTY_REGISTRY)}"
        )

    meta = COUNTY_REGISTRY[county]

    if county == "collin":
        return fetch_collin_parcels(
            year=year if year is not None else max(COLLIN_DATASET_IDS),
            core_only=core_only, limit=limit, app_token=app_token,
        )

    if county == "wise":
        return fetch_wise_parcels(
            year=year if year is not None else max(WISE_DATASET_URLS), limit=limit
        )

    if county == "tarrant":
        return fetch_tarrant_parcels(
            year=year if year is not None else max(TAD_FULL_URLS), limit=limit
        )

    if county == "dallas":
        return fetch_dcad_parcels(
            year=year if year is not None else max(DCAD_DATASET_URLS), limit=limit
        )

    if county == "denton":
        return fetch_denton_parcels(
            year=year if year is not None else max(DENTON_DATASET_URLS), limit=limit
        )

    if county == "cooke":
        # True Prodigy treats year=None as "portal's current year"
        return fetch_cooke_parcels_trueprodigy(year=year, normalize=False)

    if county == "grayson":
        # Grayson's export carries only the current certified year.
        # year=None auto-detects the roll year from the export itself.
        return fetch_grayson_parcels(year=year, limit=limit)

    # Generic Socrata path for other counties
    url = meta["socrata_url"]
    if url is None:
        raise ValueError(
            f"County {county!r} has no Socrata URL and no dedicated fetch function. "
            f"Check COUNTY_REGISTRY for the correct fetch path."
        )
    if year is None:
        year = 2024  # generic Socrata fetchers' historical default
    sort_key = meta["sort_key"]
    console.print(f"[dim]Fetching {county.title()} County parcels (year={year})...[/dim]")
    return _socrata_paginate(
        url=url,
        where_clause=f"tax_year='{year}'",
        sort_key=sort_key,
        limit=limit,
        app_token=app_token,
    )


def normalize_parcels(df: pd.DataFrame, county: str) -> pd.DataFrame:
    """Normalize raw CAD Socrata data to the canonical ParcelYear schema.

    Renames columns, cleans numeric/year/address fields, adds county and
    state columns, and computes ``ilr`` and ``age`` derived fields.

    Args:
        df: Raw DataFrame from :func:`fetch_parcels`.
        county: County name (used to select the right column mapping).

    Returns:
        Normalized DataFrame matching the ParcelYear schema.

    Raises:
        ValueError: If county is not supported or no columns could be mapped.
    """
    from geostack.analysis import (
        clean_numeric,
        clean_year,
        compute_age,
        compute_ilr,
        normalize_address,
    )

    county = county.lower()
    if county not in COUNTY_REGISTRY:
        raise ValueError(f"Unknown county {county!r}")

    meta = COUNTY_REGISTRY[county]
    column_map = meta["column_map"]

    # Apply only the columns that are actually present in the DataFrame
    df_cols_lower = {c.lower(): c for c in df.columns}
    rename = {}
    for src, dst in column_map.items():
        actual = df_cols_lower.get(src.lower())
        if actual:
            rename[actual] = dst

    if not rename:
        raise ValueError(
            f"Could not map any columns for {county}. "
            f"Available: {list(df.columns)[:10]}"
        )

    df = df.rename(columns=rename)

    # For PACS counties, map property type codes to canonical labels
    if meta.get("pacs") and "property_type" in df.columns:
        _pacs_type = {"R": "Real", "MH": "Real", "P": "Personal", "MN": "Mineral", "PP": "Personal"}
        df["property_type"] = df["property_type"].map(_pacs_type).fillna(df["property_type"])
    if county == "grayson" and "state_use_code" in df.columns and "land_state_cd" in df.columns:
        state_use = df["state_use_code"].fillna("").astype(str).str.strip()
        land_state = df["land_state_cd"].fillna("").astype(str).str.strip()
        df["state_use_code"] = state_use.where(state_use != "", land_state)

    # Add county / state
    df["county"] = county
    df["state"] = meta["state"]

    # Clean numeric value columns
    value_cols = [
        "land_value", "improvement_value", "total_value", "market_value",
        "prev_land_value", "prev_improvement_value", "prev_total_value",
        "assessed_value", "appraised_value", "land_acres", "land_sqft",
        "main_area_sqft",
    ]
    for col in value_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    # Coordinate columns (True Prodigy counties carry 'N/A' strings for missing)
    for col in ("latitude", "longitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean year columns
    if "year_built" in df.columns:
        df["year_built"] = clean_year(df["year_built"])

    if "tax_year" in df.columns:
        df["tax_year"] = pd.to_numeric(df["tax_year"], errors="coerce").astype("Int64")

    # Clean address
    if "situs_address" in df.columns:
        df["situs_address"] = normalize_address(df["situs_address"])

    # Compute derived fields
    if "land_value" in df.columns and "improvement_value" in df.columns:
        df = compute_ilr(df)

    if "year_built" in df.columns and "tax_year" in df.columns:
        # Collin doesn't have a separate effective_year — use year_built
        if "effective_year" not in df.columns:
            df["effective_year"] = df["year_built"]
        df = compute_age(df, built_col="effective_year")

    # Warn if coverage is low (< 5 canonical columns mapped)
    canonical = {
        "parcel_id", "land_value", "improvement_value",
        "total_value", "year_built", "property_type",
        "situs_address", "city", "zip_code",
    }
    mapped_count = len(canonical & set(df.columns))
    if mapped_count < 5:
        console.print(
            f"[yellow]Warning: only {mapped_count} of {len(canonical)} expected "
            f"columns mapped for {county}. Column names may have changed.[/yellow]"
        )

    return df


def load_parcels_to_postgis(
    df: pd.DataFrame,
    county: str,
    year: int,
    engine,
    table: str = "parcels",
) -> int:
    """Load a normalized parcel DataFrame into the PostGIS parcels table.

    Uses DELETE + re-insert (same pattern as ``load_roads_to_postgis``) so it
    is safe to re-run for the same county + year.

    Args:
        df: Normalized DataFrame from :func:`normalize_parcels`. May optionally
            have a ``geometry`` column (GeoDataFrame).
        county: County name (stored in the ``county`` column).
        year: Tax year (used for the DELETE scope).
        engine: SQLAlchemy engine pointing at the geostack PostGIS database.
        table: Target table name (default ``parcels``).

    Returns:
        Number of rows inserted.
    """
    from sqlalchemy import text

    with engine.begin() as conn:
        deleted = conn.execute(
            text(f"DELETE FROM {table} WHERE county = :c AND tax_year = :y"),
            {"c": county, "y": int(year)},
        ).rowcount
    if deleted:
        console.print(f"[dim]Deleted {deleted:,} existing rows for {county}/{year}[/dim]")

    is_spatial = hasattr(df, "geometry") and "geometry" in df.columns

    if is_spatial:
        import geopandas as gpd

        # Coerce any plain Polygon geometries to MultiPolygon to match the
        # GEOMETRY(MULTIPOLYGON, 4326) column type in the schema
        from shapely.geometry import MultiPolygon

        def _to_multi(geom):
            if geom is None or pd.isna(geom):
                return None
            if geom.geom_type == "Polygon":
                return MultiPolygon([geom])
            return geom

        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        gdf["geometry"] = gdf["geometry"].map(_to_multi)
        gdf.to_postgis(table, engine, if_exists="append", index=False)
    else:
        df.to_sql(table, engine, if_exists="append", index=False)

    # Recreate indexes that to_postgis/to_sql doesn't manage
    with engine.begin() as conn:
        conn.execute(text(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_county_year "
            f"ON {table}(county, tax_year)"
        ))
        conn.execute(text(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_parcel_id "
            f"ON {table}(parcel_id)"
        ))
        if is_spatial:
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_geom "
                f"ON {table} USING GIST(geometry) WHERE geometry IS NOT NULL"
            ))

    console.print(f"[green]Loaded {len(df):,} parcels → {table} ({county}/{year})[/green]")
    return len(df)


# ---------------------------------------------------------------------------
# Collin County — full-pagination fetch (primary path)
# ---------------------------------------------------------------------------

def fetch_collin_parcels(
    year: int = 2024,
    core_only: bool = True,
    limit: int | None = None,
    app_token: str | None = None,
) -> pd.DataFrame:
    """Fetch the full Collin CAD appraisal roll for a given year.

    Uses year-specific Socrata dataset IDs and paginates in 50K-row batches
    to retrieve the complete dataset.

    Args:
        year: Tax year (must be in :data:`COLLIN_DATASET_IDS`).
        core_only: Fetch only :data:`COLLIN_CORE_COLUMNS` (default True).
            Set False for the full ~75-column schema.
        limit: Optional row cap for testing (default None = all rows).
        app_token: Optional Socrata app token for higher rate limits.

    Returns:
        Raw DataFrame with Collin CAD column names.

    Raises:
        ValueError: If year is not in :data:`COLLIN_DATASET_IDS`.
    """
    dataset_id = COLLIN_DATASET_IDS.get(year)
    if not dataset_id:
        available = sorted(COLLIN_DATASET_IDS.keys())
        raise ValueError(
            f"No Collin CAD dataset for year {year}. Available: {available}"
        )

    url = f"https://data.texas.gov/resource/{dataset_id}.json"
    select_cols = COLLIN_CORE_COLUMNS if core_only else None

    console.print(
        f"[dim]Fetching Collin County parcels: year={year}, "
        f"dataset={dataset_id}, core_only={core_only}[/dim]"
    )
    return _socrata_paginate(
        url=url,
        where_clause=f"propyear='{year}'",
        sort_key="propid",
        select_cols=select_cols,
        limit=limit,
        app_token=app_token,
    )


def fetch_collin_parcels_socrata(year: int = 2024, limit: int = 50000, offset: int = 0) -> pd.DataFrame:
    """Fetch Collin County parcel data (single-page, legacy interface).

    .. deprecated::
        Use :func:`fetch_collin_parcels` for full pagination. This function
        fetches only a single page and uses the legacy static dataset URL.
    """
    import warnings
    warnings.warn(
        "fetch_collin_parcels_socrata is deprecated. "
        "Use fetch_collin_parcels(year, core_only=True) for full pagination.",
        DeprecationWarning,
        stacklevel=2,
    )
    dataset_id = COLLIN_DATASET_IDS.get(year, "6dqt-e958")
    url = f"https://data.texas.gov/resource/{dataset_id}.json"
    params = {
        "$where": f"propyear='{year}'",
        "$limit": limit,
        "$offset": offset,
        "$order": "propid",
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


# ---------------------------------------------------------------------------
# Collin County — geometry
# ---------------------------------------------------------------------------

COLLIN_GIS_URL = (
    "https://gis.collincountytx.gov/arcgis/rest/services/"
    "Public/Parcels/MapServer/0/query"
)


def fetch_collin_parcel_geometry(
    parcel_ids: list[str] | None = None,
    where_clause: str | None = None,
    max_records: int = 1000,
) -> gpd.GeoDataFrame:
    """Fetch parcel geometry from Collin CAD GIS REST service.

    Args:
        parcel_ids: List of parcel IDs to fetch geometry for.
        where_clause: ArcGIS WHERE clause (overrides parcel_ids if both given).
        max_records: Max records per request (ArcGIS limit is typically 1000).

    Returns:
        GeoDataFrame with parcel polygons in EPSG:4326.
    """
    if parcel_ids and not where_clause:
        ids_str = ",".join(f"'{pid}'" for pid in parcel_ids)
        where_clause = f"PROP_ID IN ({ids_str})"
    elif not where_clause:
        where_clause = "1=1"

    return _arcgis_geojson_fetch(COLLIN_GIS_URL, where_clause, max_records, "Collin CAD")


# ---------------------------------------------------------------------------
# Dallas County CAD helpers
# ---------------------------------------------------------------------------

def fetch_dallas_parcels(year: int = 2024, limit: int | None = None, offset: int = 0) -> pd.DataFrame:
    """Fetch Dallas County parcel/appraisal data from Socrata.

    Returns appraisal roll attributes (no geometry). Pair with
    :func:`fetch_dallas_parcel_geometry` to get spatial data.
    """
    console.print(f"[dim]Fetching Dallas parcels (year={year})...[/dim]")
    return _socrata_paginate(
        url=DALLAS_SOCRATA_DATASET,
        where_clause=f"tax_year='{year}'",
        sort_key="account_num",
        limit=limit,
    )


def fetch_dallas_parcel_geometry(
    parcel_ids: list[str] | None = None,
    where_clause: str | None = None,
    max_records: int = 1000,
) -> gpd.GeoDataFrame:
    """Fetch parcel geometry from Dallas CAD ArcGIS REST service (single page)."""
    if parcel_ids and not where_clause:
        ids_str = ",".join(f"'{pid}'" for pid in parcel_ids)
        where_clause = f"ACCOUNT_NUM IN ({ids_str})"
    elif not where_clause:
        where_clause = "1=1"
    return _arcgis_geojson_fetch(DALLAS_GIS_URL, where_clause, max_records, "Dallas CAD")


def _arcgis_paginate(
    url: str,
    out_fields: str = "*",
    where: str = "1=1",
    page_size: int = 2000,
    max_records: int | None = None,
    order_by: str = "OBJECTID",
    label: str = "ArcGIS",
) -> gpd.GeoDataFrame:
    """Page through an ArcGIS REST FeatureServer via resultOffset.

    The single-page :func:`_arcgis_geojson_fetch` caps at the server's
    maxRecordCount (typically 2000). This loops resultOffset until exhausted,
    so it can pull entire layers (e.g. all ~695k DCAD parcels).
    """
    frames: list[gpd.GeoDataFrame] = []
    offset, total = 0, 0
    while True:
        n = page_size
        if max_records is not None:
            n = min(page_size, max_records - total)
            if n <= 0:
                break
        params = {
            "where": where, "outFields": out_fields, "f": "geojson",
            "resultRecordCount": n, "resultOffset": offset, "orderByFields": order_by,
        }
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        feats = resp.json().get("features", [])
        if not feats:
            break
        frames.append(gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326"))
        got = len(feats)
        offset += got
        total += got
        if total % 20000 < page_size:
            console.print(f"[dim]  {label}: {total:,} features...[/dim]")
        if got < n:
            break
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    out = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")


def fetch_dallas_parcel_geometry_all(
    out_fields: str = "GIS_Acct",
    max_records: int | None = None,
    cache: bool = True,
    cache_path: str | None = None,
) -> gpd.GeoDataFrame:
    """Fetch ALL Dallas CAD parcel polygons (paginated, ~695k), with caching.

    Args:
        out_fields: Attribute fields to keep (default just the GIS_Acct join key).
        max_records: Cap for testing; None pulls the whole layer.
        cache: Cache the result to a local parquet for fast re-use.
        cache_path: Override cache location.

    Returns:
        GeoDataFrame of parcel polygons (EPSG:4326) with the GIS_Acct join key.
    """
    import os

    cache_path = cache_path or os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "dcad", "dcad_parcel_geometry.parquet"
    )
    if cache and max_records is None and os.path.exists(cache_path):
        console.print(f"[dim]Using cached DCAD geometry: {cache_path}[/dim]")
        return gpd.read_parquet(cache_path)

    console.print("[bold]Pulling DCAD parcel geometry (paginated)...[/bold]")
    gdf = _arcgis_paginate(DALLAS_GIS_URL, out_fields=out_fields,
                           max_records=max_records, label="Dallas CAD")
    console.print(f"[green]Retrieved {len(gdf):,} DCAD parcel polygons[/green]")
    if cache and max_records is None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        gdf.to_parquet(cache_path)
        console.print(f"[dim]Cached to {cache_path}[/dim]")
    return gdf


# ---------------------------------------------------------------------------
# True Prodigy CAD source (shared) — used by CADs on the True Prodigy platform
#
# True Prodigy is a CAD-vendor React SPA backed by a REST API. The search
# results carry full records INCLUDING latitude/longitude, appraised/land/
# improvement value, and acreage — so no geocoding is needed. Auth is a Bearer
# token minted in-browser; we capture it once via Playwright, then page the
# searchfulltext API directly. Counties on this platform are registered in
# TRUEPRODIGY_SOURCES; add a portal + ZIP list to support a new one.
# ---------------------------------------------------------------------------

TRUEPRODIGY_SOURCES: dict[str, dict] = {
    "cooke": {
        "portal": "https://www.cookecad.org/property-search",
        # ZIPs covering the county; searchfulltext matches the zip, dedup after.
        "zips": ["76240", "76250", "76252", "76253", "76265", "76272", "76273", "76238"],
    },
}

# Raw True Prodigy field -> canonical parcel column (note: lat/lon included)
_TRUEPRODIGY_COLUMN_MAP: dict[str, str] = {
    "pAccountID": "parcel_id",
    "pYear": "tax_year",
    "landValue": "land_value",
    "improvementValue": "improvement_value",
    "appraisedValue": "total_value",
    "marketValue": "market_value",
    "name": "owner_name",
    # Owner mailing address — True Prodigy JSON field names (distinct from situs)
    # addrDeliveryLine / addrCity / addrState / addrZip are the owner's mailing contact.
    # Verified against Cooke CCAD roll: addrDeliveryLine often differs from fullSitus.
    "addrDeliveryLine": "mail_addr1",
    "addrCity":         "mail_city",
    "addrState":        "mail_state",
    "addrZip":          "mail_zip",
    "fullSitus": "situs_address",
    "city": "city",
    "zip": "zip_code",
    "propType": "property_type",
    "useCd": "land_use_code",
    "legalAcreage": "land_acres",
    "latitude": "latitude",
    "longitude": "longitude",
    "geoID": "geo_id",
}


def _trueprodigy_capture_auth(portal_url: str, seed_term: str = "76240", headless: bool = True) -> dict:
    """Load a True Prodigy portal and capture the searchfulltext endpoint + auth.

    Returns {"url", "headers", "body"} captured from a real search request,
    suitable for replaying directly with ``requests``.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "playwright is required for the True Prodigy source. Install with "
            "`pip install playwright && playwright install chromium`."
        )
    cap: dict = {}

    def on_request(req):
        if "searchfulltext" in req.url and "url" not in cap:
            cap["url"] = req.url.split("?")[0]
            cap["headers"] = dict(req.headers)
            try:
                cap["body"] = json.loads(req.post_data) if req.post_data else {}
            except Exception:
                cap["body"] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        page.on("request", on_request)
        console.print(f"[dim]True Prodigy: loading {portal_url}...[/dim]")
        page.goto(portal_url, timeout=45000)
        page.wait_for_timeout(3000)
        box = page.query_selector("#searchInput")
        if not box:
            browser.close()
            raise RuntimeError("True Prodigy search input (#searchInput) not found")
        box.fill(seed_term)
        page.keyboard.press("Enter")
        page.wait_for_timeout(5000)
        browser.close()
    if "url" not in cap:
        raise RuntimeError("Did not capture a searchfulltext request")
    return cap


def fetch_trueprodigy_roll(
    portal_url: str,
    search_terms: list[str],
    year: int | str | None = None,
    page_size: int = 500,
    dedup_key: str = "pAccountID",
    headless: bool = True,
) -> pd.DataFrame:
    """Pull a full parcel roll from a True Prodigy CAD by paging searchfulltext.

    Args:
        portal_url: The CAD's property-search portal URL.
        search_terms: Terms to search (typically the county's ZIP codes — each
            paged fully, then deduped — to cover the whole roll).
        year: Appraisal year (pYear). Defaults to the portal's current year.
        page_size: API page size (server cap is typically 500).
        dedup_key: Record key to dedup on across terms.
        headless: Run the browser headless.

    Returns:
        DataFrame of raw True Prodigy records (one row per unique parcel).
        Pass to :func:`normalize_trueprodigy` for the canonical schema.
    """
    cap = _trueprodigy_capture_auth(portal_url, search_terms[0], headless=headless)
    if year is None:
        year = (cap["body"].get("pYear") or {}).get("value")
    headers = {k: v for k, v in cap["headers"].items()
               if k.lower() in ("authorization", "content-type", "accept", "origin", "referer")}
    headers.setdefault("content-type", "application/json")

    rows, seen = [], set()
    for term in search_terms:
        page, got_term = 1, 0
        while True:
            body = dict(cap["body"])
            if year is not None:
                body["pYear"] = {"operator": "=", "value": str(year)}
            body["fullTextSearch"] = {"operator": "match", "value": term}
            url = f"{cap['url']}?page={page}&pageSize={page_size}"
            resp = requests.post(url, headers=headers, json=body, timeout=60)
            if resp.status_code != 200:
                break
            results = resp.json().get("results") or []
            for r in results:
                k = r.get(dedup_key) or r.get("pid")
                if k and k not in seen:
                    seen.add(k)
                    rows.append(r)
                    got_term += 1
            if len(results) < page_size:
                break
            page += 1
            time.sleep(0.3)
        console.print(f"[dim]  {term}: +{got_term} new (total {len(rows):,})[/dim]")
    console.print(f"[green]True Prodigy: {len(rows):,} unique parcels[/green]")
    return pd.DataFrame(rows)


def normalize_trueprodigy(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw True Prodigy records to the canonical parcel schema (with lat/lon)."""
    cols = {raw: canon for raw, canon in _TRUEPRODIGY_COLUMN_MAP.items() if raw in df.columns}
    out = df[list(cols)].rename(columns=cols).copy()
    for c in ("land_value", "improvement_value", "total_value", "market_value",
              "land_acres", "latitude", "longitude", "tax_year"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fetch_cooke_parcels_trueprodigy(year: int | str | None = None,
                                    normalize: bool = True,
                                    headless: bool = True) -> pd.DataFrame:
    """Full Cooke County roll via True Prodigy (CCAD) — values + source lat/lon.

    The working replacement for the dead ArcGIS endpoint / partial Socrata feed.
    ~30k parcels, ~90% with coordinates, no geocoding required.
    """
    src = TRUEPRODIGY_SOURCES["cooke"]
    raw = fetch_trueprodigy_roll(src["portal"], src["zips"], year=year, headless=headless)
    return normalize_trueprodigy(raw) if normalize else raw


# ---------------------------------------------------------------------------
# Cooke County CAD helpers
# ---------------------------------------------------------------------------

def fetch_cooke_parcels(year: int | str | None = None, limit: int | None = None, offset: int = 0) -> pd.DataFrame:
    """Fetch Cooke County parcel/appraisal data via True Prodigy (CCAD portal).

    The original Socrata dataset (mgnk-vbdk) returned 404 as of mid-2025.
    This function now delegates to :func:`fetch_cooke_parcels_trueprodigy`,
    which scrapes the live CCAD True Prodigy portal (~30k parcels, ~90% with
    coordinates) using Playwright (headless by default).

    Args:
        year: Appraisal year. None = portal's current year (recommended).
        limit: Ignored (True Prodigy returns the full roll; no row cap supported).
        offset: Ignored (kept for API compatibility only).

    Returns:
        Raw DataFrame with True Prodigy field names. Pass to
        :func:`normalize_parcels` with ``county='cooke'``.
    """
    if limit is not None:
        console.print("[yellow]Warning: limit= is ignored for Cooke County (True Prodigy returns full roll)[/yellow]")
    return fetch_cooke_parcels_trueprodigy(year=year, normalize=False)


def fetch_cooke_parcel_geometry(
    parcel_ids: list[str] | None = None,
    where_clause: str | None = None,
    max_records: int = 1000,
) -> gpd.GeoDataFrame:
    """Fetch parcel geometry from Cooke County CAD ArcGIS REST service."""
    if parcel_ids and not where_clause:
        ids_str = ",".join(f"'{pid}'" for pid in parcel_ids)
        where_clause = f"PROP_ID IN ({ids_str})"
    elif not where_clause:
        where_clause = "1=1"
    return _arcgis_geojson_fetch(COOKE_GIS_URL, where_clause, max_records, "Cooke County CAD")


# ---------------------------------------------------------------------------
# Travis County CAD (TCAD — Austin metro) helpers
# ---------------------------------------------------------------------------

def fetch_tcad_parcels(year: int = 2024, limit: int | None = None, offset: int = 0) -> pd.DataFrame:
    """Fetch Travis County (Austin) parcel/appraisal data from Socrata."""
    console.print(f"[dim]Fetching Travis County parcels (year={year})...[/dim]")
    return _socrata_paginate(
        url=TCAD_SOCRATA_DATASET,
        where_clause=f"tax_year='{year}'",
        sort_key="prop_id",
        limit=limit,
    )


def fetch_tcad_parcel_geometry(
    parcel_ids: list[str] | None = None,
    where_clause: str | None = None,
    max_records: int = 1000,
) -> gpd.GeoDataFrame:
    """Fetch parcel geometry from Travis County CAD ArcGIS REST service."""
    if parcel_ids and not where_clause:
        ids_str = ",".join(f"'{pid}'" for pid in parcel_ids)
        where_clause = f"PROP_ID IN ({ids_str})"
    elif not where_clause:
        where_clause = "1=1"
    return _arcgis_geojson_fetch(TCAD_GIS_URL, where_clause, max_records, "Travis CAD")


# ---------------------------------------------------------------------------
# Grayson County CAD helpers
# ---------------------------------------------------------------------------

def _grayson_cache_dir() -> Path:
    """Grayson cache dir anchored to the geostack repo root (not cwd).

    cad.py lives at <repo>/src/geostack/cad.py, so repo root is three parents up.
    Anchoring here keeps cache files inside geostack's tree even when called
    from other projects (e.g. npl) that install geostack in editable mode.
    """
    return Path(__file__).resolve().parent.parent.parent / "data" / "grayson"


def fetch_grayson_parcels(year: int | None = None, limit: int | None = None, offset: int = 0) -> pd.DataFrame:
    """Fetch Grayson County parcel/appraisal data from the PACS bulk export.

    The original Socrata dataset (e4rz-6rb7) returned 404 as of mid-2025.
    Grayson CAD publishes a PACS 8.0.31 fixed-width export at
    graysonappraisal.org/public-information/ — same record layout as Wise/Denton.

    Downloads :data:`GRAYSON_EXPORT_URL`, parses APPRAISAL_INFO.TXT (9,262
    chars/record), filters to real property (prop_type_cd in 'R', 'MH') and
    certified records (sup_num == 0), then joins APPRAISAL_IMPROVEMENT_DETAIL.TXT
    for yr_built and living area sqft.

    The export is updated with each certification supplement and carries only
    the **current certified year** (2025 as of Feb 2026, ~49,500 real property
    parcels). The default ``year=None`` auto-detects the roll year from the
    export itself — this is the recommended call pattern.

    Args:
        year: Tax year. ``None`` (default) = auto-detect the current certified
            year from the export. An explicit year is enforced strictly: if the
            export does not contain that year, ValueError is raised.
        limit: Optional row cap for testing (default None = all rows).
        offset: Ignored (kept for API compatibility only).

    Returns:
        Raw DataFrame with PACS-derived column names, ready for
        :func:`normalize_parcels`.

    Raises:
        ValueError: If an explicit *year* was requested and the export
            yields zero certified records for it.
    """
    import zipfile

    cache_dir = _grayson_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Parquet cache. For year=None, use the newest cached roll (cache freshness
    # is bounded by the locally cached ZIP — delete data/grayson/ to force re-download).
    if year is not None:
        cache_path = cache_dir / f"grayson_{year}.parquet"
    else:
        cached = sorted(cache_dir.glob("grayson_*.parquet"))
        cache_path = cached[-1] if cached else None
    if cache_path is not None and cache_path.exists():
        console.print(f"[dim]Loading Grayson from parquet cache: {cache_path}[/dim]")
        df = pd.read_parquet(cache_path)
        missing_columns = _GRAYSON_CACHE_REQUIRED_COLUMNS.difference(df.columns)
        if not missing_columns:
            if limit:
                df = df.head(limit)
            console.print(f"[green]Grayson County: {len(df):,} parcels (cached)[/green]")
            return df
        console.print(
            "[dim]Ignoring stale Grayson parquet cache missing columns "
            f"{sorted(missing_columns)}; rebuilding from ZIP: {cache_path}[/dim]"
        )

    # Check for local ZIP before downloading
    zip_path = cache_dir / "GCAD_Export.zip"
    if not zip_path.exists():
        console.print(f"[dim]Downloading Grayson County export from {GRAYSON_EXPORT_URL} ...[/dim]")
        resp = requests.get(GRAYSON_EXPORT_URL, timeout=600)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        console.print(f"[dim]Downloaded {len(resp.content) / 1e6:.1f} MB → {zip_path}[/dim]")
    else:
        console.print(f"[dim]Using local Grayson ZIP: {zip_path}[/dim]")

    # Grayson uses standard Deflate (compress=8), Python zipfile handles it natively
    try:
        with zipfile.ZipFile(zip_path) as zf:
            _test = zf.namelist()
        use_python_zip = True
    except (zipfile.BadZipFile, NotImplementedError):
        use_python_zip = False

    console.print("[dim]Parsing APPRAISAL_INFO.TXT ...[/dim]")
    records = _parse_grayson_appraisal_info(str(zip_path), year, limit, use_python_zip)

    if not records:
        if year is not None:
            raise ValueError(
                f"Grayson CAD export contains no certified records for year {year}. "
                f"The export carries only the current certified roll — call with "
                f"year=None to auto-detect it."
            )
        raise ValueError(
            "Grayson CAD export yielded no certified real-property records — "
            "the export format may have changed."
        )

    df = pd.DataFrame(records)
    console.print(f"[dim]{len(df):,} real property records parsed[/dim]")

    # Detect the roll year from the parsed records (export carries one year)
    detected_year = int(df["prop_val_yr"].mode().iloc[0])
    effective_year = year if year is not None else detected_year

    console.print("[dim]Parsing APPRAISAL_IMPROVEMENT_DETAIL.TXT ...[/dim]")
    prop_ids = set(df["prop_id"])
    imprv_df = _parse_grayson_improvement_detail(str(zip_path), effective_year, prop_ids, use_python_zip)

    if not imprv_df.empty:
        df = df.merge(imprv_df, on="prop_id", how="left")
    else:
        df["yr_built"] = None
        df["main_area_sqft"] = None

    # Cache as parquet under the detected year (skip if limit was used — partial result)
    if limit is None:
        cache_path = cache_dir / f"grayson_{effective_year}.parquet"
        df.to_parquet(cache_path, index=False)
        console.print(f"[dim]Cached → {cache_path}[/dim]")

    console.print(f"[green]Grayson County {effective_year}: {len(df):,} parcels fetched[/green]")
    return df


def _grayson_find_member(zip_path: str, suffix: str) -> str:
    """Return the member name in *zip_path* whose name ends with *suffix* (case-insensitive)."""
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

    suffix_upper = suffix.upper()
    match = next((n for n in names if n.upper().endswith(suffix_upper)), None)
    if match is None:
        raise FileNotFoundError(
            f"No member ending with {suffix!r} found in {zip_path}. Contents: {names[:10]}"
        )
    return match


def _grayson_stream_member(zip_path: str, member: str, use_python_zip: bool = True):
    """Return a (reader, proc) tuple for streaming *member* from *zip_path*.

    If *use_python_zip* is True, uses Python's zipfile (standard Deflate).
    Otherwise falls back to system ``unzip -p`` (needed for Deflate64).
    """
    import io
    import subprocess
    import zipfile

    if use_python_zip:
        zf = zipfile.ZipFile(zip_path)
        raw = zf.open(member)
        reader = io.TextIOWrapper(raw, encoding="latin-1", errors="replace")
        reader._grayson_zf = zf
        return reader, None
    else:
        proc = subprocess.Popen(
            ["unzip", "-p", zip_path, member],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        reader = io.TextIOWrapper(proc.stdout, encoding="latin-1", errors="replace")
        return reader, proc


def _parse_grayson_appraisal_info(
    zip_path: str,
    year: int | None,
    limit: int | None,
    use_python_zip: bool = True,
) -> list[dict]:
    """Stream-parse APPRAISAL_INFO.TXT from Grayson CAD ZIP.

    Uses the same PACS field positions as Wise/Denton (PACS 8.0.30/8.0.31 are
    compatible for all fields used here). Filters to real property
    (prop_type_cd in 'R', 'MH'), certified records (sup_num == 0), and
    optionally the requested tax year.

    Note: Unlike Wise (where prop_val_yr is zero-padded to 5 chars), Grayson
    stores the full 4-digit year in prop_val_yr[17:22] zero-padded to 5 chars
    as '02025' — identical to Wise/Denton. The _safe_int comparison handles this.
    """
    member = _grayson_find_member(zip_path, "_APPRAISAL_INFO.TXT")
    records: list[dict] = []

    reader, proc = _grayson_stream_member(zip_path, member, use_python_zip)
    try:
        for line in reader:
            line = line.rstrip("\r\n")
            if len(line) < _AI_ASSESSED_VAL.stop:
                continue

            prop_type = line[_AI_PROP_TYPE].strip()
            if prop_type not in ("R", "MH"):
                continue
            if _safe_int(line[_AI_SUP_NUM]) != 0:
                continue
            if year is not None and _safe_int(line[_AI_PROP_VAL_YR]) != year:
                continue

            land_hstd  = _safe_int(line[_AI_LAND_HSTD]) or 0
            land_non   = _safe_int(line[_AI_LAND_NON_HSTD]) or 0
            imprv_hstd = _safe_int(line[_AI_IMPRV_HSTD]) or 0
            imprv_non  = _safe_int(line[_AI_IMPRV_NON_HSTD]) or 0

            situs_num = line[_AI_SITUS_NUM].strip() if len(line) > _AI_SITUS_NUM.stop else ""
            prefx  = line[_AI_SITUS_PREFX].strip()
            street = line[_AI_SITUS_STREET].strip()
            suffix = line[_AI_SITUS_SUFFIX].strip()
            situs  = " ".join(p for p in [situs_num, prefx, street, suffix] if p)

            imprv_state_cd = ""
            if len(line) > _AI_IMPRV_STATE_CD.stop:
                imprv_state_cd = line[_AI_IMPRV_STATE_CD].strip()
            land_state_cd = ""
            if len(line) > _AI_LAND_STATE_CD.stop:
                land_state_cd = line[_AI_LAND_STATE_CD].strip()

            # Owner mailing address (PACS py_addr_* block).
            # py_addr_line1 = attention/care-of (often blank); py_addr_line2 = street.
            # Prefer line2 (street); fall back to line1 (e.g. PO Box when line2 is empty).
            _m_line1 = line[_AI_MAIL_ADDR1].strip() if len(line) > _AI_MAIL_ADDR1.stop else ""
            _m_line2 = line[_AI_MAIL_ADDR2].strip() if len(line) > _AI_MAIL_ADDR2.stop else ""
            mail_street = _m_line2 or _m_line1

            records.append({
                "prop_id":           line[_AI_PROP_ID].strip(),
                "prop_val_yr":       str(_safe_int(line[_AI_PROP_VAL_YR])),
                "prop_type_cd":      prop_type,
                "imprv_state_cd":    imprv_state_cd,
                "land_state_cd":      land_state_cd,
                "geo_id":            line[_AI_GEO_ID].strip(),
                "py_owner_name":     line[_AI_OWNER_NAME].strip(),
                "mail_addr1":        mail_street,
                "mail_city":         line[_AI_MAIL_CITY].strip()  if len(line) > _AI_MAIL_CITY.stop  else "",
                "mail_state":        line[_AI_MAIL_STATE].strip() if len(line) > _AI_MAIL_STATE.stop else "",
                "mail_zip":          line[_AI_MAIL_ZIP].strip()   if len(line) > _AI_MAIL_ZIP.stop   else "",
                "situs_address":     situs,
                "situs_city":        line[_AI_SITUS_CITY].strip(),
                "situs_zip":         line[_AI_SITUS_ZIP].strip(),
                "hood_cd":           line[_AI_HOOD_CD].strip(),
                "land_value":        land_hstd + land_non,
                "improvement_value": imprv_hstd + imprv_non,
                "appraised_val":     _safe_int(line[_AI_APPRAISED_VAL]),
                "assessed_val":      _safe_int(line[_AI_ASSESSED_VAL]),
                "legal_acreage":     _pacs_acres(line[_AI_LEGAL_ACREAGE]),
            })

            if limit and len(records) >= limit:
                break
    finally:
        reader.close()
        if hasattr(reader, "_grayson_zf"):
            reader._grayson_zf.close()
        if proc is not None:
            proc.kill()
            proc.wait()

    return records


def _parse_grayson_improvement_detail(
    zip_path: str,
    year: int | None,
    prop_id_filter: set[str],
    use_python_zip: bool = True,
) -> pd.DataFrame:
    """Parse APPRAISAL_IMPROVEMENT_DETAIL.TXT from Grayson CAD ZIP.

    Uses the same PACS field positions as Wise/Denton. Returns yr_built
    (min across improvements) and main_area_sqft (sum of 'LA' type, or
    largest single improvement as fallback) per prop_id.

    Note: Grayson's improvement detail stores prop_val_yr as the full 4-digit
    year in positions [12:16] (not zero-padded to 5 like APPRAISAL_INFO.TXT).
    The _AID_PROP_VAL_YR slice (12:16) matches correctly for both.
    """
    try:
        member = _grayson_find_member(zip_path, "_APPRAISAL_IMPROVEMENT_DETAIL.TXT")
    except FileNotFoundError:
        console.print("[yellow]APPRAISAL_IMPROVEMENT_DETAIL.TXT not found — yr_built and main_area_sqft will be null[/yellow]")
        return pd.DataFrame(columns=["prop_id", "yr_built", "main_area_sqft"])

    agg: dict[str, dict] = {}

    reader, proc = _grayson_stream_member(zip_path, member, use_python_zip)
    try:
        for line in reader:
            line = line.rstrip("\r\n")
            if len(line) < _AID_AREA.stop:
                continue
            if year is not None and _safe_int(line[_AID_PROP_VAL_YR]) != year:
                continue

            prop_id = line[_AID_PROP_ID].strip()
            if prop_id not in prop_id_filter:
                continue

            yr = _safe_int(line[_AID_YR_BUILT])
            area = _safe_float(line[_AID_AREA])
            type_cd = line[_AID_TYPE_CD].strip()

            if prop_id not in agg:
                agg[prop_id] = {"yrs": [], "la_area": 0.0, "max_area": 0.0}
            if yr and yr > 1800:
                agg[prop_id]["yrs"].append(yr)
            if area:
                if type_cd == "LA":
                    agg[prop_id]["la_area"] += area
                agg[prop_id]["max_area"] = max(agg[prop_id]["max_area"], area)
    finally:
        reader.close()
        if hasattr(reader, "_grayson_zf"):
            reader._grayson_zf.close()
        if proc is not None:
            proc.kill()
            proc.wait()

    if not agg:
        return pd.DataFrame(columns=["prop_id", "yr_built", "main_area_sqft"])

    rows = []
    for pid, data in agg.items():
        yr_built = min(data["yrs"]) if data["yrs"] else None
        sqft = data["la_area"] if data["la_area"] > 0 else (data["max_area"] or None)
        rows.append({"prop_id": pid, "yr_built": yr_built, "main_area_sqft": sqft})

    return pd.DataFrame(rows)


def fetch_grayson_parcel_geometry(
    parcel_ids: list[str] | None = None,
    where_clause: str | None = None,
    max_records: int = 1000,
) -> gpd.GeoDataFrame:
    """Fetch parcel geometry from Grayson County CAD ArcGIS REST service."""
    if parcel_ids and not where_clause:
        ids_str = ",".join(f"'{pid}'" for pid in parcel_ids)
        where_clause = f"PROP_ID IN ({ids_str})"
    elif not where_clause:
        where_clause = "1=1"
    return _arcgis_geojson_fetch(GRAYSON_GIS_URL, where_clause, max_records, "Grayson County CAD")




def fetch_wise_parcel_geometry(max_records: int | None = None) -> gpd.GeoDataFrame:
    """Fetch Wise County parcel geometry from TxGIO using bbox/keyset paging."""
    return _fetch_txgio_county_parcel_geometry("wise", max_records=max_records)


def fetch_denton_parcel_geometry(max_records: int | None = None) -> gpd.GeoDataFrame:
    """Fetch Denton County parcel geometry from TxGIO using bbox/keyset paging."""
    return _fetch_txgio_county_parcel_geometry("denton", max_records=max_records)
# ---------------------------------------------------------------------------
# Wise County CAD — PACS fixed-width helpers
# ---------------------------------------------------------------------------

def _safe_int(s: str) -> int | None:
    """Strip and convert to int; return None on empty or non-numeric."""
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _safe_float(s: str) -> float | None:
    """Strip and convert to float; return None on empty or non-numeric."""
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _pacs_acres(s: str) -> float | None:
    """Decode a PACS legal_acreage field — 4 IMPLIED decimal places.

    The PACS layout spec ('Appraisal Export Layout - 8.0.31.xlsx', shipped
    inside the Grayson GCAD_Export.zip) defines legal_acreage as numeric(16)
    'Legal Acreage (4 Decimals)'. Live Denton 2024 and Grayson 2025/8.0.31
    exports store it as a zero-padded integer with NO decimal point:
    '0000000000001865' means 0.1865 acres, '0000000000100000' means exactly
    10 acres. Dividing by 10,000 decodes the implied scale. A literal decimal
    point, should some PACS vintage ever emit one, is honored as-is.
    """
    s = s.strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if "." in s else v / 10_000.0


def _wise_extract_working_zip(outer_bytes: bytes, tmpdir: str) -> str:
    """Extract the Wise CAD working ZIP to *tmpdir* and return its path.

    Wise CAD ZIPs use a nested structure: outer ZIP (stored, type=0) →
    inner ZIP (Deflate64, type=9) → PACS TXT files. The outer ZIP members
    are stored, so Python's zipfile can extract the inner ZIP. The inner
    ZIP files use Deflate64 and must be decompressed via system ``unzip``.

    If the outer ZIP has no nested ZIP, the outer ZIP path is returned.
    """
    import io
    import os
    import zipfile

    outer_path = os.path.join(tmpdir, "outer.zip")
    with open(outer_path, "wb") as f:
        f.write(outer_bytes)

    with zipfile.ZipFile(outer_path) as outer_zf:
        inner_names = [n for n in outer_zf.namelist() if n.lower().endswith(".zip")]
        if not inner_names:
            return outer_path
        # Outer members are stored (compress_type=0) — Python can read them
        inner_path = os.path.join(tmpdir, "inner.zip")
        with outer_zf.open(inner_names[0]) as src, open(inner_path, "wb") as dst:
            dst.write(src.read())

    return inner_path


def _wise_find_member(zip_path: str, suffix: str) -> str:
    """Return the member name in *zip_path* whose name ends with *suffix* (case-insensitive).

    Uses Python's zipfile only for the central directory (no decompression).
    """
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

    suffix_upper = suffix.upper()
    match = next((n for n in names if n.upper().endswith(suffix_upper)), None)
    if match is None:
        raise FileNotFoundError(
            f"No member ending with {suffix!r} found. Contents: {names[:10]}"
        )
    return match


def _wise_stream_member(zip_path: str, member: str) -> "subprocess.Popen":
    """Return a Popen that streams *member* from *zip_path* via ``unzip -p``.

    The caller is responsible for reading stdout and calling ``proc.wait()``.
    Requires system ``unzip`` (handles Deflate64 / compression type 9).
    """
    import subprocess

    return subprocess.Popen(
        ["unzip", "-p", zip_path, member],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def _parse_wise_appraisal_info(
    zip_path: str,
    year: int,
    limit: int | None,
) -> list[dict]:
    """Stream-parse APPRAISAL_INFO.TXT from *zip_path*, filtering to real property.

    Keeps only records where:
    - prop_type_cd in ('R', 'MH')   — real property and mobile homes
    - sup_num == '0'                 — certified roll (not supplements)
    - prop_val_yr matches *year*     — requested tax year
    """
    import io

    member = _wise_find_member(zip_path, "_APPRAISAL_INFO.TXT")
    records: list[dict] = []

    proc = _wise_stream_member(zip_path, member)
    try:
        reader = io.TextIOWrapper(proc.stdout, encoding="latin-1", errors="replace")
        for line in reader:
            line = line.rstrip("\r\n")
            if len(line) < _AI_ASSESSED_VAL.stop:
                continue

            prop_type = line[_AI_PROP_TYPE].strip()
            if prop_type not in ("R", "MH"):
                continue
            # sup_num is zero-padded to 12 chars ("000000000000") — compare as int
            if _safe_int(line[_AI_SUP_NUM]) != 0:
                continue
            # prop_val_yr is zero-padded to 5 chars ("02024") — compare as int
            if _safe_int(line[_AI_PROP_VAL_YR]) != year:
                continue

            land_hstd  = _safe_int(line[_AI_LAND_HSTD]) or 0
            land_non   = _safe_int(line[_AI_LAND_NON_HSTD]) or 0
            imprv_hstd = _safe_int(line[_AI_IMPRV_HSTD]) or 0
            imprv_non  = _safe_int(line[_AI_IMPRV_NON_HSTD]) or 0

            situs_num = line[_AI_SITUS_NUM].strip() if len(line) > _AI_SITUS_NUM.stop else ""
            prefx  = line[_AI_SITUS_PREFX].strip()
            street = line[_AI_SITUS_STREET].strip()
            suffix = line[_AI_SITUS_SUFFIX].strip()
            situs  = " ".join(p for p in [situs_num, prefx, street, suffix] if p)

            # Owner mailing address (PACS py_addr_* block — same positions as Grayson).
            _m_line1 = line[_AI_MAIL_ADDR1].strip() if len(line) > _AI_MAIL_ADDR1.stop else ""
            _m_line2 = line[_AI_MAIL_ADDR2].strip() if len(line) > _AI_MAIL_ADDR2.stop else ""
            mail_street = _m_line2 or _m_line1

            records.append({
                "prop_id":           line[_AI_PROP_ID].strip(),
                "prop_val_yr":       str(year),
                "prop_type_cd":      prop_type,
                "geo_id":            line[_AI_GEO_ID].strip(),
                "py_owner_name":     line[_AI_OWNER_NAME].strip(),
                "mail_addr1":        mail_street,
                "mail_city":         line[_AI_MAIL_CITY].strip()  if len(line) > _AI_MAIL_CITY.stop  else "",
                "mail_state":        line[_AI_MAIL_STATE].strip() if len(line) > _AI_MAIL_STATE.stop else "",
                "mail_zip":          line[_AI_MAIL_ZIP].strip()   if len(line) > _AI_MAIL_ZIP.stop   else "",
                "situs_address":     situs,
                "situs_city":        line[_AI_SITUS_CITY].strip(),
                "situs_zip":         line[_AI_SITUS_ZIP].strip(),
                "hood_cd":           line[_AI_HOOD_CD].strip(),
                "land_value":        land_hstd + land_non,
                "improvement_value": imprv_hstd + imprv_non,
                "appraised_val":     _safe_int(line[_AI_APPRAISED_VAL]),
                "assessed_val":      _safe_int(line[_AI_ASSESSED_VAL]),
                "legal_acreage":     _pacs_acres(line[_AI_LEGAL_ACREAGE]),
            })

            if limit and len(records) >= limit:
                break
    finally:
        proc.kill()
        proc.wait()

    return records


def _parse_wise_improvement_detail(
    zip_path: str,
    year: int,
    prop_id_filter: set[str],
) -> pd.DataFrame:
    """Parse APPRAISAL_IMPROVEMENT_DETAIL.TXT and return yr_built + main_area_sqft per prop_id.

    - yr_built: minimum year built across all improvements for the parcel.
    - main_area_sqft: sum of area where imprv_det_type_cd == 'LA' (living area);
      falls back to the largest single improvement if no 'LA' type is present.
    """
    import io

    try:
        member = _wise_find_member(zip_path, "_APPRAISAL_IMPROVEMENT_DETAIL.TXT")
    except FileNotFoundError:
        console.print("[yellow]APPRAISAL_IMPROVEMENT_DETAIL.TXT not found — yr_built and main_area_sqft will be null[/yellow]")
        return pd.DataFrame(columns=["prop_id", "yr_built", "main_area_sqft"])

    agg: dict[str, dict] = {}

    proc = _wise_stream_member(zip_path, member)
    try:
        reader = io.TextIOWrapper(proc.stdout, encoding="latin-1", errors="replace")
        for line in reader:
            line = line.rstrip("\r\n")
            if len(line) < _AID_AREA.stop:
                continue
            # prop_val_yr is zero-padded ("02024") — compare as int
            if _safe_int(line[_AID_PROP_VAL_YR]) != year:
                continue

            prop_id = line[_AID_PROP_ID].strip()
            if prop_id not in prop_id_filter:
                continue

            yr = _safe_int(line[_AID_YR_BUILT])
            area = _safe_float(line[_AID_AREA])
            type_cd = line[_AID_TYPE_CD].strip()

            if prop_id not in agg:
                agg[prop_id] = {"yrs": [], "la_area": 0.0, "max_area": 0.0}
            if yr and yr > 1800:
                agg[prop_id]["yrs"].append(yr)
            if area:
                if type_cd == "LA":
                    agg[prop_id]["la_area"] += area
                agg[prop_id]["max_area"] = max(agg[prop_id]["max_area"], area)
    finally:
        proc.kill()
        proc.wait()

    if not agg:
        return pd.DataFrame(columns=["prop_id", "yr_built", "main_area_sqft"])

    rows = []
    for pid, data in agg.items():
        yr_built = min(data["yrs"]) if data["yrs"] else None
        sqft = data["la_area"] if data["la_area"] > 0 else (data["max_area"] or None)
        rows.append({"prop_id": pid, "yr_built": yr_built, "main_area_sqft": sqft})

    return pd.DataFrame(rows)


def fetch_wise_parcels(year: int = 2024, limit: int | None = None) -> pd.DataFrame:
    """Fetch Wise County CAD parcels from the PACS fixed-width bulk download.

    Downloads the certified all-property ZIP from wise-cad.com, parses
    APPRAISAL_INFO.TXT (9,249 chars/record) via ``unzip -p`` (required for
    Deflate64 / compression type 9), filters to real property (prop_type_cd
    in 'R', 'MH') and certified records (sup_num == 0), then joins
    APPRAISAL_IMPROVEMENT_DETAIL.TXT for yr_built and living area sqft.

    Args:
        year: Tax year (must be in :data:`WISE_DATASET_URLS`).
        limit: Optional row cap for testing (default None = all rows).

    Returns:
        Raw DataFrame with PACS-derived column names, ready for
        :func:`normalize_parcels`.

    Raises:
        ValueError: If *year* is not in :data:`WISE_DATASET_URLS`.
    """
    import os
    import tempfile

    url = WISE_DATASET_URLS.get(year)
    if url is None:
        available = sorted(WISE_DATASET_URLS.keys())
        raise ValueError(
            f"No Wise CAD dataset for year {year}. Available: {available}"
        )

    console.print(f"[dim]Downloading Wise County {year} from {url} ...[/dim]")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    outer_bytes = resp.content
    console.print(f"[dim]Downloaded {len(outer_bytes) / 1e6:.1f} MB[/dim]")

    with tempfile.TemporaryDirectory() as tmpdir:
        working_path = _wise_extract_working_zip(outer_bytes, tmpdir)
        del outer_bytes  # free memory before streaming large files

        console.print("[dim]Parsing APPRAISAL_INFO.TXT ...[/dim]")
        records = _parse_wise_appraisal_info(working_path, year, limit)

        if not records:
            console.print(f"[yellow]No real property records found for Wise County {year}[/yellow]")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        console.print(f"[dim]{len(df):,} real property records parsed[/dim]")

        console.print("[dim]Parsing APPRAISAL_IMPROVEMENT_DETAIL.TXT ...[/dim]")
        prop_ids = set(df["prop_id"])
        imprv_df = _parse_wise_improvement_detail(working_path, year, prop_ids)

    if not imprv_df.empty:
        df = df.merge(imprv_df, on="prop_id", how="left")
    else:
        df["yr_built"] = None
        df["main_area_sqft"] = None

    console.print(f"[green]Wise County {year}: {len(df):,} parcels fetched[/green]")
    return df


# ---------------------------------------------------------------------------
# Tarrant Appraisal District (TAD) — pipe-delimited bulk download
# ---------------------------------------------------------------------------

def fetch_tarrant_parcels(year: int = 2024, limit: int | None = None) -> pd.DataFrame:
    """Fetch Tarrant County (TAD) ALL-PROPERTY parcels from the pipe-delimited bulk download.

    Downloads the certified all-property ZIP (TAD_FULL_URLS) from tad.org (or uses a
    local cache at ``data/tad/tad_full_{year}.zip``), parses the single pipe-delimited
    TXT file, strips whitespace, converts numeric columns, and filters to REAL-PROPERTY
    records (record-prefix ``R`` residential/land + ``C`` commercial) — dropping the
    ~1.1M mineral (``M``) and personal-property (``P``) accounts that are not collateral.
    Results are cached as parquet to avoid re-parsing.

    The download sends a browser User-Agent (:data:`_BULK_HEADERS`): TAD returns HTTP 403
    to the default python-requests UA. The 2025 all-property file is truncated at the
    source (see :data:`TAD_FULL_URLS`), so 2024 is the default/latest usable year.

    Args:
        year: Tax year (must be in :data:`TAD_FULL_URLS`).
        limit: Optional row cap for testing, applied AFTER the real-property filter
            (default None = all real-property rows). NOTE: the raw file is sorted with
            mineral accounts first, so the limit is applied post-filter to ensure it
            yields real-property rows, not minerals.

    Returns:
        Raw DataFrame with TAD column names (lowercase), ready for
        :func:`normalize_parcels`.

    Raises:
        ValueError: If *year* is not in :data:`TAD_FULL_URLS`.
    """
    import zipfile

    url = TAD_FULL_URLS.get(year)
    if url is None:
        available = sorted(TAD_FULL_URLS.keys())
        raise ValueError(
            f"No TAD all-property dataset for year {year}. Available: {available}"
        )

    # Parquet cache
    cache_dir = Path("data/tad")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"tad_full_{year}.parquet"
    if cache_path.exists():
        console.print(f"[dim]Loading TAD {year} from parquet cache: {cache_path}[/dim]")
        df = pd.read_parquet(cache_path)
        if limit:
            df = df.head(limit)
        console.print(f"[green]Tarrant County {year}: {len(df):,} parcels (cached)[/green]")
        return df

    # Check for local ZIP before downloading. TAD 403s a bare python-requests UA, so
    # send a browser User-Agent (_BULK_HEADERS). The all-property ZIP is ~78 MB.
    zip_path = cache_dir / f"tad_full_{year}.zip"
    if not zip_path.exists():
        console.print(f"[dim]Downloading TAD {year} from {url} ...[/dim]")
        resp = requests.get(url, timeout=600, headers=_BULK_HEADERS)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        console.print(f"[dim]Downloaded {len(resp.content) / 1e6:.1f} MB → {zip_path}[/dim]")
    else:
        console.print(f"[dim]Using local TAD ZIP: {zip_path}[/dim]")

    # Extract the single .txt file from the ZIP
    with zipfile.ZipFile(zip_path) as zf:
        txt_names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if not txt_names:
            raise FileNotFoundError(f"No .txt file found in {zip_path}. Contents: {zf.namelist()}")
        txt_name = txt_names[0]
        console.print(f"[dim]Parsing {txt_name} (pipe-delimited)...[/dim]")
        # Read ALL rows (do NOT pass nrows=limit here): the file is sorted minerals-first,
        # so a raw limit would yield mineral rows. The limit is applied after filtering.
        with zf.open(txt_name) as f:
            df = pd.read_csv(
                f,
                sep="|",
                encoding="latin-1",
                dtype=str,
            )

    # Lowercase column names and strip whitespace from all string fields
    df.columns = [c.strip().lower() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    # Filter to REAL-PROPERTY records: record-prefix 'R' (residential/land) + 'C'
    # (commercial). Drop 'M' (mineral) and 'P' (personal) — not foreclosure collateral
    # and ~60% of the file. The all-property file's first column is the record prefix.
    if "rp" in df.columns:
        before = len(df)
        df = df[df["rp"].isin(["R", "C"])].reset_index(drop=True)
        console.print(f"[dim]Real-property filter: {len(df):,} of {before:,} rows (rp in R,C)[/dim]")

    # Convert numeric columns
    numeric_cols = [
        "land_value", "improvement_value", "total_value", "appraised_value",
        "land_acres", "land_sqft", "ag_acres", "ag_value",
        "living_area", "year_built", "garage_capacity",
        "num_bedrooms", "num_bathrooms", "structure_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cache as parquet (filtered real-property set)
    df.to_parquet(cache_path, index=False)
    console.print(f"[dim]Cached → {cache_path}[/dim]")

    if limit:
        df = df.head(limit)
    console.print(f"[green]Tarrant County {year}: {len(df):,} real-property parcels fetched[/green]")
    return df


# ---------------------------------------------------------------------------
# Dallas Central Appraisal District (DCAD) — multi-CSV bulk download
# ---------------------------------------------------------------------------

def fetch_dcad_parcels(year: int = 2024, limit: int | None = None) -> pd.DataFrame:
    """Fetch Dallas County (DCAD) commercial parcels from multi-CSV bulk download.

    Downloads the certified property ZIP from dallascad.org (or uses a local
    cache at ``data/dcad/dcad_{year}.zip``), extracts ACCOUNT_INFO.CSV,
    ACCOUNT_APPRL_YEAR.CSV, and COM_DETAIL.CSV, joins on ACCOUNT_NUM, and
    filters to commercial properties (DIVISION_CD == 'COM' or SPTD_CODE
    starting with 'B' for multifamily). Results are cached as parquet.

    Args:
        year: Tax year (must be in :data:`DCAD_DATASET_URLS`).
        limit: Optional row cap for testing (default None = all rows).

    Returns:
        Raw DataFrame with DCAD column names (lowercase), ready for
        :func:`normalize_parcels`.

    Raises:
        ValueError: If *year* is not in :data:`DCAD_DATASET_URLS`.
    """
    import zipfile

    url = DCAD_DATASET_URLS.get(year)
    if url is None:
        available = sorted(DCAD_DATASET_URLS.keys())
        raise ValueError(
            f"No DCAD dataset for year {year}. Available: {available}"
        )

    # Parquet cache
    cache_dir = Path("data/dcad")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"dcad_com_{year}.parquet"
    if cache_path.exists():
        console.print(f"[dim]Loading DCAD {year} from parquet cache: {cache_path}[/dim]")
        df = pd.read_parquet(cache_path)
        if limit:
            df = df.head(limit)
        console.print(f"[green]Dallas County {year}: {len(df):,} commercial parcels (cached)[/green]")
        return df

    # Check for local ZIP before downloading. Send a browser User-Agent (_BULK_HEADERS):
    # like TAD, DCAD's host rejects the default python-requests UA. NOTE: www.dallascad.org
    # is also unreachable from some datacenter/cloud egress IPs (TCP connect timeout, verified
    # 2026-06-29) — run from a residential/permitted network, or drop a pre-downloaded ZIP at
    # this `zip_path` and the cache-first branch below will parse it without any network call.
    zip_path = cache_dir / f"dcad_{year}.zip"
    if not zip_path.exists():
        console.print(f"[dim]Downloading DCAD {year} from {url} ...[/dim]")
        resp = requests.get(url, timeout=600, headers=_BULK_HEADERS)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        console.print(f"[dim]Downloaded {len(resp.content) / 1e6:.1f} MB → {zip_path}[/dim]")
    else:
        console.print(f"[dim]Using local DCAD ZIP: {zip_path}[/dim]")

    # Read the three target CSVs from the ZIP
    console.print("[dim]Parsing ACCOUNT_INFO.CSV...[/dim]")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("ACCOUNT_INFO.CSV") as f:
            acct_info = pd.read_csv(
                f,
                encoding="latin-1",
                dtype=str,
                quoting=csv.QUOTE_ALL,
            )

    console.print("[dim]Parsing ACCOUNT_APPRL_YEAR.CSV...[/dim]")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("ACCOUNT_APPRL_YEAR.CSV") as f:
            acct_apprl = pd.read_csv(
                f,
                encoding="latin-1",
                dtype=str,
                quoting=csv.QUOTE_ALL,
            )

    console.print("[dim]Parsing COM_DETAIL.CSV...[/dim]")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("COM_DETAIL.CSV") as f:
            com_detail = pd.read_csv(
                f,
                encoding="latin-1",
                dtype=str,
                quoting=csv.QUOTE_ALL,
            )

    # Lowercase all column names
    acct_info.columns = [c.strip().lower() for c in acct_info.columns]
    acct_apprl.columns = [c.strip().lower() for c in acct_apprl.columns]
    com_detail.columns = [c.strip().lower() for c in com_detail.columns]

    # Strip whitespace from string fields
    for tbl in [acct_info, acct_apprl, com_detail]:
        for col in tbl.columns:
            if tbl[col].dtype == object:
                tbl[col] = tbl[col].str.strip()

    # Filter to commercial: DIVISION_CD == 'COM' or SPTD_CODE starting with 'B'
    # Build set of commercial account numbers from both tables
    com_accts_div = set(acct_apprl.loc[
        acct_apprl["division_cd"] == "COM", "account_num"
    ].dropna())
    com_accts_sptd = set(acct_apprl.loc[
        acct_apprl["sptd_code"].str.startswith("B", na=False), "account_num"
    ].dropna())
    com_accts = com_accts_div | com_accts_sptd
    console.print(
        f"[dim]Commercial filter: {len(com_accts_div):,} COM division + "
        f"{len(com_accts_sptd):,} B-prefix SPTD → {len(com_accts):,} unique accounts[/dim]"
    )

    # Filter account_info and account_apprl to commercial accounts
    acct_info = acct_info[acct_info["account_num"].isin(com_accts)].copy()
    acct_apprl = acct_apprl[acct_apprl["account_num"].isin(com_accts)].copy()

    # Join: account_info + account_apprl on account_num
    df = acct_info.merge(acct_apprl, on="account_num", how="inner", suffixes=("", "_apprl"))

    # Left-join COM_DETAIL (only rows with matching account_num)
    com_detail = com_detail[com_detail["account_num"].isin(com_accts)].copy()
    if not com_detail.empty:
        # Aggregate COM_DETAIL: take the first building's details per account
        # (some accounts have multiple buildings; use earliest year_built, sum gross area)
        com_detail["year_built"] = pd.to_numeric(com_detail["year_built"], errors="coerce")
        com_detail["gross_bldg_area"] = pd.to_numeric(com_detail["gross_bldg_area"], errors="coerce")
        com_detail["num_units"] = pd.to_numeric(com_detail["num_units"], errors="coerce")
        com_agg = com_detail.groupby("account_num").agg(
            year_built=("year_built", "min"),
            gross_bldg_area=("gross_bldg_area", "sum"),
            num_units=("num_units", "sum"),
            property_name=("property_name", "first"),
        ).reset_index()
        df = df.merge(com_agg, on="account_num", how="left")
    else:
        df["year_built"] = None
        df["gross_bldg_area"] = None
        df["num_units"] = None
        df["property_name"] = None

    # Convert numeric columns
    numeric_cols = [
        "impr_val", "land_val", "tot_val", "land_ag_exempt", "ag_use_val",
        "hmstd_cap_val", "prev_mkt_val", "gross_bldg_area", "num_units",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Construct full situs_address: "3029 OAK RIDGE PL" from street_num + full_street_name
    if "street_num" in df.columns and "full_street_name" in df.columns:
        sn = df["street_num"].fillna("").astype(str).str.strip()
        fsn = df["full_street_name"].fillna("").astype(str).str.strip()
        df["full_street_name"] = (sn + " " + fsn).str.strip()

    if limit:
        df = df.head(limit)

    # Cache as parquet
    df.to_parquet(cache_path, index=False)
    console.print(f"[dim]Cached → {cache_path}[/dim]")

    console.print(f"[green]Dallas County {year}: {len(df):,} commercial parcels fetched[/green]")
    return df


# ---------------------------------------------------------------------------
# Denton County CAD — PACS fixed-width bulk download (dentoncad.net)
# ---------------------------------------------------------------------------

def fetch_denton_parcels(year: int = 2024, limit: int | None = None) -> pd.DataFrame:
    """Fetch Denton County CAD parcels from the PACS fixed-width bulk download.

    Downloads the certified all-property ZIP from dentoncad.net (or uses a
    local cache at ``data/denton/denton_{year}.zip``), parses
    APPRAISAL_INFO.TXT (9,249 chars/record) and
    APPRAISAL_IMPROVEMENT_DETAIL.TXT (624 chars/record) using the same PACS
    field positions as Wise County (identical layout version 8.0.30).

    Filters to real property (prop_type_cd in 'R', 'MH') and certified
    records (sup_num == 0), then joins improvement detail for yr_built and
    living area sqft.

    Args:
        year: Tax year (must be in :data:`DENTON_DATASET_URLS`).
        limit: Optional row cap for testing (default None = all rows).

    Returns:
        Raw DataFrame with PACS-derived column names, ready for
        :func:`normalize_parcels`.

    Raises:
        ValueError: If *year* is not in :data:`DENTON_DATASET_URLS`.
    """
    import subprocess
    import tempfile
    import zipfile

    url = DENTON_DATASET_URLS.get(year)
    if url is None:
        available = sorted(DENTON_DATASET_URLS.keys())
        raise ValueError(
            f"No Denton CAD dataset for year {year}. Available: {available}"
        )

    # Parquet cache
    cache_dir = Path("data/denton")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"denton_{year}.parquet"
    if cache_path.exists():
        console.print(f"[dim]Loading Denton {year} from parquet cache: {cache_path}[/dim]")
        df = pd.read_parquet(cache_path)
        if limit:
            df = df.head(limit)
        console.print(f"[green]Denton County {year}: {len(df):,} parcels (cached)[/green]")
        return df

    # Check for local ZIP before downloading
    zip_path = cache_dir / f"denton_{year}.zip"
    if not zip_path.exists():
        console.print(f"[dim]Downloading Denton County {year} from {url} ...[/dim]")
        resp = requests.get(url, timeout=600)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        console.print(f"[dim]Downloaded {len(resp.content) / 1e6:.1f} MB → {zip_path}[/dim]")
    else:
        console.print(f"[dim]Using local Denton ZIP: {zip_path}[/dim]")

    # Denton's ZIP uses standard Deflate (not Deflate64 like Wise), so Python
    # zipfile can handle it directly. But we use the same unzip -p streaming
    # approach for consistency and memory efficiency on 4+ GB files.
    #
    # First, check if Python zipfile can read it (compression type 8 = deflate).
    # If not, fall back to system unzip.
    try:
        with zipfile.ZipFile(zip_path) as zf:
            # Verify we can open it — check compression types
            _test = zf.namelist()
        use_python_zip = True
    except (zipfile.BadZipFile, NotImplementedError):
        use_python_zip = False

    console.print("[dim]Parsing APPRAISAL_INFO.TXT ...[/dim]")
    records = _parse_denton_appraisal_info(str(zip_path), year, limit, use_python_zip)

    if not records:
        console.print(f"[yellow]No real property records found for Denton County {year}[/yellow]")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    console.print(f"[dim]{len(df):,} real property records parsed[/dim]")

    console.print("[dim]Parsing APPRAISAL_IMPROVEMENT_DETAIL.TXT ...[/dim]")
    prop_ids = set(df["prop_id"])
    imprv_df = _parse_denton_improvement_detail(str(zip_path), year, prop_ids, use_python_zip)

    if not imprv_df.empty:
        df = df.merge(imprv_df, on="prop_id", how="left")
    else:
        df["yr_built"] = None
        df["main_area_sqft"] = None

    # Cache as parquet
    df.to_parquet(cache_path, index=False)
    console.print(f"[dim]Cached → {cache_path}[/dim]")

    console.print(f"[green]Denton County {year}: {len(df):,} parcels fetched[/green]")
    return df


def _denton_find_member(zip_path: str, suffix: str) -> str:
    """Return the member name in *zip_path* whose name ends with *suffix* (case-insensitive)."""
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

    suffix_upper = suffix.upper()
    match = next((n for n in names if n.upper().endswith(suffix_upper)), None)
    if match is None:
        raise FileNotFoundError(
            f"No member ending with {suffix!r} found in {zip_path}. Contents: {names[:10]}"
        )
    return match


def _denton_stream_member(zip_path: str, member: str, use_python_zip: bool = True):
    """Return a file-like object that streams *member* from *zip_path*.

    If *use_python_zip* is True, uses Python's zipfile (works for standard
    deflate). Otherwise falls back to system ``unzip -p`` (needed for
    Deflate64 / compression type 9).
    """
    import io
    import subprocess
    import zipfile

    if use_python_zip:
        zf = zipfile.ZipFile(zip_path)
        raw = zf.open(member)
        reader = io.TextIOWrapper(raw, encoding="latin-1", errors="replace")
        # Attach the ZipFile so it stays open while reading
        reader._denton_zf = zf
        return reader, None  # (reader, proc=None)
    else:
        proc = subprocess.Popen(
            ["unzip", "-p", zip_path, member],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        reader = io.TextIOWrapper(proc.stdout, encoding="latin-1", errors="replace")
        return reader, proc


def _parse_denton_appraisal_info(
    zip_path: str,
    year: int,
    limit: int | None,
    use_python_zip: bool = True,
) -> list[dict]:
    """Stream-parse APPRAISAL_INFO.TXT from Denton CAD ZIP.

    Uses the same PACS field positions as Wise/Grayson County (PACS 8.0.30/8.0.31
    layouts are byte-identical for all fields used here — verified empirically
    against live Denton 2024 and 2025 exports on 2026-06-28). Filters to real
    property (prop_type_cd in 'R', 'MH'), certified records (sup_num == 0),
    and the requested tax year. Includes imprv_state_cd (e.g. 'B1' for
    apartments, 'B2' for duplexes) for downstream MF filtering.

    Owner mailing address uses the same PACS py_addr_* block as Grayson/Wise:
      py_addr_line1 (693-753): care-of / attention line (often blank)
      py_addr_line2 (753-813): street delivery line (primary)
      py_addr_city  (873-923): city
      py_addr_state (923-973): state
      py_addr_zip   (978-983): 5-digit ZIP
    Street address is in line2 in 2024 exports (line1 blank) and in line1 in
    2025 exports; the ``_m_line2 or _m_line1`` fallback handles both vintages.
    Verified: 99.7%+ non-blank rate across 5,000 certified 2024 records and
    100% across 2,000 certified 2025 records. Absentee-owner sample confirmed:
    mail differs from situs when owner lives off-property.
    """
    member = _denton_find_member(zip_path, "_APPRAISAL_INFO.TXT")
    records: list[dict] = []

    reader, proc = _denton_stream_member(zip_path, member, use_python_zip)
    try:
        for line in reader:
            line = line.rstrip("\r\n")
            if len(line) < _AI_ASSESSED_VAL.stop:
                continue

            prop_type = line[_AI_PROP_TYPE].strip()
            if prop_type not in ("R", "MH"):
                continue
            if _safe_int(line[_AI_SUP_NUM]) != 0:
                continue
            if _safe_int(line[_AI_PROP_VAL_YR]) != year:
                continue

            land_hstd  = _safe_int(line[_AI_LAND_HSTD]) or 0
            land_non   = _safe_int(line[_AI_LAND_NON_HSTD]) or 0
            imprv_hstd = _safe_int(line[_AI_IMPRV_HSTD]) or 0
            imprv_non  = _safe_int(line[_AI_IMPRV_NON_HSTD]) or 0

            situs_num = line[_AI_SITUS_NUM].strip() if len(line) > _AI_SITUS_NUM.stop else ""
            prefx  = line[_AI_SITUS_PREFX].strip()
            street = line[_AI_SITUS_STREET].strip()
            suffix = line[_AI_SITUS_SUFFIX].strip()
            situs  = " ".join(p for p in [situs_num, prefx, street, suffix] if p)

            # Extract state code if the line is long enough (pos 2731+)
            imprv_state_cd = ""
            if len(line) > _AI_IMPRV_STATE_CD.stop:
                imprv_state_cd = line[_AI_IMPRV_STATE_CD].strip()

            # Owner mailing address (PACS py_addr_* block — same positions as Grayson/Wise).
            # py_addr_line1 = attention/care-of (often blank); py_addr_line2 = street.
            # Prefer line2 (street); fall back to line1 (e.g. PO Box when line2 is empty).
            # 2024 exports: street is in line2 (line1 blank).
            # 2025 exports: street is in line1 (line2 blank).
            # The fallback handles both vintages correctly.
            _m_line1 = line[_AI_MAIL_ADDR1].strip() if len(line) > _AI_MAIL_ADDR1.stop else ""
            _m_line2 = line[_AI_MAIL_ADDR2].strip() if len(line) > _AI_MAIL_ADDR2.stop else ""
            mail_street = _m_line2 or _m_line1

            records.append({
                "prop_id":           line[_AI_PROP_ID].strip(),
                "prop_val_yr":       str(year),
                "prop_type_cd":      prop_type,
                "imprv_state_cd":    imprv_state_cd,
                "geo_id":            line[_AI_GEO_ID].strip(),
                "py_owner_name":     line[_AI_OWNER_NAME].strip(),
                "mail_addr1":        mail_street,
                "mail_city":         line[_AI_MAIL_CITY].strip()  if len(line) > _AI_MAIL_CITY.stop  else "",
                "mail_state":        line[_AI_MAIL_STATE].strip() if len(line) > _AI_MAIL_STATE.stop else "",
                "mail_zip":          line[_AI_MAIL_ZIP].strip()   if len(line) > _AI_MAIL_ZIP.stop   else "",
                "situs_address":     situs,
                "situs_city":        line[_AI_SITUS_CITY].strip(),
                "situs_zip":         line[_AI_SITUS_ZIP].strip(),
                "hood_cd":           line[_AI_HOOD_CD].strip(),
                "land_value":        land_hstd + land_non,
                "improvement_value": imprv_hstd + imprv_non,
                "appraised_val":     _safe_int(line[_AI_APPRAISED_VAL]),
                "assessed_val":      _safe_int(line[_AI_ASSESSED_VAL]),
                "legal_acreage":     _pacs_acres(line[_AI_LEGAL_ACREAGE]),
            })

            if limit and len(records) >= limit:
                break
    finally:
        reader.close()
        if hasattr(reader, "_denton_zf"):
            reader._denton_zf.close()
        if proc is not None:
            proc.kill()
            proc.wait()

    return records


def _parse_denton_improvement_detail(
    zip_path: str,
    year: int,
    prop_id_filter: set[str],
    use_python_zip: bool = True,
) -> pd.DataFrame:
    """Parse APPRAISAL_IMPROVEMENT_DETAIL.TXT from Denton CAD ZIP.

    Uses the same PACS field positions as Wise County. Returns yr_built
    (min across improvements) and main_area_sqft (sum of 'LA' type, or
    largest single improvement as fallback) per prop_id.
    """
    try:
        member = _denton_find_member(zip_path, "_APPRAISAL_IMPROVEMENT_DETAIL.TXT")
    except FileNotFoundError:
        console.print("[yellow]APPRAISAL_IMPROVEMENT_DETAIL.TXT not found — yr_built and main_area_sqft will be null[/yellow]")
        return pd.DataFrame(columns=["prop_id", "yr_built", "main_area_sqft"])

    agg: dict[str, dict] = {}

    reader, proc = _denton_stream_member(zip_path, member, use_python_zip)
    try:
        for line in reader:
            line = line.rstrip("\r\n")
            if len(line) < _AID_AREA.stop:
                continue
            if _safe_int(line[_AID_PROP_VAL_YR]) != year:
                continue

            prop_id = line[_AID_PROP_ID].strip()
            if prop_id not in prop_id_filter:
                continue

            yr = _safe_int(line[_AID_YR_BUILT])
            area = _safe_float(line[_AID_AREA])
            type_cd = line[_AID_TYPE_CD].strip()

            if prop_id not in agg:
                agg[prop_id] = {"yrs": [], "la_area": 0.0, "max_area": 0.0}
            if yr and yr > 1800:
                agg[prop_id]["yrs"].append(yr)
            if area:
                if type_cd == "LA":
                    agg[prop_id]["la_area"] += area
                agg[prop_id]["max_area"] = max(agg[prop_id]["max_area"], area)
    finally:
        reader.close()
        if hasattr(reader, "_denton_zf"):
            reader._denton_zf.close()
        if proc is not None:
            proc.kill()
            proc.wait()

    if not agg:
        return pd.DataFrame(columns=["prop_id", "yr_built", "main_area_sqft"])

    rows = []
    for pid, data in agg.items():
        yr_built = min(data["yrs"]) if data["yrs"] else None
        sqft = data["la_area"] if data["la_area"] > 0 else (data["max_area"] or None)
        rows.append({"prop_id": pid, "yr_built": yr_built, "main_area_sqft": sqft})

    return pd.DataFrame(rows)
