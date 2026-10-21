# Housing supply exhibits

Publishable charts on housing starts, completions, permits, household formation
and growth rates for DFW, the other big Texas metros, and U.S. regions.

```bash
uv pip install -e ".[exhibits]"
python scripts/build_supply_exhibits.py                     # public data; no footer credit
python scripts/build_costar_exhibits.py                     # CoStar exports in data/costar/raw
python scripts/build_supply_exhibits.py --brand "Your Name"  # adds a credit bottom-right
```

Output goes to `outputs/supply_exhibits/<date>/` and `outputs/costar_exhibits/<date>/`
(both gitignored). Each exhibit is
written as a PNG (1600px wide, for posting) and an SVG (for editing), alongside
the CSV of the data behind it. `panel_texas_and_peers.csv` holds the full annual
metro panel.

Headlines are computed from the data and fall back to a neutral title when the
claim no longer holds. Still, read them before you post.

## Exhibits

| # | Exhibit | Source |
|---|---------|--------|
| 01 | U.S. multifamily (5+ unit) starts vs. completions | Census/HUD New Residential Construction via FRED |
| 02 | Starts vs. completions, four Census regions | same |
| 03 | Permits by structure type, DFW / Houston / Austin / San Antonio | Census Building Permits Survey |
| 04 | Permits per 1,000 residents vs. U.S. | BPS + Census Population Estimates |
| 05 | Housing supply growth vs. population and job growth | BPS + PEP + BLS CES via FRED |
| 06 | Units permitted per household added | BPS + ACS 1-year |
| 07 | Year-to-date permits vs. prior year | BPS monthly YTD files |
| 08 | Scorecard table | all of the above |
| 09 | Permits per 1,000 residents, Texas vs. peers (ranked) | BPS + PEP |
| 10 | Units permitted per household added, Texas vs. peers | BPS + ACS |
| 11 | Supply growth vs. population growth scatter | BPS + PEP |
| 12 | Year-to-date permit change, Texas vs. peers | BPS |
| 13 | Multifamily permits per resident, 2021–22 vs. latest (dumbbell) | BPS + PEP |

Peers (`PEER_METROS` in `housing_supply.py`): Phoenix, Atlanta, Nashville,
Charlotte, Denver, Orlando, Tampa, Raleigh. Adding a metro takes its CBSA code
plus its FRED total-nonfarm series ID in `FRED_PAYROLL_IDS`.

### CoStar exhibits (`build_costar_exhibits.py`, all cite "Source: CoStar")

| # | Exhibit |
|---|---------|
| 20 | Starts vs. deliveries vs. absorption (trailing 12 months), big four, with forecast |
| 21 | Under construction as % of inventory, big four |
| 22 | Vacancy rate with CoStar forecast |
| 23 | Effective rent growth with CoStar forecast |
| 24 | Apartments delivered per household added: 2015–19, 2020–24, 2025–30F |
| 25 | Pipeline share of inventory, every Texas metro with 10K+ units |
| 26–29 | Submarket pipeline for DFW, Austin, San Antonio, Houston (when exported) |

### Excess supply (`build_excess_supply.py`)

Writes `outputs/excess_supply/<date>/` with exhibits 30–33 and `summary.md`:

| # | Exhibit |
|---|---------|
| 30 | Excess vacant apartments over time, big four, with CoStar forecast |
| 31 | Apartments delivered vs. absorbed since 2020, and the excess left vacant |
| 32 | CoStar deliveries vs. ACS renter households added, 2021–2024 |
| 33 | Excess vacant rentals across all rental housing (ACS), Texas + peers |

"Excess" here means vacant units above a normal vacancy rate:
`inventory × (vacancy − 2015–19 average vacancy)` for CoStar apartments, and
`for-rent units − 2019 rental vacancy rate × rental stock` for all rentals in the
ACS (with 90% margins of error). Renter households in 5+ unit buildings
(ACS B25032) are the public match for CoStar's multifamily inventory.

## Method notes

- **Metro permits** are county BPS files summed to CBSAs with the 2023 OMB
  delineation, so boundaries are constant across every year. The estimates
  include Census imputation for permit offices that didn't report.
- **Population changes** are taken within one PEP vintage, so re-benchmarking to
  the 2020 Census isn't counted as growth.
- **Households** come from ACS 1-year estimates (B25003). Without a
  `CENSUS_API_KEY`, only the keyless summary files are available: 2019 and
  2021 onward (there is no 2020 1-year release). Exhibit 06 starts at 2021
  because the switch to post-2020 Census population controls inflates
  2019→2021 household growth. With a key in `.env`, `cbsa_households()` pulls
  2005–2025 from the API.
- **Metro starts and completions aren't published by Census**; they exist only
  nationally and by region. For metro-level starts and deliveries, use CoStar
  (below).
- FRED is read through its keyless `fredgraph.csv` endpoint.

## CoStar data drop

Put CoStar exports in **`data/costar/raw/<export-date>/`**, for example
`data/costar/raw/2026-10-20/`. The folder is gitignored, so licensed data never
gets committed. Exhibits built from it must cite **"Source: CoStar"**.

Export the following:

- **Property type:** Multifamily (all star ratings, plus 4 & 5 Star separately
  if you can).
- **Geographies:** markets for Dallas-Fort Worth, Houston, Austin, San Antonio,
  any peer metros you want, and National. Add DFW submarkets for the drill-down.
- **Periodicity:** quarterly, 2000 to date, with forecast quarters if available.
- **Columns:** Inventory Units, Under Construction Units, Construction Starts
  Units, Net Delivered Units, Absorption Units, Vacancy Rate, Market Asking
  Rent/Unit, Market Effective Rent/Unit, Market Asking Rent Growth 12 Mo.

Excel or CSV, exactly as CoStar exports it. The loader
(`geostack.costar_market.load_costar`) reads every file in the folder. When
exports overlap, the one with the latest "As Of" wins, so a newer export can
simply be dropped in next to the old ones. CoStar's Data Export caps rows per
file, so split large pulls (e.g. by state or region) into several files.
Don't rename the columns; the loader keys on CoStar's own labels.
