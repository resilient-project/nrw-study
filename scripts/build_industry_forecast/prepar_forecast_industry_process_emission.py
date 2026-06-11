#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare FORECAST process-emission data for the NRW workflow.

The script reads the raw FORECAST process-emissions workbook from
`data/forecast_industry/industrial_process_emission_forecast.xlsx`, validates
year and coordinate columns, assigns each record to a NUTS3 region based on
longitude and latitude, harmonises metadata columns, and exports
scenario-specific CSV files.

Outputs are written in the layout consumed by
`build_industrial_energy_demand_per_node_forecast.py`:
`data/forecast_industry/<scenario>/process_emissions.csv`.
An additional Excel workbook with one sheet per scenario is also written for
inspection.
"""

import os
import warnings
from pathlib import Path
from typing import List

import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError as exc:
    raise ImportError(
        "This script requires geopandas and shapely.\n"
        "Install with e.g.: pip install geopandas shapely"
    ) from exc


# =============================================================================
# 1. BASIC CONFIGURATION AND FILE PATHS
# =============================================================================

# Name of the sheet containing the process-emissions table.
# Set to None to use the first sheet.
TARGET_SHEET = "Process-emissions"

# Column names
SCENARIO_COL = "Scenario"
LONGITUDE_COL = "longitude"
LATITUDE_COL = "latitude"

# Root directory:
# - If run as a script: go two levels up from this file.
# - If run interactively: use current working directory.
try:
    ROOT = Path(__file__).resolve().parents[2]
except NameError:
    ROOT = Path.cwd().resolve()


# Input and output paths used by the FORECAST preprocessing workflow
xlsx_infile = (
    ROOT / "data" / "forecast_industry" / "industrial_process_emission_forecast.xlsx"
)
if "snakemake" in globals():
    nuts3_file = Path(snakemake.input.nuts3_shapes)
else:
    nuts3_file = ROOT / "resources" / "nuts3_shapes.geojson"

xlsx_outfile = (
    ROOT
    / "data"
    / "forecast_industry"
    / "industrial_process_emissions_by_scenario.xlsx"
)
csv_outdir = ROOT / "data" / "forecast_industry"

# Validate input and ensure output folders exist
if not xlsx_infile.exists():
    raise FileNotFoundError(f"Input Excel not found: {xlsx_infile}")

if not nuts3_file.exists():
    warnings.warn(
        f"NUTS3 geometry file not found: {nuts3_file}\n"
        f"Proceeding WITHOUT NUTS3 assignment.",
        UserWarning,
    )

xlsx_outfile.parent.mkdir(parents=True, exist_ok=True)
csv_outdir.mkdir(parents=True, exist_ok=True)

# Meta columns to prioritise in output
META_COLS = [
    "Scenario",
    "Country",
    "Energy_carrier",
    "Sector",
    "Region",        # or "NUTS3_code" if you kept that name
    "Name_Region",   # or "NUTS3_name"
    "Subsector",
    "Application",
    "Technologie",
]

# =============================================================================
# 2. TRANSLATION DICTIONARY
# =============================================================================
# Only a subset of columns needs renaming for this workflow.
translate = {
    # --- Column headers ---
    "Szenario": SCENARIO_COL,
    "Land": "Country",
    "Prozess": "Process",
    "Sektor": "Sector",
    "Subsektor": "Subsector",
    "Einheit": "Unit",
    "Energieträger": "Energy_carrier",
    "Anwendung": "Application",
    "Lan": LONGITUDE_COL,
    "long": LATITUDE_COL,
}

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def safe_sheet_name(name: str) -> str:
    """Return an Excel-safe sheet name (no illegal characters, max length 31)."""
    invalid_chars = "[]:*?/\\"
    safe = str(name)
    for ch in invalid_chars:
        safe = safe.replace(ch, "_")
    safe = safe[:31]
    return safe or "Sheet"

def detect_year_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect year columns in the DataFrame.

    A year column is defined here as a column name that:
    - is all digits
    - represents a year between 1900 and 2100.
    """
    year_cols = []
    for col in df.columns:
        col_str = str(col).strip()
        if col_str.isdigit():
            year_int = int(col_str)
            if 1900 <= year_int <= 2100:
                year_cols.append(col_str)
    return year_cols

def assign_nuts3_from_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign NUTS3 regions based on longitude and latitude.

    - Requires LONGITUDE_COL and LATITUDE_COL to exist in df.
    - Uses a spatial join with nuts3_file.
    - Adds 'NUTS3_code' and 'NUTS3_name' columns.
    - If some points fall outside all polygons, assigns them to the
      *nearest* NUTS3 polygon and prints a warning.
    """
    if not nuts3_file.exists():
        print("[NUTS3] Geometry file is missing. Skipping NUTS3 assignment.")
        df["NUTS3_code"] = pd.NA
        df["NUTS3_name"] = pd.NA
        return df

    if LONGITUDE_COL not in df.columns or LATITUDE_COL not in df.columns:
        warnings.warn(
            f"Columns '{LONGITUDE_COL}' and/or '{LATITUDE_COL}' not found. "
            f"Skipping NUTS3 assignment.",
            UserWarning,
        )
        df["NUTS3_code"] = pd.NA
        df["NUTS3_name"] = pd.NA
        return df

    # --- basic coordinate checks ---
    lon = pd.to_numeric(df[LONGITUDE_COL], errors="coerce")
    lat = pd.to_numeric(df[LATITUDE_COL], errors="coerce")

    total_rows = len(df)
    missing_coords = lon.isna().sum() + lat.isna().sum()
    print(f"[NUTS3] Total rows: {total_rows:,}")
    print(
        f"[NUTS3] Rows with missing lon/lat values (counting lon+lat): {missing_coords:,}"
    )

    out_of_range = ((lon < -180) | (lon > 180) | (lat < -90) | (lat > 90)).sum()
    if out_of_range > 0:
        warnings.warn(
            f"[NUTS3] Found {out_of_range:,} rows with lon/lat outside valid ranges.",
            UserWarning,
        )

    # --- create GeoDataFrame of points ---
    print("[NUTS3] Creating point geometries from coordinates...")
    df_points = df.copy()
    df_points[LONGITUDE_COL] = lon
    df_points[LATITUDE_COL] = lat

    geometry = [
        Point(xy) if pd.notna(xy[0]) and pd.notna(xy[1]) else None
        for xy in zip(df_points[LONGITUDE_COL], df_points[LATITUDE_COL])
    ]
    gdf_points = gpd.GeoDataFrame(df_points, geometry=geometry, crs="EPSG:4326")

    # --- load NUTS3 geometries ---
    print(f"[NUTS3] Loading NUTS3 regions from: {nuts3_file}")
    nuts3 = gpd.read_file(nuts3_file)

    # CRS: your file uses CRS84, treat as EPSG:4326
    if nuts3.crs is None:
        nuts3.set_crs("EPSG:4326", inplace=True)
    else:
        nuts3 = nuts3.to_crs("EPSG:4326")

    # In the PyPSA resources, `index` is the region code and `name` the label.
    nuts3_cols = []
    if "index" in nuts3.columns:
        nuts3_cols.append("index")
    if "name" in nuts3.columns:
        nuts3_cols.append("name")
    if "country" in nuts3.columns:
        nuts3_cols.append("country")
    nuts3_cols.append("geometry")

    nuts3 = nuts3[nuts3_cols]


    # --- spatial join: point-in-polygon ---
    print("[NUTS3] Performing spatial join (point-in-polygon)...")
    joined = gpd.sjoin(gdf_points, nuts3, how="left", predicate="within")

    # Rename to workflow-specific column names.
    if "index" in joined.columns:
        joined.rename(columns={"index": "NUTS3_code"}, inplace=True)
    else:
        joined["NUTS3_code"] = pd.NA

    if "name" in joined.columns:
        joined.rename(columns={"name": "NUTS3_name"}, inplace=True)
    else:
        joined["NUTS3_name"] = pd.NA

    # Country: take from NUTS3 'country' attribute where available
    if "country" in joined.columns:
        if "Country" in joined.columns:
            # Fill existing Country column if it has missing values
            joined["Country"] = joined["Country"].fillna(joined["country"])
        else:
            joined.rename(columns={"country": "Country"}, inplace=True)
    else:
        # Fallback if the NUTS3 layer has no `country` field.
        if "Country" not in joined.columns:
            joined["Country"] = pd.NA


    # --- stats before nearest-neighbour fix ---
    assigned_initial = joined["NUTS3_code"].notna().sum()
    print(
        f"[NUTS3] Rows with assigned NUTS3_code after polygon match: "
        f"{assigned_initial} ({assigned_initial / total_rows * 100:.2f}% of total)"
    )

    # --- handle unmatched points: assign nearest NUTS3 ---
    missing_mask = joined["NUTS3_code"].isna()
    missing_count = int(missing_mask.sum())

    if missing_count > 0:
        warnings.warn(
            f"[NUTS3] {missing_count} row(s) could not be matched to any polygon. "
            f"Assigning nearest NUTS3 region based on geometry distance.",
            UserWarning,
        )

        # For each unmatched row, compute distance to all NUTS3 geometries
        for idx, row in joined[missing_mask].iterrows():
            point_geom = row.get("geometry", None)
            if point_geom is None or point_geom.is_empty:
                # cannot assign if geometry is missing
                continue

            distances = nuts3.geometry.distance(point_geom)
            nearest_idx = distances.idxmin()
            nearest_code = (
                nuts3.at[nearest_idx, "index"] if "index" in nuts3.columns else pd.NA
            )
            nearest_name = (
                nuts3.at[nearest_idx, "name"] if "name" in nuts3.columns else pd.NA
            )
            nearest_dist = float(distances[nearest_idx])

            joined.at[idx, "NUTS3_code"] = nearest_code
            joined.at[idx, "NUTS3_name"] = nearest_name

            print(
                f"[NUTS3] WARNING: Row {idx} at "
                f"({row[LONGITUDE_COL]}, {row[LATITUDE_COL]}) "
                f"was outside all polygons -> assigned to nearest "
                f"'{nearest_name}' ({nearest_code}), "
                f"distance ~ {nearest_dist:.4f} degrees."
            )

    # --- final stats ---
    assigned_final = joined["NUTS3_code"].notna().sum()
    print(
        f"[NUTS3] Rows with assigned NUTS3_code after nearest-assignment: "
        f"{assigned_final} ({assigned_final / total_rows * 100:.2f}% of total)"
    )

    # Clean up join helper columns
    if "index_right" in joined.columns:
        joined.drop(columns="index_right", inplace=True)

    # Drop geometry if you prefer a plain DataFrame
    if "geometry" in joined.columns:
        joined = pd.DataFrame(joined.drop(columns="geometry"))

    return joined


# =============================================================================
# 4. LOAD EXCEL FILE AND APPLY TRANSLATION
# =============================================================================

print("\n=== Step 1: Loading Excel workbook ===")
xls = pd.ExcelFile(xlsx_infile, engine="openpyxl")
available_sheets = xls.sheet_names
print(f"Available sheets: {available_sheets}")

if TARGET_SHEET is not None and TARGET_SHEET in available_sheets:
    print(f"Reading main sheet: '{TARGET_SHEET}'")
    df = pd.read_excel(xls, sheet_name=TARGET_SHEET)
else:
    # fallback: use first sheet
    warnings.warn(
        f"Sheet '{TARGET_SHEET}' not found. "
        f"Using first sheet '{available_sheets[0]}' instead.",
        UserWarning,
    )
    df = pd.read_excel(xls, sheet_name=available_sheets[0])

print(f"Loaded DataFrame shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

# Strip whitespace from column names
df.columns = [str(c).strip() for c in df.columns]

# Apply translation to the subset of column headers used by the workflow.
rename_map = {col: translate[col] for col in df.columns if col in translate}
df.rename(columns=rename_map, inplace=True)


# =============================================================================
# 5. VALIDATION OF NUMERIC YEAR COLUMNS
# =============================================================================

print("\n=== Step 2: Detecting and validating year columns ===")
year_cols = detect_year_columns(df)

if not year_cols:
    warnings.warn(
        "No year columns detected (expected 4-digit headers like 2020..2100).",
        UserWarning,
    )
else:
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors="coerce")
    print(f"Detected {len(year_cols)} year columns: {year_cols[0]} .. {year_cols[-1]}")

    # Missing values across numeric cells
    missing_share = df[year_cols].isna().mean().mean() * 100
    print(f"Missing value share across numeric cells: {missing_share:.2f}%")

    # Negative values
    negative_mask = df[year_cols] < 0
    negative_rows = negative_mask.any(axis=1)
    num_neg = negative_rows.sum()
    print(f"Negative entries (rows with at least one negative value): {num_neg:,}")
    if num_neg:
        print("Showing up to 5 rows with negatives:")
        for idx, row in df.loc[negative_rows].head(5).iterrows():
            bad_years = [c for c in year_cols if row[c] < 0]
            print(f"  Row {idx}: negative in years {bad_years}")

    # Zero-only rows (sum across all year columns equals zero)
    zero_only_count = (df[year_cols].sum(axis=1) == 0).sum()
    print(f"Rows with all zero values across year columns: {zero_only_count:,}")


# =============================================================================
# 6. SCENARIO CHECK
# =============================================================================

print("\n=== Step 3: Scenario check ===")
if SCENARIO_COL in df.columns:
    scenarios = sorted(df[SCENARIO_COL].dropna().astype(str).unique())
    if len(scenarios) > 1:
        warnings.warn(
            f"The dataset contains {len(scenarios)} scenarios "
            f"({', '.join(scenarios[:5])}"
            f"{'...' if len(scenarios) > 5 else ''}).",
            UserWarning,
        )
    else:
        print(f"Single scenario detected: {scenarios[0]}")
else:
    print(
        f"Warning: No '{SCENARIO_COL}' column found. Treating data as one scenario 'All'."
    )
    SCENARIO_COL_MISSING = True
    df[SCENARIO_COL] = "All"
    scenarios = ["All"]

# =============================================================================
# 7. ASSIGN NUTS3 REGIONS FROM LONGITUDE/LATITUDE
# =============================================================================

print("\n=== Step 4: Assigning NUTS3 regions from coordinates ===")
df = assign_nuts3_from_coordinates(df)

# =============================================================================
# 8. HARMONISE META COLUMNS (Country, Sector, Subsector, Application)
# =============================================================================

# Sector is fixed to `Industry` for this workflow.
df["Sector"] = "Industry"

# Use the original `sector` column as `Subsector` when available.
if "sector" in df.columns:
    df["Subsector"] = df["sector"]
else:
    df["Subsector"] = pd.NA

# Align NUTS3 columns with the naming expected by the downstream forecast rule.
if "NUTS3_code" in df.columns and "Region" not in df.columns:
    df.rename(columns={"NUTS3_code": "Region"}, inplace=True)

if "NUTS3_name" in df.columns and "Name_Region" not in df.columns:
    df.rename(columns={"NUTS3_name": "Name_Region"}, inplace=True)

# Application derived from Unit (e.g. "CO2-Abscheidung [Mt/a]")
if "Unit" in df.columns:
    # Initialise if not present.
    if "Application" not in df.columns:
        df["Application"] = pd.NA

    # Tag rows describing CO2 capture.
    mask_capture = df["Unit"].astype(str).str.contains("CO2-Abscheidung", case=False, na=False)
    df.loc[mask_capture, "Application"] = "CO2 capture"
else:
    if "Application" not in df.columns:
        df["Application"] = pd.NA

# Use a dedicated placeholder if the source workbook has no energy-carrier column.
if "Energy_carrier" not in df.columns:
    df["Energy_carrier"] = "process_emissions"

# Add `Technologie` as an empty placeholder column if missing.
if "Technologie" not in df.columns:
    df["Technologie"] = pd.NA

# 8b. CLEAN UP UNNEEDED COLUMNS
# =============================================================================

# Remove columns that are no longer needed
cols_to_drop = []

# Remove the original sector column after copying it to `Subsector`.
if "sector" in df.columns:
    cols_to_drop.append("sector")

# Keep coordinate columns in the export for traceability.
# if "latitude" in df.columns:
#     cols_to_drop.append("latitude")
# if "longitude" in df.columns:
#     cols_to_drop.append("longitude")
# if "index_right0" in df.columns:
#     cols_to_drop.append("index_right0")

# Drop them only if they exist
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

# =============================================================================
# 8c. REORDER COLUMNS: META FIRST, THEN EVERYTHING ELSE
# =============================================================================


# Make sure all listed meta columns exist (if not already created earlier)
for col in META_COLS:
    if col not in df.columns:
        df[col] = pd.NA

current_cols = list(df.columns)

# Meta columns in this exact order, but only those that actually exist.
meta_first = [c for c in META_COLS if c in current_cols]

# Keep all remaining columns after the metadata block.
other_cols = [c for c in current_cols if c not in meta_first]

# final column order: meta first, then the rest
df = df[meta_first + other_cols]


# =============================================================================
# 9. SIMPLE STATISTICS
# =============================================================================

print("\n=== Step 5: Simple statistics ===")
if year_cols:
    # Total emissions per scenario and year
    total_by_scenario = df.groupby(SCENARIO_COL)[year_cols].sum()
    print("\nTotal emissions by scenario (first 5 rows):")
    print(total_by_scenario.head())

    # Total emissions per assigned region.
    if "Region" in df.columns:
        total_by_region = (
            df.groupby("Region")[year_cols]
            .sum()
            .sort_values(by=year_cols[-1], ascending=False)
        )
        print("\nTotal emissions by Region (top 5 by last year):")
        print(total_by_region.head())
else:
    print("No year columns found; skipping numeric statistics.")


# =============================================================================
# 10. EXPORT TO EXCEL AND PER-SCENARIO CSVS
# -----------------------------------------------------------------------------
# Export one Excel sheet per scenario and one CSV per scenario in the
# Snakemake input layout under data/forecast_industry/<scenario>/.
# =============================================================================

print("\n=== Step 6: Exporting results ===")

# Ensure we have a list of scenarios again (in case it changed)
scenarios = sorted(df[SCENARIO_COL].dropna().astype(str).unique())

# --- Multi-sheet Excel (one sheet per scenario) ---
with pd.ExcelWriter(xlsx_outfile, engine="openpyxl") as writer:
    for scen in scenarios:
        df_scen = df[df[SCENARIO_COL].astype(str) == scen]
        sheet_name = safe_sheet_name(scen)
        df_scen.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"  Added sheet '{sheet_name}' with {len(df_scen):,} rows")

print(f"\nWrote Excel with {len(scenarios)} sheet(s): {xlsx_outfile}")

# --- Per-scenario CSVs in the Snakemake layout ---
for scen in scenarios:
    df_scen = df[df[SCENARIO_COL].astype(str) == scen]
    scenario_dir = csv_outdir / str(scen)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    csv_out = scenario_dir / "process_emissions.csv"
    df_scen.to_csv(csv_out, index=False, encoding="utf-8")
    print(f"  Wrote CSV for scenario '{scen}': {csv_out}")

print("\nAll done. Process emission forecast preprocessing completed successfully.\n")
