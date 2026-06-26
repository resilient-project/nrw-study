# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Plot process-emission locations for NRW by subsector, scaled by emission
value (kt/a) for the given planning horizon. Circles colored by subsector.
One map per scenario x planning horizon.

Matches reference figure layout:
- Subsector color legend (top-left, inside map)
- Bubble-size legend (top-right, inside map)
- Municipality labels only within NRW
- No labels outside NRW bounds

The forecast-industry CSV is selected via the scenario mapping in
config.industry.forecast_industry.scenario_mapping.
"""

import ast
import re
import textwrap

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import MultiPolygon, Polygon

from scripts._helpers import configure_logging, set_scenario_config

# Bounding box for NRW (DEA): [lon_min, lon_max, lat_min, lat_max]
NRW_BOUNDS = [5.75, 9.55, 50.2, 52.65]

NEIGHBOUR_LABELS = [
    (6.15, 52.0,  "NIEDERLANDE"),
    (6.03, 50.34, "BELGIEN"),
    (7.07, 50.26, "RHEINLAND-PFALZ"),
    (8.9,  51.05, "HESSEN"),
    (7.75, 52.55, "NIEDERSACHSEN"),
]

# Subsector colors + German names matching reference figure
# English data name  →  (hex color,  German legend label)
SUBSECTOR_STYLE = {
    "Cement clinker":                  ("#7B5B3A", "Klinker (Zement)"),
    "Iron and steel":                  ("#FFD700", "Metallerzeugung"),
    "Refinery":                        ("#708090", "Raffinerien"),
    "Chemical park":                   ("#6CA0DC", "Grundstoffchemie"),
    "Steam cracker":                   ("#D4C5A9", "Papiergewerbe"),
    "Lime":                            ("#C8A882", "Kalk"),
    "municipal waste incinerator.":    ("#4A5A5A", "Müllverbrennung"),
}


def _clean_nuts3_name(name: str) -> str:
    return re.sub(r",\s*Kreisfreie\s+Stadt$", "", name).strip()


def _exterior_only(geom):
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])


def plot_nrw_basemap(regions, nuts3_path: str, proj) -> tuple:
    """Draw region background, DEA boundaries, NUTS3 labels, neighbour labels."""
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})

    # Modelled regions background
    regions_proj = regions.to_crs(proj.proj4_init)
    regions_proj.plot(
        ax=ax, facecolor="#f5f5f5", edgecolor="#aaaaaa", linewidth=0.4, zorder=1
    )

    # Set map extent to NRW only
    ax.set_extent(NRW_BOUNDS, crs=ccrs.PlateCarree())
    ax.set_facecolor("white")

    # DEA (NRW) internal boundaries
    dea_regions = regions[regions.index.str.startswith("DEA")].to_crs(proj.proj4_init)
    dea_regions.boundary.plot(ax=ax, edgecolor="grey", linewidth=0.5, zorder=3)

    # NRW outer boundary (no interior holes)
    dea_outer = dea_regions.dissolve()
    gpd.GeoDataFrame(
        geometry=dea_outer.geometry.apply(_exterior_only), crs=dea_outer.crs
    ).boundary.plot(ax=ax, edgecolor="black", linewidth=1.0, zorder=4)

    # NUTS3 labels (NRW only)
    nuts3 = gpd.read_file(nuts3_path)
    dea_nuts3 = nuts3[nuts3["index"].str.startswith("DEA")].to_crs(proj.proj4_init)
    for _, row in dea_nuts3.iterrows():
        geom = row.geometry
        centroid = geom.centroid
        name = _clean_nuts3_name(row["name"])
        bbox_width = geom.bounds[2] - geom.bounds[0]
        fontsize = max(4.0, min(7.0, bbox_width / 9_000))
        wrapped = "\n".join(textwrap.wrap(name, width=12, break_long_words=False))
        ax.text(
            centroid.x, centroid.y, wrapped,
            fontsize=fontsize, ha="center", va="center",
            color="#333333", zorder=5, clip_on=True,
            multialignment="center", fontweight="bold", alpha=0.5,
        )

    # Neighbour labels
    geo_crs = ccrs.PlateCarree()
    for lon, lat, label in NEIGHBOUR_LABELS:
        ax.text(
            lon, lat, label,
            transform=geo_crs, fontsize=8.5, ha="center", va="center",
            color="#555555", style="italic", fontweight="bold",
            alpha=0.6, zorder=5, clip_on=True,
        )

    return fig, ax


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_process_emissions_nrw_subsectors",
            opts="",
            clusters="adm",
            sector_opts="",
            planning_horizons="2045",
            configfiles=["config/config.nrw.yaml"],
            run="endo-grid___CCS-Exp__offshore-co2",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    settings = snakemake.params.plotting["process_emissions_nrw_subsectors"]
    figsize              = ast.literal_eval(settings["figsize"])
    dpi                  = settings["dpi"]
    circle_factor        = settings["circle_factor"]
    circle_alpha         = settings["circle_alpha"]
    label_threshold_mt   = settings["label_threshold_mt"]
    circle_sizes         = settings["circle_sizes"]

    # Optional: show scenario name + year annotation on map
    show_scenario_year = settings["show_scenario_year"]

    # Optional: rename FORECAST scenario names to short display labels
    # e.g. Orientierungsszenario_Strom -> HtA, Industrie-CCS -> HtA+
    scenario_display_names = settings["scenario_display_names"]

    planning_horizon = str(snakemake.wildcards.planning_horizons)
    run_name         = str(snakemake.wildcards.run)

    # Resolve the FORECAST scenario name for this run
    forecast_scenario = snakemake.config["industry"]["forecast_industry"][
        "scenario_mapping"
    ].get(run_name, run_name)

    # Apply optional display name (e.g. "HtA", "HtA+")
    display_name = scenario_display_names.get(forecast_scenario, forecast_scenario)

    regions = gpd.read_file(snakemake.input.regions).set_index("name")
    proj    = ccrs.Mercator()

    # Load and filter process-emissions data
    df = pd.read_csv(snakemake.input.process_emissions)
    df = df[df[planning_horizon] > 0].copy()

    fig, ax = plot_nrw_basemap(regions, snakemake.input.nuts3_shapes, proj)

    # ── Plot circles per subsector ────────────────────────────────────────
    subsectors_present = [s for s in SUBSECTOR_STYLE if s in df["Subsector"].values]

    for subsector in subsectors_present:
        df_sub = df[df["Subsector"] == subsector]
        color, _label = SUBSECTOR_STYLE[subsector]

        ax.scatter(
            df_sub["longitude"],
            df_sub["latitude"],
            s=df_sub[planning_horizon] * circle_factor,
            c=color,
            alpha=circle_alpha,
            transform=ccrs.PlateCarree(),
            zorder=6,
            edgecolors="white",
            linewidths=0.5,
        )

        # City labels only for major NRW emitters (>1 Mt/a) within NRW bounds
        for _, row in df_sub.iterrows():
            lon, lat = row["longitude"], row["latitude"]
            in_nrw_bounds = (
                NRW_BOUNDS[0] <= lon <= NRW_BOUNDS[1]
                and NRW_BOUNDS[2] <= lat <= NRW_BOUNDS[3]
            )
            if row[planning_horizon] > 1.0 and in_nrw_bounds:
                ax.text(
                    lon, lat,
                    _clean_nuts3_name(row["Name_Region"]),
                    fontsize=7, ha="center", va="center",
                    transform=ccrs.PlateCarree(),
                    zorder=8, color="black", alpha=0.75,
                )

    # ── Optional: scenario name + year annotation ────────────────────────
    # Enabled via plotting.process_emissions_nrw_subsectors.show_scenario_year: true
    # Display name controlled via scenario_display_names in the same config block
    if show_scenario_year:
        ax.text(
            0.02, 0.02,
            f"{display_name}  |  {planning_horizon}",
            transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            va="bottom", ha="left",
            color="#333333", zorder=9,
        )

    # ── Legend 1: Industry sector (top-left, inside map) ─────────────────
    sector_handles = [
        Patch(facecolor=color, edgecolor="grey", linewidth=0.4, label=label)
        for subsector, (color, label) in SUBSECTOR_STYLE.items()
        if subsector in df["Subsector"].values
    ]
    legend1 = ax.legend(
        handles=sector_handles,
        title="Industry Sector",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        framealpha=0.85,
        edgecolor="grey",
        fontsize=7.5,
        title_fontproperties={"weight": "bold", "size": 8},
    )
    ax.add_artist(legend1)

    # ── Legend 2: Bubble sizes (top-right, inside map) ───────────────────
    size_handles = [
        Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor="#888888",
            markersize=sz,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=label,
            linestyle="None",
        )
        for sz, label in [(7, "100 kt/a"), (13, "500 kt/a")]
    ]
    ax.legend(
        handles=size_handles,
        title="CO2 Emissionen",
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        frameon=True,
        framealpha=0.85,
        edgecolor="grey",
        fontsize=7.5,
        title_fontproperties={"weight": "bold", "size": 8},
        ncol=2,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(snakemake.output.map, bbox_inches="tight")
        fig.savefig(snakemake.output.png, bbox_inches="tight", dpi=150)

    plt.show()
    plt.close(fig)
