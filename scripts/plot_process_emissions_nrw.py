# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Plot process-emission capture locations for NRW, scaled by Mt/a value for the
given planning horizon.  One map per scenario × planning horizon.

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
from shapely.geometry import MultiPolygon, Polygon

from scripts._helpers import configure_logging, set_scenario_config

# Bounding box for NRW (DEA): [lon_min, lon_max, lat_min, lat_max]
NRW_BOUNDS = [5.75, 9.55, 50.2, 52.65]

NEIGHBOUR_LABELS = [
    (6.15, 52.0, "NIEDERLANDE"),
    (6.03, 50.34, "BELGIEN"),
    (7.07, 50.26, "RHEINLAND-PFALZ"),
    (8.9, 51.05, "HESSEN"),
    (7.75, 52.55, "NIEDERSACHSEN"),
]


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

    # Set map extent
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

    # NUTS3 labels
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
            "plot_process_emissions_nrw",
            opts="",
            clusters="adm",
            sector_opts="",
            planning_horizons="2035",
            configfiles=["config/config.nrw.yaml"],
            run="endo-grid___CCS-Exp__offshore-co2",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    settings = snakemake.params.plotting["process_emissions_nrw"]
    figsize = ast.literal_eval(settings["figsize"])
    dpi = settings["dpi"]
    circle_factor = settings["circle_factor"]
    circle_color = settings["circle_color"]
    circle_alpha = settings["circle_alpha"]
    circle_sizes = settings["circle_sizes"]

    planning_horizon = str(snakemake.wildcards.planning_horizons)

    regions = gpd.read_file(snakemake.input.regions).set_index("name")
    proj = ccrs.Mercator()

    # Load and filter process-emissions data
    df = pd.read_csv(snakemake.input.process_emissions)
    df = df[df[planning_horizon] > 0].copy()

    fig, ax = plot_nrw_basemap(regions, snakemake.input.nuts3_shapes, proj)

    # Scatter circles at each location, area ∝ value (Mt/a)
    sc = ax.scatter(
        df["longitude"],
        df["latitude"],
        s=df[planning_horizon] * circle_factor,
        c=circle_color,
        alpha=circle_alpha,
        transform=ccrs.PlateCarree(),
        zorder=6,
        edgecolors="white",
        linewidths=0.3,
    )

    # Year annotation
    ax.text(
        0.02, 0.97, planning_horizon,
        transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="left",
        color="#333333", zorder=7,
    )

    # Bubble-size legend
    legend_kw = dict(
        loc="lower right",
        frameon=False,
        alignment="left",
        title_fontproperties={"weight": "bold"},
    )
    handles = [
        plt.scatter(
            [], [],
            s=s * circle_factor,
            c=circle_color,
            alpha=circle_alpha,
            edgecolors="white",
            linewidths=0.3,
            label=f"{s} Mt/a",
        )
        for s in circle_sizes
    ]
    ax.legend(
        handles=handles,
        title="Prozessemissionen",
        bbox_to_anchor=(0.99, 0.02),
        **legend_kw,
    )

    fig.savefig(snakemake.output.map, bbox_inches="tight")
    fig.savefig(snakemake.output.png, bbox_inches="tight", dpi=150)
    plt.close(fig)
