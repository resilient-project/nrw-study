# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Plot industry energy demand for NRW (DEA) NUTS3 regions as proportional pie
charts, one per planning horizon.  Pie radius scales with sqrt(regional total
TWh); slices are coloured by energy carrier.

The forecast-industry CSV is selected via
config.industry.forecast_industry.scenario_mapping.
"""

import ast
import re
import textwrap

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

from scripts._helpers import configure_logging, set_scenario_config

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


def _draw_basemap(regions, dea_nuts3_proj, proj, figsize):
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})

    regions_proj = regions.to_crs(proj.proj4_init)
    regions_proj.plot(
        ax=ax, facecolor="#f5f5f5", edgecolor="#aaaaaa", linewidth=0.4, zorder=1
    )
    ax.set_extent(NRW_BOUNDS, crs=ccrs.PlateCarree())
    ax.set_facecolor("white")

    dea_regions = regions[regions.index.str.startswith("DEA")].to_crs(proj.proj4_init)
    dea_regions.boundary.plot(ax=ax, edgecolor="grey", linewidth=0.5, zorder=3)

    dea_outer = dea_regions.dissolve()
    gpd.GeoDataFrame(
        geometry=dea_outer.geometry.apply(_exterior_only), crs=dea_outer.crs
    ).boundary.plot(ax=ax, edgecolor="black", linewidth=1.0, zorder=4)

    # NUTS3 labels (reduced alpha so they don't clash with pies)
    for _, row in dea_nuts3_proj.iterrows():
        geom = row.geometry
        centroid = geom.centroid
        name = _clean_nuts3_name(row["name"])
        bbox_width = geom.bounds[2] - geom.bounds[0]
        fontsize = max(3.5, min(6.0, bbox_width / 9_000))
        wrapped = "\n".join(textwrap.wrap(name, width=12, break_long_words=False))
        ax.text(
            centroid.x, centroid.y, wrapped,
            fontsize=fontsize, ha="center", va="center",
            color="#333333", zorder=8, clip_on=True,
            multialignment="center", fontweight="bold", alpha=0.3,
        )

    geo_crs = ccrs.PlateCarree()
    for lon, lat, label in NEIGHBOUR_LABELS:
        ax.text(
            lon, lat, label,
            transform=geo_crs, fontsize=8.5, ha="center", va="center",
            color="#555555", style="italic", fontweight="bold",
            alpha=0.6, zorder=5, clip_on=True,
        )

    return fig, ax


def _industry_scenario_name(snakemake) -> str:
    mapping = snakemake.config["industry"]["forecast_industry"].get(
        "scenario_mapping", {}
    )
    return mapping.get(str(snakemake.wildcards.run), str(snakemake.wildcards.run))


def _title_label(planning_horizon: str, industry_scenario: str, settings: dict) -> str:
    if settings.get("show_industry_scenario_label", False):
        return f"{industry_scenario} · {planning_horizon}"
    return planning_horizon


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_energy_demand_nrw",
            opts="",
            clusters="adm",
            sector_opts="",
            planning_horizons="2045",
            configfiles=["config/config.nrw.yaml"],
            run="endo-grid___CCS-Exp__offshore-co2",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    settings = snakemake.params.plotting["process_energy_demand_nrw"]
    figsize = ast.literal_eval(settings["figsize"])
    dpi = settings["dpi"]
    max_pie = settings["max_pie_size"]
    min_pie = settings["min_pie_size"]
    legend_sizes = settings["legend_sizes"]       # reference TWh values for legend
    carrier_colors = settings["carrier_colors"]
    carrier_names = settings["carrier_names"]

    planning_horizon = str(snakemake.wildcards.planning_horizons)
    industry_scenario = _industry_scenario_name(snakemake)

    proj = ccrs.Mercator()
    regions = gpd.read_file(snakemake.input.regions).set_index("name")

    nuts3 = gpd.read_file(snakemake.input.nuts3_shapes)
    dea_nuts3 = nuts3[nuts3["index"].str.startswith("DEA")].copy()
    dea_nuts3_proj = dea_nuts3.to_crs(proj.proj4_init)

    # --- data ---
    df = pd.read_csv(snakemake.input.energy_demand)
    dea = df[df["Region"].str.startswith("DEA")].copy()

    # Sum all applications and subsectors: Region × energy carrier → TWh
    pivot = (
        dea.groupby(["Region", "Energy_carrier"])[planning_horizon]
        .sum()
        .unstack("Energy_carrier")
        .fillna(0)
    )
    # Drop carriers with no activity across all DEA regions
    pivot = pivot.loc[:, pivot.sum() > 0]

    region_totals = pivot.sum(axis=1)
    max_total = region_totals.max()

    # --- basemap ---
    fig, ax = _draw_basemap(regions, dea_nuts3_proj, proj, figsize)

    # Force axes transforms to be initialised before coordinate lookups
    fig.canvas.draw()

    # --- pie charts ---
    for _, nuts3_row in dea_nuts3_proj.iterrows():
        region = nuts3_row["index"]
        if region not in pivot.index:
            continue
        total = region_totals.get(region, 0)
        if total <= 0:
            continue

        cx, cy = nuts3_row.geometry.centroid.x, nuts3_row.geometry.centroid.y

        # Data → display → axes-fraction coordinates
        disp = ax.transData.transform((cx, cy))
        ax_x, ax_y = ax.transAxes.inverted().transform(disp)

        # Skip centroids that fall outside the clipped axes extent
        if not (0.01 < ax_x < 0.99 and 0.01 < ax_y < 0.99):
            continue

        # Radius scaled by sqrt(total / max_total)
        size = max(min_pie, max_pie * np.sqrt(total / max_total))

        ax_pie = ax.inset_axes(
            [ax_x - size / 2, ax_y - size / 2, size, size],
            zorder=6,
        )
        ax_pie.set_axis_off()
        ax_pie.patch.set_visible(False)

        slice_vals = pivot.loc[region]
        slice_vals = slice_vals[slice_vals > 0]
        colors = [carrier_colors.get(s, "#cccccc") for s in slice_vals.index]

        ax_pie.pie(
            slice_vals.values,
            colors=colors,
            startangle=90,
            wedgeprops={"linewidth": 0.3, "edgecolor": "white"},
        )

    # --- industry scenario and year annotation ---
    ax.text(
        0.02, 0.97, _title_label(planning_horizon, industry_scenario, settings),
        transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="left",
        color="#333333", zorder=9,
    )

    # --- total NRW annotation ---
    nrw_total = region_totals.sum()
    ax.text(
        0.02, 0.90, f"NRW gesamt: {nrw_total:.0f} TWh",
        transform=ax.transAxes,
        fontsize=7, va="top", ha="left",
        color="#555555", zorder=9,
    )

    # --- size reference legend (bottom-left) ---
    legend_kw = dict(frameon=False, alignment="left",
                     title_fontproperties={"weight": "bold"})
    size_handles = []
    for ref_twh in legend_sizes:
        ref_size = max(min_pie, max_pie * np.sqrt(ref_twh / max_total))
        # Represent as a grey circle scaled to the same axes-fraction radius
        # Use a scatter dot with area proportional to (ref_size in pts)²
        pts = ref_size * fig.get_size_inches()[0] * fig.dpi  # axes-fraction → pts
        size_handles.append(
            plt.scatter([], [], s=pts ** 2 * 0.0025, c="#888888",
                        alpha=0.8, label=f"{ref_twh} TWh")
        )
    size_leg = ax.legend(
        handles=size_handles,
        title="Energiebedarf",
        loc="lower left",
        bbox_to_anchor=(0.01, 0.02),
        ncol=1,
        fontsize=7,
        labelspacing=1.5,
        borderpad=0.8,
        **legend_kw,
    )
    ax.add_artist(size_leg)

    # --- carrier colour legend (bottom-right) ---
    active_carriers = [s for s in carrier_colors if s in pivot.columns and pivot[s].sum() > 0]
    color_handles = [
        mpatches.Patch(
            facecolor=carrier_colors[s],
            label=carrier_names.get(s, s),
        )
        for s in active_carriers
    ]
    ax.legend(
        handles=color_handles,
        title="Energieträger",
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        ncol=2,
        fontsize=6,
        **legend_kw,
    )

    fig.savefig(snakemake.output.map, bbox_inches="tight")
    fig.savefig(snakemake.output.png, bbox_inches="tight", dpi=dpi)
    plt.show()
    plt.close(fig)
