# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Creates a CO₂ flow map for the NRW region with directional arrows showing
net annual CO₂ transport on each pipeline corridor.

Combines the NRW map style from plot_carbon_dioxide_network_nrw with the
flow-arrow approach from plot_balance_map. Useful for verifying that no
artificial circulation loops appear after the CO₂ pipeline efficiency fix.
"""

import re
import textwrap

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from matplotlib.collections import LineCollection
from packaging.version import Version, parse
from pypsa.plot import add_legend_lines, add_legend_patches, add_legend_semicircles
from pypsa.statistics import get_transmission_carriers
from shapely.geometry import MultiPolygon, Polygon
from shapely import wkt

from scripts._helpers import configure_logging, retry, set_scenario_config
from scripts.make_summary import assign_locations

SEMICIRCLE_CORRECTION_FACTOR = 2 if parse(pypsa.__version__) <= Version("0.33.2") else 1

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


def _draw_co2_project_links(ax, plot_links, link_width, link_colors_map, co2_projects_geom):
    """Draw co2_projects_geom segments styled with the solved network's link widths and colors."""
    pc = ccrs.PlateCarree()
    segments, widths, colors = [], [], []
    for _, row in co2_projects_geom.to_crs("EPSG:4326").iterrows():
        link_id = row.get("id")
        geom = row.geometry
        if geom is None or geom.is_empty or link_id not in plot_links.index:
            continue
        lw = link_width.get(link_id, 0.0)
        color = link_colors_map.get(plot_links.loc[link_id, "carrier"], "#888888")
        parts = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        for part in parts:
            segments.append([proj.transform_point(x, y, pc) for x, y in part.coords])
            widths.append(lw)
            colors.append(color)
    if segments:
        ax.add_collection(LineCollection(
            segments, linewidths=widths, colors=colors, capstyle="round", zorder=5,
        ))


@retry
def plot_co2_flow_map(
    n: pypsa.Network,
    co2_projects: gpd.GeoDataFrame | None = None,
    co2_projects_geom: gpd.GeoDataFrame | None = None,
    co2_projects_enable: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot net annual CO₂ pipeline flows with directional arrows for NRW."""
    plot_network = n.copy()
    assign_locations(plot_network)

    tech_colors = snakemake.params.plotting["tech_colors"]
    settings = snakemake.params.plotting["co2_flow_map_nrw"]

    bus_size_factor = settings["bus_factor"]
    unit_conversion = settings["unit_conversion"]
    flow_linewidth_factor = settings["flow_linewidth_factor"]
    flow_factor = settings.get("flow_factor", 1.0)

    # --- bus sizes from energy balance (same as capacity map) ---
    bus_carrier = "co2 stored"
    transmission_carriers = get_transmission_carriers(
        plot_network, bus_carrier=bus_carrier
    ).rename({"name": "carrier"})

    eb = plot_network.statistics.energy_balance(
        bus_carrier=bus_carrier, groupby=["bus", "carrier"]
    )

    components = transmission_carriers.unique("component")
    carriers = transmission_carriers.unique("carrier")
    carriers_in_eb = carriers[carriers.isin(eb.index.get_level_values("carrier"))]
    eb.loc[components] = eb.loc[components].drop(index=carriers_in_eb, level="carrier")
    eb = eb.dropna()
    bus_size = eb.groupby(level=["bus", "carrier"]).sum().div(unit_conversion)
    bus_size = bus_size.sort_values(ascending=False)

    n.carriers.update({"color": tech_colors})
    carrier_colors = n.carriers.color.copy().replace("", "grey")

    colors = (
        bus_size.index.get_level_values("carrier")
        .unique()
        .to_series()
        .map(carrier_colors)
    )

    co2_bus_carriers = ["co2 stored", "co2 sequestered"]
    plot_buses = plot_network.buses.loc[
        plot_network.buses.carrier.isin(co2_bus_carriers)
    ].copy()

    link_colors = {
        "CO2 pipeline": tech_colors["CO2 pipeline"],
        "CO2 pipeline short": tech_colors["CO2 pipeline short"],
    }

    # Exclude reversed links — only forward links define corridor direction
    is_reversed = plot_network.links.get(
        "reversed", pd.Series(False, index=plot_network.links.index)
    ).fillna(False)
    plot_links = plot_network.links.loc[
        plot_network.links.carrier.isin(link_colors) & ~is_reversed
    ].copy()

    # --- net annual flows ---
    # statistics.transmission returns total annual flow (MW·h = t_CO₂ for CO₂ pipelines)
    flow = plot_network.statistics.transmission(
        groupby=False, bus_carrier=bus_carrier
    ).div(unit_conversion)

    link_flow_series = pd.Series(dtype=float)
    if not flow.empty:
        raw = flow.get("Link", pd.Series(dtype=float))
        if not raw.empty:
            # Net out reversed-link flows: forward - reverse = signed net on corridor
            rev_mask = raw.index.str.contains("-reversed")
            flow_rev = raw[rev_mask].rename(lambda x: x.replace("-reversed", ""))
            link_flow_series = raw[~rev_mask].subtract(flow_rev, fill_value=0)

    # Assign flow to each forward link; aggregate parallel links sharing the same bus pair
    plot_links["_flow"] = link_flow_series.reindex(plot_links.index).fillna(0)
    # Per-link widths before dedup — used for co2_projects_geom geometry drawing
    plot_links_all = plot_links.copy()
    individual_link_widths = plot_links_all["_flow"].abs().mul(flow_linewidth_factor).clip(lower=0)
    summed_cap = plot_links.groupby(["bus0", "bus1"])["p_nom_opt"].sum()
    summed_flow = plot_links.groupby(["bus0", "bus1"])["_flow"].sum()
    plot_links = plot_links.drop_duplicates(subset=["bus0", "bus1"]).copy()
    plot_links["p_nom_opt"] = [
        summed_cap.at[b0, b1] for b0, b1 in zip(plot_links.bus0, plot_links.bus1)
    ]
    plot_links["_flow"] = [
        summed_flow.at[b0, b1] for b0, b1 in zip(plot_links.bus0, plot_links.bus1)
    ]

    link_flow = plot_links["_flow"]
    # Line thickness proportional to absolute net flow; minimum width so zero-flow
    # links remain visible as thin grey lines
    link_width = link_flow.abs().mul(flow_linewidth_factor).clip(lower=0)

    plot_network.buses = plot_buses
    plot_network.links = plot_links

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": proj})

    # Grey background for regions outside NRW/DEA
    non_dea = regions[~regions.index.str.startswith("DEA")].to_crs(proj.proj4_init)
    non_dea.plot(
        ax=ax, facecolor="#f2f2f2", edgecolor="#999999", linewidth=0.3, zorder=1
    )
    non_dea[non_dea.index.str.len() <= 2].boundary.plot(
        ax=ax, edgecolor="#999999", linewidth=1.0, zorder=2
    )
    de_non_dea = non_dea[
        non_dea.index.str.startswith("DE") & (non_dea.index.str.len() > 3)
    ].copy()
    if not de_non_dea.empty:
        de_non_dea["nuts1"] = de_non_dea.index.str[:3]
        de_non_dea.dissolve(by="nuts1").boundary.plot(
            ax=ax, edgecolor="#999999", linewidth=1.0, zorder=2
        )

    ax.add_feature(  # type: ignore[attr-defined]
        cfeature.NaturalEarthFeature(
            "physical", "rivers_lake_centerlines", "10m", facecolor="none"
        ),
        edgecolor="#6baed6", linewidth=0.6, zorder=3,
    )
    ax.add_feature(  # type: ignore[attr-defined]
        cfeature.NaturalEarthFeature("physical", "lakes", "10m"),
        facecolor="#c6dbef", edgecolor="#6baed6", linewidth=0.3, zorder=3,
    )

    if co2_projects_enable:
        # Suppress pypsa's straight bus-to-bus lines; draw real geometries below
        plot_network.plot(
            geomap=True,
            geomap_color=False,
            bus_size=bus_size * bus_size_factor,
            bus_color=colors,
            bus_alpha=0.9,
            bus_split_circle=True,
            branch_components=[],
            ax=ax,
            **map_opts,
        )
    else:
        plot_network.plot(
            geomap=True,
            geomap_color=False,
            bus_size=bus_size * bus_size_factor,
            bus_color=colors,
            bus_alpha=0.9,
            bus_split_circle=True,
            link_color=plot_links.carrier.map(link_colors),
            link_width=link_width,
            link_flow=link_flow * flow_factor,
            branch_components=["Link"],
            ax=ax,
            **map_opts,
        )

    if co2_projects_enable and co2_projects_geom is not None:
        _draw_co2_project_links(ax, plot_links_all, individual_link_widths, link_colors, co2_projects_geom)

    ax_collections = ax.collections
    for col in ax_collections:
        col.set_capstyle("round")

    # --- legends ---
    legend_kw = dict(
        loc="upper left",
        frameon=False,
        alignment="left",
        title_fontproperties={"weight": "bold"},
    )

    pad = 0.02
    n.carriers.loc["", "color"] = "None"

    pos_carriers = bus_size[bus_size > 0].index.unique("carrier")
    neg_carriers = bus_size[bus_size < 0].index.unique("carrier")
    common_carriers = pos_carriers.intersection(neg_carriers)

    def get_total_abs(carrier, sign):
        values = bus_size.loc[:, carrier]
        return values[values * sign > 0].abs().sum()

    supp_carriers = sorted(
        set(pos_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) >= get_total_abs(c, -1)}
    )
    cons_carriers = sorted(
        set(neg_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) < get_total_abs(c, -1)}
    )

    carrier_german = snakemake.params.plotting.get("carrier_german", {})
    add_legend_patches(
        ax,
        n.carriers.color[supp_carriers],
        [carrier_german.get(c, c) for c in supp_carriers],
        legend_kw={
            "bbox_to_anchor": (0, -pad),
            "ncol": 1,
            "title": "Erzeugung (CO$_2$-Abscheidung)",
            **legend_kw,
        },
    )

    add_legend_patches(
        ax,
        n.carriers.color[cons_carriers],
        [carrier_german.get(c, c) for c in cons_carriers],
        legend_kw={
            "bbox_to_anchor": (0.5, -pad),
            "ncol": 1,
            "title": "Nutzung",
            **legend_kw,
        },
    )

    legend_bus_size = settings["bus_sizes"]
    carrier_unit = settings["unit"]
    flow_unit = settings["flow_unit"]
    legend_flow_sizes = settings["flow_sizes"]

    br_kw = {**legend_kw, "loc": "lower right", "bbox_to_anchor": (0.99, 0.02)}
    if legend_bus_size is not None:
        add_legend_semicircles(
            ax,
            [
                s * bus_size_factor * SEMICIRCLE_CORRECTION_FACTOR
                for s in legend_bus_size
            ],
            [f"{s} {carrier_unit}" for s in legend_bus_size],
            patch_kw={"color": "#666"},
            legend_kw=br_kw,
        )

    if legend_flow_sizes is not None:
        add_legend_lines(
            ax,
            [s * flow_linewidth_factor for s in legend_flow_sizes],
            [f"{s} {flow_unit}" for s in legend_flow_sizes],
            patch_kw=dict(color="#666", solid_capstyle="round"),
            legend_kw={**br_kw, "bbox_to_anchor": (0.78, 0.02)},
        )

    ax.set_facecolor("white")

    return fig, ax


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_co2_flow_map_nrw",
            opts="",
            clusters="adm",
            sector_opts="",
            planning_horizons="2040",
            configfiles=["config/config.nrw.yaml"],
            run="oge-grid___Ref___offshore-co2",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)
    regions = gpd.read_file(snakemake.input.regions).set_index("name")
    co2_projects = pd.read_csv(snakemake.input.co2_projects, index_col=0)
    co2_projects["geometry"] = co2_projects["geometry"].apply(wkt.loads)
    co2_projects = gpd.GeoDataFrame(co2_projects, geometry="geometry", crs="EPSG:3035").to_crs("EPSG:4326")

    # Original project geometry    
    co2_projects_geom = gpd.read_file(snakemake.input.co2_projects_geom, crs="EPSG:4326")
    co2_projects_enable = snakemake.params.co2_projects_enable

    map_opts = snakemake.params.plotting["map"]
    map_opts["boundaries"] = NRW_BOUNDS
    map_opts.pop("geomap_colors", None)

    proj = ccrs.Mercator()
    fig, ax = plot_co2_flow_map(
        n,
        co2_projects=co2_projects if co2_projects_enable else None,
        co2_projects_geom=co2_projects_geom if co2_projects_enable else None,
        co2_projects_enable=co2_projects_enable,
    )

    # Overlay DEA (NRW) administrative boundaries
    dea_regions = regions[regions.index.str.startswith("DEA")].to_crs(proj.proj4_init)
    dea_outer = dea_regions.dissolve()

    dea_regions.boundary.plot(ax=ax, edgecolor="grey", linewidth=0.5, zorder=3)

    def _exterior_only(geom):
        if geom.geom_type == "Polygon":
            return Polygon(geom.exterior)
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])

    gpd.GeoDataFrame(
        geometry=dea_outer.geometry.apply(_exterior_only), crs=dea_outer.crs
    ).boundary.plot(ax=ax, edgecolor="black", linewidth=1.0, zorder=4)

    nuts3 = gpd.read_file(snakemake.input.nuts3_shapes)
    dea_nuts3 = nuts3[nuts3["index"].str.startswith("DEA")].copy()
    dea_nuts3 = dea_nuts3.to_crs(proj.proj4_init)
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

    geo_crs = ccrs.PlateCarree()
    for lon, lat, label in NEIGHBOUR_LABELS:
        ax.text(
            lon, lat, label,
            transform=geo_crs, fontsize=8.5, ha="center", va="center",
            color="#555555", style="italic", fontweight="bold",
            alpha=0.6, zorder=5, clip_on=True,
        )

    fig.savefig(snakemake.output.map, bbox_inches="tight")
    fig.savefig(snakemake.output.png, bbox_inches="tight", dpi=150)
    plt.close(fig)
