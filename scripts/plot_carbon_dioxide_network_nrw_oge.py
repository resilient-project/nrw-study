# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Creates a zoomed map of the optimised carbon dioxide network for NRW (DEA region)
using original OGE pipeline geometries instead of straight bus-to-bus lines.
"""

import re
import textwrap

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import pandas as pd
import pypsa
from packaging.version import Version, parse
from pypsa.plot import add_legend_lines, add_legend_patches, add_legend_semicircles
from pypsa.statistics import get_transmission_carriers
from shapely.geometry import MultiPolygon, Polygon

from scripts._helpers import configure_logging, retry, set_scenario_config
from scripts.make_summary import assign_locations

SEMICIRCLE_CORRECTION_FACTOR = 2 if parse(pypsa.__version__) <= Version("0.33.2") else 1

FIGURE_SIZE = (4.5, 7)  # (width, height) in inches

# Bounding box for NRW (DEA): [lon_min, lon_max, lat_min, lat_max]
NRW_BOUNDS = [5.75, 9.55, 50.2, 52.65]

# Neighbouring region labels: (lon, lat, name)
NEIGHBOUR_LABELS = [
    (6.15, 52.0, "NIEDERLANDE"),
    (6.03, 50.34, "BELGIEN"),
    (7.07, 50.26, "RHEINLAND-PFALZ"),
    (8.9, 51.05, "HESSEN"),
    (7.75, 52.55, "NIEDERSACHSEN"),
]



def _clean_nuts3_name(name: str) -> str:
    return re.sub(r",\s*Kreisfreie\s+Stadt$", "", name).strip()


def load_projection(plotting_params: dict) -> ccrs.CRS:
    """Instantiate the cartopy CRS defined in plotting_params['projection']."""
    proj_kwargs = dict(
        plotting_params.get("projection", {"name": "EqualEarth"})
    )
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)


def _draw_co2_project_links(ax, plot_links_all, link_width, link_colors_map, dark_short, co2_projects_geom):
    """Draw co2_projects_geom segments styled with the solved network's capacity widths and colors."""
    pc = ccrs.PlateCarree()
    segments, widths, colors = [], [], []
    for _, row in co2_projects_geom.to_crs("EPSG:4326").iterrows():
        link_id = row.get("id")
        geom = row.geometry
        if geom is None or geom.is_empty or link_id not in plot_links_all.index:
            continue
        lw = link_width.get(link_id, 0.0)
        carrier = plot_links_all.loc[link_id, "carrier"]
        color = link_colors_map.get(carrier, "#888888") if str(link_id).startswith("CO2 pipeline") else dark_short
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
def plot_co2_map(
    n: pypsa.Network,
    co2_projects_geom: gpd.GeoDataFrame,
    bus_size_factor: float,
    linewidth_factor: float,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the optimised CO2 network zoomed into NRW (DEA region) using OGE geometries."""
    plot_network = n.copy()
    assign_locations(plot_network)

    tech_colors = snakemake.params.plotting["tech_colors"]
    settings = snakemake.params.plotting["carbon_dioxide_network_nrw_oge"]

    unit_conversion = settings["unit_conversion"]

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

    if settings.get("aggregate_industry_emissions", False):
        _agg_label = settings.get("aggregate_label", "Emissionen Industrie")
        _agg_color = settings.get("aggregate_color", tech_colors.get("process emissions CC", "#000000"))
        _industry_cc = {"gas for industry CC", "process emissions CC"}
        _rename = {c: _agg_label for c in _industry_cc}
        _new_carriers = bus_size.index.get_level_values("carrier").map(
            lambda c: _rename.get(c, c)
        )
        bus_size.index = pd.MultiIndex.from_arrays(
            [bus_size.index.get_level_values("bus"), _new_carriers],
            names=["bus", "carrier"],
        )
        bus_size = bus_size.groupby(level=["bus", "carrier"]).sum()
        carrier_colors[_agg_label] = _agg_color
        n.carriers.loc[_agg_label, "color"] = _agg_color

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
    dark_short = "#2c4f7a"  # dark matte blue for Überlappung Planprojekte
    is_reversed = plot_network.links.get(
        "reversed", pd.Series(False, index=plot_network.links.index)
    ).fillna(False)
    plot_links = plot_network.links.loc[
        plot_network.links.carrier.isin(link_colors) & ~is_reversed
    ].copy()

    # Save per-link widths before deduplication — needed for geometry-based drawing
    plot_links_all = plot_links.copy()
    individual_link_widths = plot_links_all["p_nom_opt"].mul(linewidth_factor)

    # Sum p_nom_opt for parallel links (same bus0/bus1) so widths don't overlay
    summed = plot_links.groupby(["bus0", "bus1"])["p_nom_opt"].sum()
    plot_links = plot_links.drop_duplicates(subset=["bus0", "bus1"]).copy()
    plot_links["p_nom_opt"] = [
        summed.at[b0, b1] for b0, b1 in zip(plot_links.bus0, plot_links.bus1)
    ]

    plot_network.buses = plot_buses
    plot_network.links = plot_links

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, subplot_kw={"projection": proj})

    # Grey background for regions outside NRW/DEA
    non_dea = regions[~regions.index.str.startswith("DEA")].to_crs(proj.proj4_init)
    non_dea.plot(
        ax=ax, facecolor="#f2f2f2", edgecolor="#999999", linewidth=0.3, zorder=1
    )
    # Foreign country borders (2-char codes) — already at country level
    non_dea[non_dea.index.str.len() <= 2].boundary.plot(
        ax=ax, edgecolor="#999999", linewidth=1.0, zorder=2
    )
    # German federal state borders: dissolve non-NRW NUTS3 polygons to NUTS1 level
    de_non_dea = non_dea[
        non_dea.index.str.startswith("DE") & (non_dea.index.str.len() > 3)
    ].copy()
    if not de_non_dea.empty:
        de_non_dea["nuts1"] = de_non_dea.index.str[:3]
        de_non_dea.dissolve(by="nuts1").boundary.plot(
            ax=ax, edgecolor="#999999", linewidth=1.0, zorder=2
        )

    # Rivers and lakes (Natural Earth 10 m).
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

    # Draw buses only; OGE geometries replace pypsa's straight bus-to-bus lines
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

    _draw_co2_project_links(
        ax, plot_links_all, individual_link_widths, link_colors, dark_short, co2_projects_geom
    )

    ax_collections = ax.collections
    for col in ax_collections:
        col.set_capstyle("round")

    # --- legends ---
    legend_kw = dict(
        loc="upper left",
        frameon=False,
        alignment="left",
        title_fontproperties={"weight": "bold"},
        handlelength=1.1,
    )

    pad = 0.05
    n.carriers.loc["", "color"] = "None"

    pos_carriers = bus_size[bus_size > 0].index.unique("carrier")
    neg_carriers = bus_size[bus_size < 0].index.unique("carrier")
    common_carriers = pos_carriers.intersection(neg_carriers)

    def get_total_abs(carrier, sign):
        values = bus_size.loc[:, carrier]
        return values[values * sign > 0].abs().sum()

    _supp_set = (
        set(pos_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) >= get_total_abs(c, -1)}
    )
    _legend_order = settings.get("legend_order", [])
    supp_carriers = [c for c in _legend_order if c in _supp_set] + sorted(
        _supp_set - set(_legend_order)
    )
    _cons_set = (
        set(neg_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) < get_total_abs(c, -1)}
    )
    _legend_order_nutzung = settings.get("legend_order_nutzung", [])
    cons_carriers = [c for c in _legend_order_nutzung if c in _cons_set] + sorted(
        _cons_set - set(_legend_order_nutzung)
    )

    carrier_german = snakemake.params.plotting.get("carrier_german", {})
    add_legend_patches(
        ax,
        n.carriers.color[supp_carriers],
        [carrier_german.get(c, c) for c in supp_carriers],
        legend_kw={
            "bbox_to_anchor": (0, -pad),
            "ncol": 1,
            "title": "CO$_2$-Abscheidung",
            **legend_kw,
            "handlelength": 0.8,
            "handleheight": 0.8,
        },
    )

    nutzung_leg = add_legend_patches(
        ax,
        n.carriers.color[cons_carriers],
        [carrier_german.get(c, c) for c in cons_carriers],
        legend_kw={
            "bbox_to_anchor": (0.5, -pad),
            "ncol": 1,
            "title": "Nutzung",
            **legend_kw,
            "handlelength": 0.8,
            "handleheight": 0.8,
        },
    )

    legend_bus_size = settings["bus_sizes"]
    carrier_unit = settings["unit"]
    branch_unit = settings["branch_unit"]
    branch_unit_conversion = settings["branch_unit_conversion"]

    if legend_bus_size is not None:
        add_legend_semicircles(
            ax,
            [
                s * bus_size_factor * SEMICIRCLE_CORRECTION_FACTOR
                for s in legend_bus_size
            ],
            [f"{s} {carrier_unit}" for s in legend_bus_size],
            patch_kw={"color": "#666"},
            legend_kw={
                "bbox_to_anchor": (0.56, -0.015),
                **legend_kw,
                "loc": "lower center",
            },
        )

    legend_branch_sizes = settings["branch_sizes"]
    if legend_branch_sizes is not None:
        add_legend_lines(
            ax,
            [s * linewidth_factor for s in legend_branch_sizes],
            [
                f"{s / branch_unit_conversion:.0f} {branch_unit}"
                for s in legend_branch_sizes
            ],
            patch_kw=dict(color=tech_colors["CO2 pipeline"], solid_capstyle="round"),
            legend_kw={"bbox_to_anchor": (0.96, -0.015), **legend_kw, "loc": "lower right"},
        )

    ax.set_facecolor("white")

    # Überlappung Planprojekte — below Nutzung (bottom right)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    nutzung_bottom_ax = ax.transAxes.inverted().transform(
        nutzung_leg.get_window_extent(renderer).min
    )[1]
    ueberlappung_leg = ax.legend(
        handles=[Line2D([0], [0], color=dark_short, linewidth=2, solid_capstyle="round")],
        labels=["PCI-PMI Projekte"],
        bbox_to_anchor=(0.5, nutzung_bottom_ax - 0.005),
        **legend_kw,
    )
    ax.add_artist(ueberlappung_leg)

    return fig, ax


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_carbon_dioxide_network_nrw_oge",
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
    co2_projects_geom = gpd.read_file(snakemake.input.co2_projects_geom, crs="EPSG:4326")

    map_opts = snakemake.params.plotting["map"]
    map_opts["boundaries"] = NRW_BOUNDS
    map_opts.pop("geomap_colors", None)

    settings = snakemake.params.plotting["carbon_dioxide_network_nrw_oge"]

    proj = ccrs.Mercator()
    fig, ax = plot_co2_map(
        n,
        co2_projects_geom=co2_projects_geom,
        bus_size_factor=settings["bus_factor"],
        linewidth_factor=settings["branch_factor"],
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

    # NUTS3 region name labels inside each DEA subregion
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

    # Neighbouring region/country labels
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
