# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Creates a zoomed map of the optimised carbon dioxide network for NRW (DEA region).
Neighbouring regions remain visible but the view is centred on North Rhine-Westphalia.
"""

import re
import textwrap

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from packaging.version import Version, parse
from pypsa.plot import add_legend_lines, add_legend_patches, add_legend_semicircles
from pypsa.statistics import get_transmission_carriers
from shapely.geometry import MultiPolygon, Polygon

from scripts._helpers import configure_logging, retry, set_scenario_config
from scripts.make_summary import assign_locations

SEMICIRCLE_CORRECTION_FACTOR = 2 if parse(pypsa.__version__) <= Version("0.33.2") else 1

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

# Carrier name translations (English → German) for Supply/Consumption legends
CARRIER_NAMES_DE: dict[str, str] = {
    "co2 sequestered": "CO$_2$-Sequestrierung",
    "DAC": "Direct Air Capture",
    "gas for industry CC": "Gas für Industrie",
    "methanolisation": "Methanisierung",
    "process emissions CC": "Prozessemissionen",
    "SMR CC": "Dampfreformierung",
    "solid biomass for industry CC": "Feste Biomasse für Industrie",
    "urban central gas CHP CC": "Städtische Gas-KWK",
    "urban central solid biomass CHP CC": "Städtische Biomasse-KWK",

}

# City reference data — kept for optional future use
# NRW_CITIES = pd.DataFrame(
#     {
#         "name": [
#             "Münster", "Aachen", "Bonn", "Köln",
#             "Düsseldorf", "Wuppertal", "Gelsenkirchen", "Duisburg",
#             "Essen", "Bochum", "Bielefeld",
#         ],
#         "lon": [
#             7.6261, 6.0839, 7.0982, 6.9603,
#             6.7735, 7.1827, 7.0956, 6.7623,
#             7.0116, 7.2156, 8.5325,
#         ],
#         "lat": [
#             51.9607, 50.7753, 50.7374, 50.9333,
#             51.2217, 51.2562, 51.5177, 51.4344,
#             51.4556, 51.4818, 52.0302,
#         ],
#     }
# )


def _clean_nuts3_name(name: str) -> str:
    return re.sub(r",\s*Kreisfreie\s+Stadt$", "", name).strip()


def load_projection(plotting_params: dict) -> ccrs.CRS:
    """Instantiate the cartopy CRS defined in plotting_params['projection']."""
    proj_kwargs = dict(
        plotting_params.get("projection", {"name": "EqualEarth"})
    )  # shallow copy so pop doesn't mutate the config
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)


@retry
def plot_co2_map(n: pypsa.Network) -> tuple[plt.Figure, plt.Axes]:
    """Plot the optimised CO2 network zoomed into NRW (DEA region)."""
    plot_network = n.copy()
    assign_locations(plot_network)

    tech_colors = snakemake.params.plotting["tech_colors"]
    settings = snakemake.params.plotting["carbon_dioxide_network_nrw"]

    bus_size_factor = settings["bus_factor"]
    unit_conversion = settings["unit_conversion"]
    linewidth_factor = 2e3

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
    plot_links = plot_network.links.loc[
        plot_network.links.carrier.isin(link_colors)
    ].copy()

    # Sum p_nom_opt for parallel links (same bus0/bus1) so widths don't overlay
    summed = plot_links.groupby(["bus0", "bus1"])["p_nom_opt"].sum()
    plot_links = plot_links.drop_duplicates(subset=["bus0", "bus1"]).copy()
    plot_links["p_nom_opt"] = [
        summed.at[b0, b1] for b0, b1 in zip(plot_links.bus0, plot_links.bus1)
    ]

    plot_network.buses = plot_buses
    plot_network.links = plot_links

    link_width = plot_links.p_nom_opt.div(linewidth_factor)

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": proj})

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

    # Urban area footprints (Natural Earth 10 m).
    # ax.add_feature(  # type: ignore[attr-defined]
    #     cfeature.NaturalEarthFeature("cultural", "urban_areas", "10m"),
    #     facecolor="#e0d8d0", edgecolor="none", zorder=2,
    # )

    # Rivers and lakes (Natural Earth 10 m).
    # ax is a GeoAxes at runtime (subplot_kw={"projection": proj}); Pyright sees Axes.
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

    plot_network.plot(
        geomap=True,
        geomap_color=False,
        bus_size=bus_size * bus_size_factor,
        bus_color=colors,
        bus_alpha=0.9,
        bus_split_circle=True,
        link_color=plot_links.carrier.map(link_colors),
        link_width=link_width,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
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

    add_legend_patches(
        ax,
        n.carriers.color[supp_carriers],
        [CARRIER_NAMES_DE.get(c, c) for c in supp_carriers],
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
        [CARRIER_NAMES_DE.get(c, c) for c in cons_carriers],
        legend_kw={
            "bbox_to_anchor": (0.5, -pad),
            "ncol": 1,
            "title": "Nutzung",
            **legend_kw,
        },
    )

    legend_bus_size = settings["bus_sizes"]
    carrier_unit = settings["unit"]
    branch_unit = settings["branch_unit"]
    branch_unit_conversion = settings["branch_unit_conversion"]

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

    legend_branch_sizes = settings["branch_sizes"]
    if legend_branch_sizes is not None:
        add_legend_lines(
            ax,
            [s / linewidth_factor for s in legend_branch_sizes],
            [
                f"{s / branch_unit_conversion} {branch_unit}"
                for s in legend_branch_sizes
            ],
            patch_kw=dict(color="#666", solid_capstyle="round"),
            legend_kw={**br_kw, "bbox_to_anchor": (0.78, 0.02)},
        )

    ax.set_facecolor("white")

    return fig, ax


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_carbon_dioxide_network_nrw",
            opts="",
            clusters="adm",
            sector_opts="",
            planning_horizons="2045",
            configfiles=["config/config.nrw.yaml"],
            run="endo-grid___Ref___offshore-co2",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)

    regions = gpd.read_file(snakemake.input.regions).set_index("name")

    map_opts = snakemake.params.plotting["map"]
    map_opts["boundaries"] = NRW_BOUNDS
    map_opts.pop("geomap_colors", None)  # replaced by explicit geomap_color=False below

    proj = ccrs.Mercator()
    fig, ax = plot_co2_map(n)

    # Overlay DEA (NRW) administrative boundaries
    dea_regions = regions[regions.index.str.startswith("DEA")].to_crs(proj.proj4_init)
    dea_outer = dea_regions.dissolve()

    dea_regions.boundary.plot(ax=ax, edgecolor="grey", linewidth=0.5, zorder=3)

    # Strip interior holes (gaps between source polygons show up as black rings after
    # dissolve) by rebuilding each polygon from its exterior ring only.
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

        # Scale font by bounding-box width in projected metres.
        # Mercator x-coords at NRW latitudes: ~70 km per degree longitude.
        # Rural Kreis (~60 km wide) → ~7 pt; small Kreisfreie Stadt (~15 km) → ~4 pt.
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
