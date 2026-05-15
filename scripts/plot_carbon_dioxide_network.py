# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Creates a map of the optimised carbon dioxide network, storage and sequestration infrastructure.
"""

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pypsa
from packaging.version import Version, parse
from pypsa.plot import add_legend_lines, add_legend_patches, add_legend_semicircles
from pypsa.statistics import get_transmission_carriers

from scripts._helpers import configure_logging, retry, set_scenario_config
from scripts.make_summary import assign_locations

SEMICIRCLE_CORRECTION_FACTOR = 2 if parse(pypsa.__version__) <= Version("0.33.2") else 1


def load_projection(plotting_params: dict) -> ccrs.CRS:
    """Instantiate the cartopy CRS defined in plotting_params['projection']."""
    proj_kwargs = dict(
        plotting_params.get("projection", {"name": "EqualEarth"})
    )  # shallow copy so pop doesn't mutate the config
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)


@retry
def plot_co2_map(n: pypsa.Network) -> tuple[plt.Figure, plt.Axes]:
    """Plot the optimised CO2 network, storage, and sequestration infrastructure."""
    plot_network = n.copy()
    assign_locations(plot_network)

    tech_colors = snakemake.params.plotting["tech_colors"]
    # read plotting settings from dedicated carbon_dioxide_network config
    settings = snakemake.params.plotting["carbon_dioxide_network"]

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

    plot_network.plot(
        geomap=True,
        bus_size=bus_size * bus_size_factor,
        bus_color=colors,
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

    pad = 0.18
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
        supp_carriers,
        legend_kw={
            "bbox_to_anchor": (0, -pad),
            "ncol": 1,
            "title": "Supply",
            **legend_kw,
        },
    )

    add_legend_patches(
        ax,
        n.carriers.color[cons_carriers],
        cons_carriers,
        legend_kw={
            "bbox_to_anchor": (0.5, -pad),
            "ncol": 1,
            "title": "Consumption",
            **legend_kw,
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
                "bbox_to_anchor": (0, 1),
                **legend_kw,
            },
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
            patch_kw=dict(color="lightgrey", solid_capstyle="round"),
            legend_kw={"bbox_to_anchor": (0.25, 1), **legend_kw},
        )

    ax.set_facecolor("white")

    return fig, ax


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_carbon_dioxide_network",
            opts="",
            clusters="adm",
            sector_opts="",
            planning_horizons="2035",
            configfiles=["config/config.nrw.yaml"],
            run="greenfield-oge-extendable-only-offshore-storage",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)

    regions = gpd.read_file(snakemake.input.regions).set_index("name")

    map_opts = snakemake.params.plotting["map"]

    if map_opts["boundaries"] is None:
        map_opts["boundaries"] = regions.total_bounds[[0, 2, 1, 3]] + [-1, 1, -1, 1]

    proj = load_projection(snakemake.params.plotting)

    fig, ax = plot_co2_map(n)

    fig.savefig(snakemake.output.map, bbox_inches="tight")
    plt.close(fig)
