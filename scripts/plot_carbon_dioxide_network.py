# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Creates a map of the optimised carbon dioxide network, storage and sequestration infrastructure.
"""

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from matplotlib.gridspec import GridSpec
from packaging.version import Version, parse
from pypsa.plot import add_legend_lines, add_legend_patches, add_legend_semicircles
from pypsa.statistics import get_transmission_carriers

from scripts._helpers import configure_logging, retry, set_scenario_config
from scripts.make_summary import assign_locations

SEMICIRCLE_CORRECTION_FACTOR = 2 if parse(pypsa.__version__) <= Version("0.33.2") else 1

# Carrier name translations (English → German) for Supply/Consumption legends
CARRIER_NAMES_DE: dict[str, str] = {
    "co2 sequestered": "CO$_2$-Sequestrierung",
    "DAC": "Direct Air Capture",
    "gas for industry CC": "Gas für Industrie",
    "methanolisation": "Methanisierung",
    "process emissions CC": "Prozessemissionen",
    "SMR CC": "Dampfreformierung",
    "solid biomass for industry CC": "Biomasse für Industrie",
    "urban central gas CHP CC": "Gas-KWK",
    "urban central solid biomass CHP CC": "Biomasse-KWK",
}


def load_projection(plotting_params: dict) -> ccrs.CRS:
    """Instantiate the cartopy CRS defined in plotting_params['projection']."""
    proj_kwargs = dict(
        plotting_params.get("projection", {"name": "EqualEarth"})
    )  # shallow copy so pop doesn't mutate the config
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)


@retry
def plot_co2_map(n: pypsa.Network, ax=None) -> tuple[plt.Figure, plt.Axes]:
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": proj})
    else:
        fig = ax.get_figure()

    # Draw modelled regions as background instead of PyPSA's global cartopy map
    regions_proj = regions.to_crs(proj.proj4_init)
    regions_proj.plot(
        ax=ax, facecolor="#f5f5f5", edgecolor="#aaaaaa", linewidth=0.4, zorder=1
    )

    plot_network.plot(
        geomap=True,
        geomap_color=False,
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

    pad = 0.05
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
            "title": "CO$_2$-Abscheidung",
            **legend_kw,
        },
    )

    add_legend_patches(
        ax,
        n.carriers.color[cons_carriers],
        [CARRIER_NAMES_DE.get(c, c) for c in cons_carriers],
        legend_kw={
            "bbox_to_anchor": (0.7, -pad),
            "ncol": 1,
            "title": "Nutzung",
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
            legend_kw={"bbox_to_anchor": (0.45, 1), **legend_kw},
        )

    ax.set_facecolor("white")

    # Pipeline-length bar legend (shown when include_lengths is enabled)
    if settings.get("include_lengths", False):
        length_colors = settings.get("lengths_bar_colors", {})
        add_legend_patches(
            ax,
            list(length_colors.values()),
            ["Onshore ⌀70cm", "Onshore ⌀40cm", "Offshore ⌀70cm"],
            legend_kw={
                "bbox_to_anchor": (0.7, -0.4),
                "ncol": 1,
                "title": "Durchmesser",
                **legend_kw,
            },
        )

    return fig, ax


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_carbon_dioxide_network",
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
    map_opts.pop("geomap_colors", None)  # replaced by explicit geomap_color=False

    # [lon_min, lon_max, lat_min, lat_max]
    # West: Dublin (~-6.3°), East: eastern Poland (~24.0°),
    # South: Corsica (~41.5°), North: Tromsø (~70.0°)
    map_opts["boundaries"] = [-9.5, 24.0, 41.5, 70.0]

    proj = load_projection(snakemake.params.plotting)

    settings = snakemake.params.plotting["carbon_dioxide_network"]
    include_lengths = settings.get("include_lengths", False)
    # Use declared input if present, otherwise derive from the network path
    lengths_path = getattr(snakemake.input, "lengths", None) or (
        snakemake.input.network
        .replace("/networks/base_s_", "/nrw-study/co2_pipeline_length_base_s_")
        .replace(".nc", ".csv")
    )

    fig, map_ax = plot_co2_map(n)

    if include_lengths:
        # Inset stacked bar on the lower-right corner of the map
        # [x0, y0, width, height] in axes-fraction coordinates
        bar_ax = map_ax.inset_axes([0.80, 0.02, 0.18, 0.41])

        lengths_df = pd.read_csv(lengths_path)

        # Onshore: pivot NRW and DE by pipe diameter
        onshore_pivot = (
            lengths_df[
                (lengths_df["terrain"] == "onshore")
                & lengths_df["region"].isin(["DE", "DEA"])
            ]
            .pivot_table(index="region", columns="carrier", values="length_km", fill_value=0)
            .rename(columns={"CO2 pipeline": "⌀70cm", "CO2 pipeline short": "⌀40cm"})
            .rename(index={"DEA": "NRW"})
        )
        # Offshore: sum DE + DEA, CO2 pipeline only — stacked on top of DE
        offshore_km = lengths_df[
            (lengths_df["terrain"] == "offshore")
            & lengths_df["region"].isin(["DE", "DEA"])
            & (lengths_df["carrier"] == "CO2 pipeline")
        ]["length_km"].sum()

        onshore_pivot["Offshore"] = 0.0
        onshore_pivot.loc["DE", "Offshore"] = offshore_km

        bar_data = onshore_pivot.reindex(["NRW", "DE"])
        bar_colors = list(settings.get("lengths_bar_colors", {}).values())
        ylim_max = settings.get("lengths_bar_ylim", 6000)

        bar_data.plot(
            kind="bar", stacked=True, ax=bar_ax,
            color=bar_colors, legend=False, width=0.6,
        )
        totals = bar_data.sum(axis=1)
        for i, total in enumerate(totals):
            bar_ax.text(i, total, f"{total:.0f}\nkm", ha="center", va="bottom", fontsize=8)
        bar_ax.set_ylim(0, ylim_max)
        bar_ax.set_xlabel("")
        bar_ax.set_ylabel("")
        bar_ax.tick_params(labelsize=6)
        bar_ax.set_xticklabels(bar_data.index, rotation=0, fontsize=8)
        bar_ax.set_yticks([])
        bar_ax.grid(False)
        bar_ax.patch.set_alpha(0.3)
        for spine in bar_ax.spines.values():
            spine.set_visible(False)

    fig.savefig(snakemake.output.map, bbox_inches="tight")
    fig.savefig(snakemake.output.png, bbox_inches="tight", dpi=150)
    plt.close(fig)
