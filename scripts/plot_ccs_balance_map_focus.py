#!/usr/bin/env python3

# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""Plot NRW- and Germany-focused static CO2 stored balance maps."""

import logging
from itertools import product

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import yaml
from packaging.version import Version, parse
from pypsa.plot import add_legend_lines, add_legend_patches, add_legend_semicircles
from pypsa.statistics import get_transmission_carriers

from scripts._helpers import PYPSA_V1, configure_logging, set_scenario_config
from scripts.add_electricity import sanitize_carriers
from scripts.plot_power_network import load_projection

logger = logging.getLogger(__name__)

SEMICIRCLE_CORRECTION_FACTOR = 2 if parse(pypsa.__version__) <= Version("0.33.2") else 1
CO2_STORED = "co2 stored"
FOCUS_BRANCH_FACTOR_SCALE = 0.35
FOCUS_BRANCH_WIDTH_MAX = 0.5
NETWORK_ONLY_SUFFIX = "_network_only"

FOCUS_SCOPES = {
    "nrw": {
        "label": "NRW",
        "prefixes": ("DEA",),
        "boundaries": [5.5, 9.8, 50.1, 52.8],
        "figsize": (5.0, 5.2),
    },
    "de": {
        "label": "Germany",
        "prefixes": ("DE",),
        "boundaries": [5.0, 15.6, 47.0, 55.4],
        "figsize": (5.0, 6.2),
    },
}


def mock_wildcard_sets_from_config(configfile):
    """Return all configured wildcard combinations for local mock execution."""
    with open(configfile) as f:
        config = yaml.safe_load(f)

    scenario = config["scenario"]
    runs = config["run"]["name"] if config["run"].get("scenarios", {}).get("enable") else [None]

    wildcard_sets = []
    for run, clusters, opts, sector_opts, planning_horizons in product(
        runs,
        scenario["clusters"],
        scenario["opts"],
        scenario["sector_opts"],
        scenario["planning_horizons"],
    ):
        wildcards = {
            "clusters": clusters,
            "opts": opts,
            "sector_opts": sector_opts,
            "planning_horizons": str(planning_horizons),
        }
        if run is not None:
            wildcards["run"] = run
        wildcard_sets.append(wildcards)

    return wildcard_sets


def focus_scope_and_balance_mode(output_name):
    """Return focus scope and whether to draw supply/consumption pies."""
    if output_name.endswith(NETWORK_ONLY_SUFFIX):
        return output_name[: -len(NETWORK_ONLY_SUFFIX)], False
    return output_name, True


def prepare_network(network_path, config):
    """Load the network and apply plotting carrier sanitization."""
    n = pypsa.Network(network_path)
    sanitize_carriers(n, config)
    pypsa.set_option("params.statistics.round", 8)
    pypsa.set_option("params.statistics.drop_zero", True)
    pypsa.set_option("params.statistics.nice_names", False)
    n.carriers["color"] = n.carriers.color.mask(
        n.carriers.color.isna() | n.carriers.color.eq(""), "lightgrey"
    )
    return n


def prepare_bus_locations(n, plotting_config):
    """Use location-level coordinates while keeping offshore fallbacks."""
    eu_location = plotting_config["eu_node_location"]
    n.buses.loc["EU", ["x", "y"]] = eu_location["x"], eu_location["y"]
    n.buses["location"] = n.buses["location"].replace("", "EU").fillna("EU")
    n.buses["x"] = n.buses.location.map(n.buses.x).fillna(n.buses["x"])
    n.buses["y"] = n.buses.location.map(n.buses.y).fillna(n.buses["y"])


def scope_buses(n, prefixes):
    """Return network buses whose names start with one of the scope prefixes."""
    return n.buses[n.buses.index.to_series().str.startswith(prefixes)].index


def scope_regions(regions, prefixes):
    """Return region geometries belonging to the requested scope prefixes."""
    mask = regions.index.to_series().astype(str).str.startswith(prefixes)
    return regions[mask].copy()


def carrier_bus_balance(n, settings, buses):
    """Return scoped bus energy-balance sizes for the CO2 stored carrier."""
    eb = n.statistics.energy_balance(bus_carrier=CO2_STORED, groupby=["bus", "carrier"])
    eb = drop_transmission_losses(n, eb)
    bus_size = eb.groupby(level=["bus", "carrier"]).sum().div(settings["unit_conversion"])
    bus_size = bus_size[bus_size.index.get_level_values("bus").isin(buses)]
    return bus_size.sort_values(ascending=False)


def drop_transmission_losses(n, energy_balance):
    """Remove transmission-carrier losses from an energy-balance table."""
    transmission_carriers = get_transmission_carriers(n, bus_carrier=CO2_STORED).rename(
        {"name": "carrier"}
    )
    components = transmission_carriers.unique("component")
    carriers = transmission_carriers.unique("carrier")
    carriers_in_balance = carriers[
        carriers.isin(energy_balance.index.get_level_values("carrier"))
    ]
    if len(components) and len(carriers_in_balance):
        energy_balance.loc[components] = energy_balance.loc[components].drop(
            index=carriers_in_balance, level="carrier"
        )
    return energy_balance.dropna()


def transmission_flow(n, settings):
    """Return CO2 stored transmission flow converted to plotting units."""
    flow = n.statistics.transmission(groupby=False, bus_carrier=CO2_STORED).div(
        settings["unit_conversion"]
    )
    if flow.empty:
        return flow
    reversed_mask = flow.index.get_level_values(1).str.contains("reversed")
    reversed_flow = flow[reversed_mask].rename(lambda x: x.replace("-reversed", ""))
    return flow[~reversed_mask].subtract(reversed_flow, fill_value=0)


def component_width(flow, component, fallback):
    """Return absolute transmission width for one component."""
    return flow.get(component, fallback).abs()


def scoped_branch_width(width, components, buses):
    """Keep branch widths only where a branch touches scoped buses."""
    if width.empty or components.empty:
        return width
    bus_columns = [column for column in components.columns if column.startswith("bus")]
    touches_scope = pd.Series(False, index=components.index)
    for column in bus_columns:
        touches_scope |= components[column].isin(buses)
    return width.where(width.index.to_series().map(touches_scope).fillna(False), 0.0)


def clipped_plot_width(width, factor, maximum):
    """Scale branch width and apply an optional visual cap."""
    plot_width = width * factor
    if maximum is not None:
        plot_width = plot_width.clip(upper=maximum)
    return plot_width


def average_prices(n, regions, settings, buses):
    """Return scoped region prices for background shading."""
    co2_buses = n.buses[(n.buses.carrier == CO2_STORED) & n.buses.index.isin(buses)].index
    if co2_buses.empty:
        regions["price"] = 0.0
        return regions, 0.0, 0.0

    weights = n.snapshot_weightings.generators
    prices = weights @ n.buses_t.marginal_price[co2_buses] / weights.sum()
    level = "name" if PYPSA_V1 else "Bus"
    price = prices.rename(n.buses.location).groupby(level=level).mean()

    if "CO2Limit" in n.global_constraints.index:
        price = price - n.global_constraints.loc["CO2Limit", "mu"]

    regions["price"] = price.reindex(regions.index).fillna(0)
    return regions, color_scale_min(regions, settings), color_scale_max(regions, settings)


def color_scale_min(regions, settings):
    """Return configured or data-driven lower color scale bound."""
    return settings["vmin"] if settings["vmin"] is not None else regions.price.min()


def color_scale_max(regions, settings):
    """Return configured or data-driven upper color scale bound."""
    return settings["vmax"] if settings["vmax"] is not None else regions.price.max()


def carrier_colors(n, plotting_config, bus_size):
    """Return colors matching the bus-size carrier order."""
    n.carriers.update({"color": plotting_config["tech_colors"]})
    colors = n.carriers.color.copy().replace("", "grey")
    return bus_size.index.get_level_values("carrier").unique().to_series().map(colors)


def add_legends(ax, n, bus_size, settings, branch_width_factor, branch_width_max):
    """Add supply, consumption, bus-size, and branch-size legends."""
    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "title_fontproperties": {"weight": "bold"},
    }
    n.carriers.loc["", "color"] = "None"
    add_carrier_legends(ax, n, bus_size, legend_kwargs)
    add_bus_legend(ax, settings, legend_kwargs)
    add_branch_legend(ax, settings, branch_width_factor, branch_width_max, legend_kwargs)


def add_network_only_legend(ax, settings, branch_width_factor, branch_width_max):
    """Add only the branch-size legend for network-only maps."""
    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "title_fontproperties": {"weight": "bold"},
    }
    add_branch_legend(ax, settings, branch_width_factor, branch_width_max, legend_kwargs)


def add_carrier_legends(ax, n, bus_size, legend_kwargs):
    """Add separate supply and consumption carrier legends."""
    if bus_size.empty:
        return
    pos_carriers = bus_size[bus_size > 0].index.unique("carrier")
    neg_carriers = bus_size[bus_size < 0].index.unique("carrier")
    add_legend_patches(
        ax,
        n.carriers.color[sorted(pos_carriers)],
        sorted(pos_carriers),
        legend_kw={"bbox_to_anchor": (0, -0.15), "ncol": 1, "title": "Supply", **legend_kwargs},
    )
    add_legend_patches(
        ax,
        n.carriers.color[sorted(neg_carriers)],
        sorted(neg_carriers),
        legend_kw={"bbox_to_anchor": (0.48, -0.15), "ncol": 1, "title": "Consumption", **legend_kwargs},
    )


def add_bus_legend(ax, settings, legend_kwargs):
    """Add bus-size legend if configured."""
    if settings["bus_sizes"] is None:
        return
    add_legend_semicircles(
        ax,
        [s * settings["bus_factor"] * SEMICIRCLE_CORRECTION_FACTOR for s in settings["bus_sizes"]],
        [f"{s} {settings['unit']}" for s in settings["bus_sizes"]],
        patch_kw={"color": "#666"},
        legend_kw={"bbox_to_anchor": (0, 1), **legend_kwargs},
    )


def add_branch_legend(ax, settings, branch_width_factor, branch_width_max, legend_kwargs):
    """Add branch-width legend if configured."""
    if settings["branch_sizes"] is None:
        return
    widths = [s * branch_width_factor for s in settings["branch_sizes"]]
    if branch_width_max is not None:
        widths = [min(width, branch_width_max) for width in widths]
    add_legend_lines(
        ax,
        widths,
        [f"{s} {settings['unit']}" for s in settings["branch_sizes"]],
        patch_kw={"color": "#666"},
        legend_kw={"bbox_to_anchor": (0.25, 1), **legend_kwargs},
    )


def plot_focus_map(
    n,
    regions,
    scope,
    plotting_config,
    settings,
    output_path,
    show_balances=True,
):
    """Plot one focused static CO2 stored balance map."""
    scope_config = FOCUS_SCOPES[scope]
    buses = scope_buses(n, scope_config["prefixes"])
    scoped_regions = scope_regions(regions, scope_config["prefixes"])
    bus_size = carrier_bus_balance(n, settings, buses)
    flow = transmission_flow(n, settings)
    fallback = pd.Series(dtype=float)

    line_width = scoped_branch_width(component_width(flow, "Line", fallback), n.lines, buses)
    link_width = scoped_branch_width(component_width(flow, "Link", fallback), n.links, buses)
    line_flow = flow.get("Line")
    link_flow = flow.get("Link")
    transformer_flow = flow.get("Transformer")

    branch_factor = settings["branch_factor"] * FOCUS_BRANCH_FACTOR_SCALE
    branch_width_max = FOCUS_BRANCH_WIDTH_MAX
    line_plot_width = clipped_plot_width(line_width, branch_factor, branch_width_max)
    link_plot_width = clipped_plot_width(link_width, branch_factor, branch_width_max)

    scoped_regions, vmin, vmax = average_prices(n, scoped_regions, settings, buses)
    crs = load_projection(plotting_config)
    fig, ax = plt.subplots(
        figsize=scope_config["figsize"], subplot_kw={"projection": crs}, layout="constrained"
    )
    plot_regions(scoped_regions, crs, ax, settings, vmin, vmax)
    n.plot(
        bus_size=bus_size * settings["bus_factor"] if show_balances else 0,
        bus_color=carrier_colors(n, plotting_config, bus_size) if show_balances else None,
        bus_split_circle=show_balances,
        line_width=line_plot_width,
        link_width=link_plot_width,
        line_flow=line_flow * settings["flow_factor"] if line_flow is not None else None,
        link_flow=link_flow * settings["flow_factor"] if link_flow is not None else None,
        link_color=settings.get("branch_color") or "darkseagreen",
        transformer_flow=transformer_flow * settings["flow_factor"] if transformer_flow is not None else None,
        ax=ax,
        margin=0.05,
        geomap_color={"border": "darkgrey", "coastline": "darkgrey"},
        geomap=True,
        boundaries=scope_config["boundaries"],
    )
    title_kind = "balance" if show_balances else "network"
    ax.set_title(f"CO$_2$ {title_kind} — {scope_config['label']}", fontsize=10)
    add_colorbar(fig, ax, settings, vmin, vmax)
    if show_balances:
        add_legends(ax, n, bus_size, settings, branch_factor, branch_width_max)
    else:
        add_network_only_legend(ax, settings, branch_factor, branch_width_max)
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved focused CO2 stored map: %s", output_path)


def plot_regions(regions, crs, ax, settings, vmin, vmax):
    """Plot regional price shading below network overlays."""
    if regions.empty:
        return
    regions.to_crs(crs.proj4_init).plot(
        ax=ax,
        column="price",
        cmap=settings["cmap"],
        vmin=vmin,
        vmax=vmax,
        edgecolor="darkgrey",
        linewidth=0.2,
        alpha=0.45,
    )


def add_colorbar(fig, ax, settings, vmin, vmax):
    """Add marginal-price colorbar."""
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=settings["cmap"], norm=norm)
    cbr = fig.colorbar(
        sm,
        ax=ax,
        label=f"Average Marginal Price [{settings['region_unit']}]",
        shrink=0.95,
        pad=0.03,
        aspect=50,
        orientation="horizontal",
    )
    cbr.outline.set_edgecolor("None")


def run_plotting_job(snakemake):
    """Run one Snakemake job for all requested focus outputs."""
    set_scenario_config(snakemake)
    logger.info("Running focused CCS balance map job with wildcards: %s", dict(snakemake.wildcards))
    logger.info("Requested focused CCS map outputs: %s", list(snakemake.output.keys()))

    network = prepare_network(snakemake.input.network, snakemake.config)
    prepare_bus_locations(network, snakemake.params.plotting)
    region_shapes = gpd.read_file(snakemake.input.regions).set_index("name")

    for output_name, output in snakemake.output.items():
        focus_scope, show_balances = focus_scope_and_balance_mode(output_name)
        logger.info(
            "Plotting %s map for %s to %s",
            "balance" if show_balances else "network-only",
            focus_scope,
            output,
        )
        plot_focus_map(
            network,
            region_shapes,
            focus_scope,
            snakemake.params.plotting,
            snakemake.params.settings,
            output,
            show_balances=show_balances,
        )


def run_all_mock_jobs(configfiles=None):
    """Run focused CCS maps for all configured local mock jobs."""
    from scripts._helpers import mock_snakemake

    if configfiles is None:
        configfiles = ["config/config.nrw.yaml"]

    wildcard_sets = mock_wildcard_sets_from_config(configfiles[0])
    logger.info("Running %s local mock jobs from %s", len(wildcard_sets), configfiles[0])

    for wildcards in wildcard_sets:
        snakemake = mock_snakemake(
            "plot_ccs_balance_map_focus",
            **wildcards,
            configfiles=configfiles,
        )
        configure_logging(snakemake)
        run_plotting_job(snakemake)


def running_interactively():
    """Return True when executed from IPython/Jupyter instead of Snakemake."""
    try:
        get_ipython  # noqa: F821
    except NameError:
        return False
    return True


if __name__ == "__main__":
    if "snakemake" in globals() and not running_interactively():
        configure_logging(snakemake)
        run_plotting_job(snakemake)
    else:
        run_all_mock_jobs()
