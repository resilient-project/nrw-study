#!/usr/bin/env python3
# Save as: scripts/plot_co2_pipeline_overview.py

# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plot CO2 pipeline capacity overview for all scenarios side-by-side.
Produces separate EU, Germany, and NRW figures.

Shows installed CO2 pipeline capacity split by:
- Long-distance (inter-regional): carrier "CO2 pipeline"
- Short (intra-regional): carrier "CO2 pipeline short"

Utilization rate (%) is shown inside each block.
Total capacity is shown on top.
"""

import logging
import ast
import re
import pypsa
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

CO2_PIPELINE_CARRIERS = ["CO2 pipeline", "CO2 pipeline short"]
CO2_BUS_CARRIERS = ["co2 stored", "co2 sequestered"]
NRW_PREFIX = "DEA"

SCOPE_LABELS = {
    "eu": "EU",
    "de": "Germany",
    "nrw": "NRW",
}
SCOPE_BUS_PREFIXES = {
    "eu": None,
    "de": ("DE",),
    "nrw": (NRW_PREFIX,),
}

PIPE_TYPE_LABELS = {
    "CO2 pipeline": "CO2 Long-distance",
    "CO2 pipeline short": "CO2 Short (intra-regional)",
}

# Keep this plot locally distinguishable: the global tech_colors currently assigns
# the same color to both CO2 pipeline carriers because they share one tech group.
PIPE_TYPE_COLORS = {
    "CO2 Long-distance": "#0072B2",        # blue, Okabe-Ito palette
    "CO2 Short (intra-regional)": "#D55E00",  # vermillion, Okabe-Ito palette
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_cap_col(df, prefer="p_nom_opt", fallback="p_nom"):
    """Safely determine which capacity column to use."""
    if df.empty:
        return fallback
    if prefer in df.columns and df[prefer].abs().sum() > 1e-3:
        return prefer
    return fallback


def _as_tuple(value):
    """Return YAML tuple strings or sequences as tuples."""
    if isinstance(value, str):
        return ast.literal_eval(value)
    return tuple(value)


def _network_paths_from_input(network_paths, run_names, planning_horizons):
    """Map declared Snakemake network inputs to (path, run_name, year) tuples."""
    if not network_paths:
        return []

    planning_horizons_order = [str(year) for year in planning_horizons]
    planning_horizons_set = set(planning_horizons_order)
    file_map = []

    for path_like in network_paths:
        path = Path(path_like)
        path_parts = set(path.parts)

        run_name = next((name for name in run_names if name in path_parts), None)
        match = re.search(r"(?:^|_)(\d{4})(?:\.nc)?$", path.name)
        year = match.group(1) if match else None

        if run_name is None or year not in planning_horizons_set:
            logger.warning(f"Could not assign network input to run/year: {path}")
            continue

        file_map.append((str(path), run_name, year))

    order = {
        (run, str(year)): i
        for i, (run, year) in enumerate(
            (run, year) for run in run_names for year in planning_horizons_order
        )
    }
    file_map.sort(key=lambda item: order.get((item[1], str(item[2])), len(order)))
    return file_map


def find_network_files(results_dir, run_names, planning_horizons, network_paths=None):
    """Find network files for all runs and planning horizons."""
    file_map = _network_paths_from_input(network_paths, run_names, planning_horizons)
    if file_map:
        return file_map

    file_map = []

    for run_name in run_names:
        for year in planning_horizons:
            candidates = [
                Path(results_dir) / run_name / "postnetworks" / f"base_s_adm___{year}.nc",
                Path(results_dir) / run_name / "postnetworks" / f"elec_s_adm___{year}.nc",
                Path(results_dir) / run_name / "networks" / f"base_s_adm___{year}.nc",
                Path(results_dir) / run_name / "networks" / f"elec_s_adm___{year}.nc",
            ]

            found = False
            for path in candidates:
                if path.exists():
                    file_map.append((str(path), run_name, year))
                    found = True
                    break

            if not found:
                run_dir = Path(results_dir) / run_name
                if run_dir.exists():
                    glob_results = list(run_dir.rglob(f"*{year}*.nc"))
                    if glob_results:
                        file_map.append((str(glob_results[0]), run_name, year))
                        found = True

            if not found:
                logger.warning(f"Not found: {run_name} / {year}")

    return file_map


def get_scope_outputs(snakemake):
    """Return requested scope outputs from the Snakemake object."""
    outputs = {}
    for scope in SCOPE_LABELS:
        try:
            outputs[scope] = getattr(snakemake.output, scope)
        except AttributeError:
            continue

    if outputs:
        return outputs

    try:
        output_path = Path(snakemake.output.plot)
    except AttributeError:
        output_path = Path(snakemake.output[0])

    stem = output_path.stem
    for suffix in ("_eu", "_de", "_nrw"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    if stem in {
        "co2_pipeline_overview",
        "co2_pipeline_capacity_overview",
    }:
        stem = "co2_pipeline_capacity"

    return {
        scope: output_path.with_name(f"{stem}_{scope}{output_path.suffix}")
        for scope in SCOPE_LABELS
    }


def filter_links_by_scope(n, links, scope):
    """Filter pipeline links by geographical scope using bus prefix."""
    prefixes = SCOPE_BUS_PREFIXES[scope]
    if prefixes is None:
        return links

    buses = n.buses[n.buses.index.to_series().str.startswith(prefixes)].index
    return links[links.bus0.isin(buses) | links.bus1.isin(buses)]


def get_co2_pipeline_links(n):
    """
    Extract CO2 pipeline links, excluding reversed duplicates.
    Returns only non-reversed links with p_nom_opt > 0.
    """
    co2_links = n.links[
        n.links.carrier.isin(CO2_PIPELINE_CARRIERS)
        & ~n.links.index.str.contains("-reversed", na=False)
    ].copy()

    cap_col = get_cap_col(co2_links)
    active = co2_links[co2_links[cap_col] > 1e-3].copy()

    return active, cap_col


def compute_utilization_by_type(n, links, cap_col):
    """
    Compute utilization rate separately for each pipeline type.
    Returns dict: {carrier_label: utilization_pct}
    """
    utilization = {}

    if links.empty:
        return utilization

    if not hasattr(n, 'links_t') or 'p0' not in n.links_t or n.links_t.p0.empty:
        return utilization

    # Get snapshot weightings
    if hasattr(n.snapshot_weightings, 'generators'):
        weights = n.snapshot_weightings.generators
    elif hasattr(n.snapshot_weightings, 'objective'):
        weights = n.snapshot_weightings.objective
    else:
        weights = n.snapshot_weightings.iloc[:, 0]

    for carrier in CO2_PIPELINE_CARRIERS:
        carrier_links = links[links.carrier == carrier]
        if carrier_links.empty:
            continue

        max_flow = carrier_links[cap_col].sum() * 8760.0

        if max_flow < 1e-3:
            continue

        valid_links = carrier_links.index.intersection(n.links_t.p0.columns)
        if valid_links.empty:
            utilization[PIPE_TYPE_LABELS[carrier]] = 0.0
            continue

        actual_flow = (
            n.links_t.p0[valid_links].abs()
            .multiply(weights, axis=0)
            .sum().sum()
        )

        utilization[PIPE_TYPE_LABELS[carrier]] = actual_flow / max_flow * 100

    return utilization


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_pipeline_data(file_map, scopes):
    """
    Extract CO2 pipeline capacity and per-type utilization.

    Returns:
        capacity_by_scope: dict[scope] -> list of dicts
        utilization_by_scope: dict[scope] -> list of dicts
    """
    capacity_by_scope = {scope: [] for scope in scopes}
    utilization_by_scope = {scope: [] for scope in scopes}

    for path_str, run_name, year in file_map:
        try:
            n = pypsa.Network(path_str)
        except Exception as e:
            logger.error(f"Error loading {path_str}: {e}")
            continue

        active_links, cap_col = get_co2_pipeline_links(n)

        if active_links.empty:
            logger.info(f"  No active CO2 pipelines in {run_name}/{year}")
            continue

        for scope in scopes:
            scoped_links = filter_links_by_scope(n, active_links, scope)

            if scoped_links.empty:
                continue

            # Capacity by type
            for carrier in CO2_PIPELINE_CARRIERS:
                carrier_links = scoped_links[scoped_links.carrier == carrier]
                cap_gw = carrier_links[cap_col].sum() / 1e3
                if cap_gw > 1e-6:
                    capacity_by_scope[scope].append({
                        "planning_horizon": str(year),
                        "name": run_name,
                        "carrier": PIPE_TYPE_LABELS[carrier],
                        "value": cap_gw,
                    })

            # Utilization per type
            util_by_type = compute_utilization_by_type(n, scoped_links, cap_col)
            for carrier_label, util_pct in util_by_type.items():
                utilization_by_scope[scope].append({
                    "planning_horizon": str(year),
                    "name": run_name,
                    "carrier": carrier_label,
                    "value": util_pct,
                })

    return capacity_by_scope, utilization_by_scope


# =============================================================================
# PLOTTING
# =============================================================================

def save_empty_plot(output_path, scope, figsize, dpi):
    """Create an empty placeholder plot."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.text(
        0.5, 0.5,
        f"No CO2 pipeline capacity found ({SCOPE_LABELS[scope]})",
        ha="center", va="center", transform=ax.transAxes, fontsize=14,
    )
    ax.set_axis_off()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_pipeline_scope(
    capacity_data,
    utilization_data,
    scope,
    output_path,
    planning_horizons,
    run_order,
    run_nice_names,
    tech_colors,
    figsize,
    dpi,
    font,
    fontsize,
    subfontsize,
):
    """
    Plot CO2 pipeline capacity: long vs short (stacked).
    Total on top, utilization % inside each block.
    """
    if not capacity_data:
        save_empty_plot(output_path, scope, figsize, dpi)
        return

    cap_df = pd.DataFrame(capacity_data)

    # Build utilization lookup: (planning_horizon, name, carrier) -> pct
    util_lookup = {}
    if utilization_data:
        for row in utilization_data:
            util_lookup[(row["planning_horizon"], row["name"], row["carrier"])] = row["value"]

    n_planning_horizons = len(planning_horizons)

    ymax = (
        cap_df.groupby(["planning_horizon", "name"], observed=True)["value"]
        .sum().max()
    )
    ymax = 1 if pd.isna(ymax) or ymax <= 0 else ymax

    pipe_colors = PIPE_TYPE_COLORS
    pipe_order = ["CO2 Long-distance", "CO2 Short (intra-regional)"]

    x_anchor = 0
    ncol = 3
    handlelength = 1
    handleheight = 1.1
    xpad = 0.03

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_planning_horizons,
        figsize=figsize,
        dpi=dpi,
        sharey=True,
        tight_layout=True,
    )
    plt.rc("font", **font)

    for i, planning_horizon in enumerate(planning_horizons):
        ax = np.atleast_1d(axes)[i]
        ph_str = str(planning_horizon)

        cap_subset = cap_df.query("planning_horizon == @ph_str").copy()
        if cap_subset.empty:
            cap_pivot = pd.DataFrame(index=run_order)
        else:
            cap_pivot = cap_subset.pivot(
                index="name", columns="carrier", values="value"
            ).fillna(0)

        cap_pivot = cap_pivot.reindex(run_order).fillna(0)

        data_order = [col for col in pipe_order if col in cap_pivot.columns]
        data_order += [col for col in cap_pivot.columns if col not in data_order]
        cap_pivot = cap_pivot[data_order]

        cap_pivot = cap_pivot.rename(index=run_nice_names)
        run_labels = [run_nice_names.get(name, name) for name in run_order]

        if cap_pivot.empty or len(cap_pivot.columns) == 0:
            ax.set_xticks(np.arange(len(run_labels)))
            ax.set_xticklabels(run_labels, rotation=90, fontsize=subfontsize)
        else:
            cap_pivot.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                width=0.8,
                color=[pipe_colors.get(col, "#999999") for col in cap_pivot.columns],
                legend=False,
            )

            # Utilization % inside each block
            for j, run_label in enumerate(cap_pivot.index):
                run_name = run_order[j] if j < len(run_order) else None
                bottom = 0.0

                for carrier_label in data_order:
                    block_height = cap_pivot.loc[run_label, carrier_label]

                    if block_height > ymax * 0.05:
                        util_pct = util_lookup.get((ph_str, run_name, carrier_label), None)

                        if util_pct is not None:
                            mid_y = bottom + block_height / 2
                            ax.text(
                                x=j,
                                y=mid_y,
                                s=f"{util_pct:.0f}%",
                                ha="center",
                                va="center",
                                fontsize=subfontsize - 1,
                                color="white",
                                fontweight="bold",
                            )

                    bottom += block_height

        # Formatting
        ax.set_xlabel(f"{ph_str}", fontsize=fontsize)
        if i == 0:
            ax.set_ylabel(
                f"CO$_2$ pipeline capacity ({SCOPE_LABELS[scope]}) [GW]",
                fontsize=fontsize,
            )

        ax.set_ylim(0, ymax * 1.15)
        ax.set_xticklabels(
            cap_pivot.index if len(cap_pivot.index) else run_labels,
            rotation=90,
            fontsize=subfontsize,
        )
        ax.grid(False)

        if i > 0:
            ax.yaxis.set_visible(False)

        # Total on top
        if len(cap_pivot.columns) > 0:
            cap_totals = cap_pivot.sum(axis=1)
            for j, cap_val in enumerate(cap_totals):
                if cap_val > 0:
                    ax.text(
                        x=j, y=cap_val, s=f"{cap_val:.0f}",
                        ha="center", va="bottom", fontsize=subfontsize,
                    )

        ax.axhline(0, color="black", lw=0.5)

    # Tick styling
    for ax in np.atleast_1d(axes):
        ax.tick_params(axis="y", labelsize=subfontsize)

    # Legend
    all_carriers = cap_df["carrier"].unique()
    legend_items = [c for c in pipe_order if c in all_carriers]
    for c in all_carriers:
        if c not in legend_items:
            legend_items.append(c)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=pipe_colors.get(c, "#999999"), label=c)
        for c in legend_items[::-1]
    ]

    legend = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(x_anchor + xpad, 0.03),
        ncol=ncol,
        fontsize=subfontsize,
        title="(%) = utilization rate",
        title_fontsize=subfontsize,
        frameon=False,
        handlelength=handlelength,
        handleheight=handleheight,
    )
    legend.get_title().set_fontweight("bold")
    legend._legend_box.align = "left"

    # Border styling
    for ax in np.atleast_1d(axes):
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("black")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_co2_pipeline_overview",
            configfiles=["config/config.nrw.yaml"],
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    config = snakemake.config
    plotting = snakemake.params.plotting_fig

    # =========================================================================
    # READ CONFIG
    # =========================================================================

    tech_colors = config["plotting"]["tech_colors"]

    figsize = _as_tuple(plotting["figsize"])
    fontsize = plotting["font"]["size"]
    subfontsize = fontsize
    dpi = plotting["dpi"]
    font = plotting["font"]

    planning_horizons = config["scenario"]["planning_horizons"]
    lt_order = [col for col in plotting["run_order"]]
    lt_order_nice_names = plotting["nice_names"]

    # =========================================================================
    # FIND AND LOAD NETWORKS
    # =========================================================================

    results_dir = Path("results/nrw")
    scope_outputs = get_scope_outputs(snakemake)

    try:
        input_networks = list(snakemake.input.networks)
    except AttributeError:
        input_networks = list(snakemake.input)

    file_map = find_network_files(
        results_dir, lt_order, planning_horizons, input_networks
    )

    if not file_map:
        logger.error(f"No network files found in {results_dir}")
        raise FileNotFoundError(f"No networks in {results_dir}")

    # =========================================================================
    # EXTRACT DATA
    # =========================================================================

    capacity_by_scope, utilization_by_scope = collect_pipeline_data(
        file_map, scope_outputs.keys()
    )

    # =========================================================================
    # PLOT
    # =========================================================================

    for scope, output_path in scope_outputs.items():
        plot_pipeline_scope(
            capacity_by_scope[scope],
            utilization_by_scope[scope],
            scope,
            output_path,
            planning_horizons,
            lt_order,
            lt_order_nice_names,
            tech_colors,
            figsize,
            dpi,
            font,
            fontsize,
            subfontsize,
        )
