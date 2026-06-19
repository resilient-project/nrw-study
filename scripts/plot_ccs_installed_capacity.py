#!/usr/bin/env python3
# Save as: scripts/plot_ccs_installed_capacity.py

# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plot exact CCS CO2 capture capacity for all scenarios side-by-side.
Produces separate EU, Germany, and NRW figures using PyPSA plotting config.

The figures show installed CO2 capture capacity as a full-load annual equivalent:
``p_nom_opt`` × CO2 capture coefficient × 8760 h/a.
Utilization rate (%) is shown below the capacity value.
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
# CCS CONFIGURATION
# =============================================================================

CCS_CAPTURE_CARRIERS = [
    "SMR CC",
    "DAC",
    "gas for industry CC",
    "process emissions CC",
    "solid biomass for industry CC",
    "urban central gas CHP CC",
    "urban central solid biomass CHP CC",
]

CCS_EXCLUDE = [
    "CCGT",
    "OCGT",
    "co2 sequestered",
    "CO2 pipeline",
    "CO2 pipeline short",
]

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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_cap_col(df, prefer="p_nom_opt", fallback="p_nom"):
    """Safely determine which nominal capacity column to use."""
    if prefer in df.columns and df[prefer].abs().sum() > 1e-3:
        return prefer
    return fallback


def co2_capture_capacity_mt_per_a(links, cap_col):
    """
    Return full-load CO2 capture capacity by link in MtCO2/a.

    p_nom_opt × CO2-stored output coefficient × 8760 h/a / 1e6
    """
    capture_factor = pd.Series(0.0, index=links.index)

    for i in range(1, 5):
        bus_col = f"bus{i}"
        if bus_col not in links.columns:
            continue

        efficiency_col = "efficiency" if i == 1 else f"efficiency{i}"
        if efficiency_col not in links.columns:
            continue

        is_co2_stored_output = links[bus_col].fillna("").astype(str).str.contains(
            "co2 stored", case=False, na=False
        )
        capture_factor.loc[is_co2_stored_output] += links.loc[
            is_co2_stored_output, efficiency_col
        ].abs()

    return links[cap_col].fillna(0.0) * capture_factor * 8760.0 / 1e6


def co2_actual_captured_mt_per_a(n, links):
    """
    Return actual annual CO2 captured per link in MtCO2/a from dispatch time series.

    In PyPSA, output flows (p1, p2, ...) are NEGATIVE when flowing from the link
    to the bus. We take abs() to get the captured amount.
    Snapshot weightings are applied to convert from instantaneous to annual.
    """
    actual_capture = pd.Series(0.0, index=links.index)

    if links.empty:
        return actual_capture

    # Get snapshot weightings
    if hasattr(n.snapshot_weightings, 'generators'):
        weights = n.snapshot_weightings.generators
    elif hasattr(n.snapshot_weightings, 'objective'):
        weights = n.snapshot_weightings.objective
    else:
        weights = n.snapshot_weightings.iloc[:, 0]

    for i in range(1, 5):
        bus_col = f"bus{i}"
        if bus_col not in links.columns:
            continue

        efficiency_col = "efficiency" if i == 1 else f"efficiency{i}"
        if efficiency_col not in links.columns:
            continue

        is_co2_stored_output = links[bus_col].fillna("").astype(str).str.contains(
            "co2 stored", case=False, na=False
        )
        relevant_links = links[is_co2_stored_output].index

        if relevant_links.empty:
            continue

        p_col = f"p{i}"
        if not hasattr(n, 'links_t') or p_col not in n.links_t:
            continue

        links_t_p = n.links_t[p_col]
        if links_t_p.empty:
            continue

        valid_links = relevant_links.intersection(links_t_p.columns)
        if valid_links.empty:
            continue

        annual_flow = (
            links_t_p[valid_links].abs()
            .multiply(weights, axis=0)
            .sum(axis=0)
        )

        actual_capture.loc[valid_links] += annual_flow

    return actual_capture / 1e6


def find_ccs_carriers(n):
    """Find carbon capture link carriers present in the network."""
    all_carriers = set(n.links.carrier.unique())

    found = [c for c in CCS_CAPTURE_CARRIERS if c in all_carriers]

    for c in all_carriers:
        c_str = str(c)
        is_cc = (
            c_str.endswith(" CC")
            or "CCS" in c_str
            or c_str == "DAC"
            or "capture" in c_str.lower()
        )
        is_excluded = c_str in CCS_EXCLUDE or "pipeline" in c_str.lower()

        if is_cc and not is_excluded and c not in found:
            found.append(c)

    return sorted(found)


def _as_tuple(value):
    """Return YAML tuple strings or sequences as tuples for matplotlib."""
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
        "ccs_overview",
        "ccs_capture_capacity_overview",
    }:
        stem = "ccs_co2_capture_capacity"

    return {
        scope: output_path.with_name(f"{stem}_{scope}{output_path.suffix}")
        for scope in SCOPE_LABELS
    }


def filter_links_by_scope(n, links, scope):
    """Filter CCS links to EU, Germany, or NRW by connected bus prefix."""
    prefixes = SCOPE_BUS_PREFIXES[scope]
    if prefixes is None:
        return links

    buses = n.buses[n.buses.index.to_series().str.startswith(prefixes)].index
    return links[links.bus0.isin(buses) | links.bus1.isin(buses)]


def carrier_color(carrier, tech_colors):
    """Return the exact-carrier color, with generic CCS fallback."""
    return tech_colors.get(
        carrier,
        tech_colors.get("CC", tech_colors.get("CCS", "#999999")),
    )


def collect_ccs_data(file_map, scopes):
    """
    Extract CCS carrier capacities AND actual dispatch for all requested scopes.

    Returns:
        capacity_by_scope: dict[scope] -> list of dicts
        dispatch_by_scope: dict[scope] -> list of dicts
    """
    capacity_by_scope = {scope: [] for scope in scopes}
    dispatch_by_scope = {scope: [] for scope in scopes}

    for path_str, run_name, year in file_map:
        try:
            n = pypsa.Network(path_str)
        except Exception as e:
            logger.error(f"Error loading {path_str}: {e}")
            continue

        ccs_carriers = find_ccs_carriers(n)
        ccs_links = n.links[
            n.links.carrier.isin(ccs_carriers)
            & ~n.links.index.str.contains("-reversed", na=False)
        ].copy()

        if ccs_links.empty:
            logger.info(f"  No CCS links in {run_name}/{year}")
            continue

        cap_col = get_cap_col(ccs_links)

        # Compute capacity and actual capture once
        ccs_links["capacity_mt_per_a"] = co2_capture_capacity_mt_per_a(ccs_links, cap_col)
        ccs_links["actual_mt_per_a"] = co2_actual_captured_mt_per_a(n, ccs_links)

        for scope in scopes:
            scoped_links = filter_links_by_scope(n, ccs_links, scope)

            if scoped_links.empty:
                continue

            # Capacity
            cap_grouped = scoped_links.groupby("carrier")["capacity_mt_per_a"].sum()
            for carrier, value in cap_grouped.items():
                capacity_by_scope[scope].append({
                    "planning_horizon": str(year),
                    "name": run_name,
                    "carrier": carrier,
                    "value": value,
                })

            # Dispatch
            disp_grouped = scoped_links.groupby("carrier")["actual_mt_per_a"].sum()
            for carrier, value in disp_grouped.items():
                dispatch_by_scope[scope].append({
                    "planning_horizon": str(year),
                    "name": run_name,
                    "carrier": carrier,
                    "value": value,
                })

    return capacity_by_scope, dispatch_by_scope


def save_empty_plot(output_path, scope, figsize, dpi):
    """Create an empty placeholder plot."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.text(
        0.5, 0.5,
        f"No CCS CO2 capture capacity found ({SCOPE_LABELS[scope]})",
        ha="center", va="center", transform=ax.transAxes, fontsize=14,
    )
    ax.set_axis_off()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.warning(f"Saved empty CCS overview: {output_path}")


def plot_ccs_scope(
    capacity_data,
    dispatch_data,
    scope,
    output_path,
    planning_horizons,
    run_order,
    run_nice_names,
    legend_order,
    tech_colors,
    figsize,
    dpi,
    font,
    fontsize,
    subfontsize,
):
    """
    Plot CCS overview:
    - Solid stacked bars: installed capacity (full-load equivalent MtCO2/a)
    - Label on top: total capacity value
    - Label below: utilization (%) in brackets
    """
    if not capacity_data:
        save_empty_plot(output_path, scope, figsize, dpi)
        return

    cap_df = pd.DataFrame(capacity_data)
    has_dispatch = bool(dispatch_data) and len(dispatch_data) > 0
    disp_df = pd.DataFrame(dispatch_data) if has_dispatch else pd.DataFrame()

    if has_dispatch and disp_df["value"].sum() < 1e-6:
        has_dispatch = False

    n_planning_horizons = len(planning_horizons)

    ymax = (
        cap_df.groupby(["planning_horizon", "name"], observed=True)["value"]
        .sum().max()
    )
    ymax = 1 if pd.isna(ymax) or ymax <= 0 else ymax
    ymin = 0

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

        # --- Capacity data ---
        cap_subset = cap_df.query("planning_horizon == @ph_str").copy()
        if cap_subset.empty:
            cap_pivot = pd.DataFrame(index=run_order)
        else:
            cap_pivot = cap_subset.pivot(
                index="name", columns="carrier", values="value"
            ).fillna(0)
        cap_pivot = cap_pivot.reindex(run_order).fillna(0)

        # --- Dispatch data (for utilization label only) ---
        if has_dispatch:
            disp_subset = disp_df.query("planning_horizon == @ph_str").copy()
            if not disp_subset.empty:
                disp_pivot = disp_subset.pivot(
                    index="name", columns="carrier", values="value"
                ).fillna(0)
                disp_pivot = disp_pivot.reindex(run_order).fillna(0)
            else:
                disp_pivot = None
        else:
            disp_pivot = None

        # Order columns by legend_order
        data_order = [col for col in legend_order if col in cap_pivot.columns]
        data_order += [col for col in cap_pivot.columns if col not in data_order]
        cap_pivot = cap_pivot[data_order]

        # Rename to nice names
        cap_pivot = cap_pivot.rename(index=run_nice_names)
        if disp_pivot is not None:
            disp_pivot = disp_pivot.rename(index=run_nice_names)

        run_labels = [run_nice_names.get(name, name) for name in run_order]

        if cap_pivot.empty or len(cap_pivot.columns) == 0:
            ax.set_xticks(np.arange(len(run_labels)))
            ax.set_xticklabels(run_labels, rotation=90, fontsize=subfontsize)
        else:
            # Plot capacity as solid stacked bars
            cap_pivot.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                width=0.8,
                color=[carrier_color(col, tech_colors) for col in cap_pivot.columns],
                legend=False,
            )

        # Formatting
        ax.set_xlabel(f"{ph_str}", fontsize=fontsize)
        if i == 0:
            ax.set_ylabel(
            f"Installed CO$_2$ capture capacity ({SCOPE_LABELS[scope]})\n"
            "[MtCO$_2$/a] (utilization %)",
            fontsize=fontsize,
          )


        ax.set_ylim(ymin, ymax * 1.2)
        ax.set_xticklabels(
            cap_pivot.index if len(cap_pivot.index) else run_labels,
            rotation=90,
            fontsize=subfontsize,
        )
        ax.grid(False)

        if i > 0:
            ax.yaxis.set_visible(False)

        # Labels on top
        if len(cap_pivot.columns) > 0:
            cap_totals = cap_pivot.sum(axis=1)

            for j, cap_val in enumerate(cap_totals):
                if cap_val > 0:
                    if has_dispatch and disp_pivot is not None:
                        disp_val = disp_pivot.sum(axis=1).iloc[j]
                        util_pct = disp_val / cap_val * 100
                        ax.text(
                            x=j,
                            y=cap_val,
                            s=f"{cap_val:.1f}\n({util_pct:.0f}%)",
                            ha="center",
                            va="bottom",
                            fontsize=subfontsize,
                        )
                    else:
                        ax.text(
                            x=j,
                            y=cap_val,
                            s=f"{cap_val:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=subfontsize,
                        )

        ax.axhline(0, color="black", lw=0.5)

    # Tick styling
    for ax in np.atleast_1d(axes):
        ax.tick_params(axis="y", labelsize=subfontsize)

    # Legend
    all_carriers = cap_df["carrier"].unique()
    legend_items = [c for c in legend_order if c in all_carriers]
    for c in all_carriers:
        if c not in legend_items:
            legend_items.append(c)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=carrier_color(c, tech_colors), label=c)
        for c in legend_items[::-1]
    ]

    legend = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(x_anchor + xpad, 0.03),
        ncol=ncol,
        fontsize=subfontsize,
        title="",
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
            "plot_ccs_installed_capacity",
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

    legend_order = plotting["legend_order"]

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

    capacity_by_scope, dispatch_by_scope = collect_ccs_data(
        file_map, scope_outputs.keys()
    )

    # =========================================================================
    # PLOT
    # =========================================================================

    for scope, output_path in scope_outputs.items():
        plot_ccs_scope(
            capacity_by_scope[scope],
            dispatch_by_scope[scope],
            scope,
            output_path,
            planning_horizons,
            lt_order,
            lt_order_nice_names,
            legend_order,
            tech_colors,
            figsize,
            dpi,
            font,
            fontsize,
            subfontsize,
        )
