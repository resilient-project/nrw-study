#!/usr/bin/env python3
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Dashboard: CO₂ CCS cost decomposition – EU scope, 1 row × n_planning_horizons cols.

Each column shows:
  - Bars:    Transport cost [€/tCO₂ captured]
             = Σ capital_cost × max(p_nom, p_nom_opt) for all CO₂ pipeline links
               / total annual CO₂ captured by all CC + DAC technologies
  - Circles: Capture cost [€/tCO₂] for DAC and (optionally) process emissions CC.

Toggle `show_process_emissions_cc` in plotting config to include/exclude process CC.
Note: process emissions CC costs in this model are CAPEX-only (waste heat assumed
free, electricity for compression not modelled) and underestimate real costs by ~2-6×.
"""

import ast
import logging
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CAPACITY_MW       = 1e-5  # per-link solver-noise threshold [MW]
MIN_CARRIER_TOTAL_MW  = 1.0   # minimum total installed capacity per carrier; below this it's solver noise
MIN_DISPATCH_T        = 1.0   # minimum annual capture [tCO₂] to avoid division-by-zero

CC_CARRIERS = [
    "SMR CC",
    "DAC",
    "gas for industry CC",
    "process emissions CC",
    "solid biomass for industry CC",
    "urban central gas CHP CC",
    "urban central solid biomass CHP CC",
]

# Carriers whose buses carry CO₂ mass (not energy) → skip in energy-cost calc
CO2_LIKE_CARRIERS = frozenset({
    "co2", "co2 stored", "co2 sequestered", "co2 atmosphere",
    "process emissions",
})

# Pipeline link carriers to include in transport CAPEX
CO2_PIPE_CARRIERS = frozenset({"CO2 pipeline", "CO2 pipeline short"})

# Technologies shown as capture-cost circles
CAPTURE_CARRIERS = ["DAC", "process emissions CC"]

# Default cost-component colours (overridable via config)
COST_COLORS_DEFAULT = {
    "transport": "#457b9d",
}

# Fallback colours for capture circles if not in tech_colors
_CAPTURE_COLORS_DEFAULT = {
    "DAC":                  "#2a9d8f",
    "process emissions CC": "#e76f51",
}


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def _cc_carriers(n: pypsa.Network) -> list[str]:
    present = set(n.links.carrier.unique())
    found = [c for c in CC_CARRIERS if c in present]
    for c in present - set(found):
        if (c.endswith(" CC") or c == "DAC" or "capture" in c.lower()) \
                and "pipeline" not in c.lower():
            found.append(c)
    return found


def _dispatch_Mt(n: pypsa.Network, links: pd.DataFrame) -> pd.Series:
    """Annual CO₂ captured [MtCO₂/a] per link — flow into 'co2 stored' buses."""
    result = pd.Series(0.0, index=links.index)
    if links.empty:
        return result
    weights = n.snapshot_weightings.generators
    for i in range(1, 5):
        bus_col, p_col = f"bus{i}", f"p{i}"
        if bus_col not in links.columns or p_col not in n.links_t:
            continue
        mask = links[bus_col].fillna("").str.contains("co2 stored", case=False)
        valid = links[mask].index.intersection(n.links_t[p_col].columns)
        if valid.empty:
            continue
        result.loc[valid] += (
            n.links_t[p_col][valid].abs().multiply(weights, axis=0).sum()
        )
    return result / 1e6


def _co2_buses(n: pypsa.Network) -> frozenset:
    return frozenset(n.buses.index[n.buses.carrier.isin(CO2_LIKE_CARRIERS)])


def _annual_capex(links: pd.DataFrame, cap_col: str) -> pd.Series:
    return links[cap_col].fillna(0.0) * links["capital_cost"].fillna(0.0)


def _annual_opex(n: pypsa.Network, links: pd.DataFrame) -> pd.Series:
    result = pd.Series(0.0, index=links.index)
    if "p0" not in n.links_t or n.links_t.p0.empty:
        return result
    weights = n.snapshot_weightings.generators
    valid = links.index.intersection(n.links_t.p0.columns)
    if valid.empty:
        return result
    mc = links.loc[valid, "marginal_cost"].fillna(0.0)
    dispatch = n.links_t.p0[valid].abs().multiply(weights, axis=0).sum()
    result.loc[valid] = mc * dispatch
    return result


def _annual_energy_cost(n: pypsa.Network, links: pd.DataFrame) -> pd.Series:
    """
    Net annual fuel/energy cost [EUR/a].
    Formula: Σ_t [ p0×mp0 − Σ_{i≥1, non-CO₂} p_i×mp_i ] × w_t
    Positive = net cost; negative = net co-product revenue.
    """
    result = pd.Series(0.0, index=links.index)
    if not hasattr(n.buses_t, "marginal_price") or n.buses_t.marginal_price.empty:
        return result

    mp = n.buses_t.marginal_price
    weights = n.snapshot_weightings.generators
    co2_b = _co2_buses(n)

    bus_cols = ["bus0"] + [f"bus{i}" for i in range(1, 5)]
    p_cols   = ["p0"]   + [f"p{i}"   for i in range(1, 5)]
    signs    = [+1]     + [-1] * 4

    for bus_col, p_col, sign in zip(bus_cols, p_cols, signs):
        if bus_col not in links.columns or p_col not in n.links_t:
            continue
        p_ts = n.links_t[p_col]
        bus_series = links[bus_col].dropna()
        for bus_name, grp in bus_series.groupby(bus_series):
            if not bus_name or bus_name in co2_b:
                continue
            if bus_name not in mp.columns:
                continue
            valid = grp.index.intersection(p_ts.columns)
            if valid.empty:
                continue
            contrib = (
                p_ts[valid]
                .multiply(mp[bus_name], axis=0)
                .multiply(weights, axis=0)
                .sum()
            ) * sign
            result.loc[valid] += contrib

    return result


# ---------------------------------------------------------------------------
# Per-network records
# ---------------------------------------------------------------------------

def _pipeline_record(n: pypsa.Network, run: str, year: str,
                     total_captured_t: float) -> dict | None:
    """
    Transport cost per tonne CO₂ captured [EUR/tCO₂].

    Numerator:   Σ capital_cost × max(p_nom, p_nom_opt)  for all CO₂ pipeline links
    Denominator: total annual CO₂ captured across all CC + DAC technologies
    """
    pipes = n.links[
        n.links.carrier.isin(CO2_PIPE_CARRIERS)
        & ~n.links.index.str.contains("-reversed", na=False)
    ].copy()
    if pipes.empty:
        return None

    cap = pipes[["p_nom", "p_nom_opt"]].fillna(0.0).max(axis=1)
    total_capex = float((cap * pipes["capital_cost"].fillna(0.0)).sum())

    cost_per_t = total_capex / total_captured_t if total_captured_t > 0 else np.nan
    return {
        "planning_horizon": year,
        "name": run,
        "total_capex_eur": total_capex,
        "total_captured_t": total_captured_t,
        "cost_per_t": cost_per_t,
    }


def _capture_records(n: pypsa.Network, run: str, year: str) -> list[dict]:
    """
    Cost per tonne CO₂ [EUR/tCO₂] for each technology in CAPTURE_CARRIERS.
    Includes CAPEX, VOM, and net energy cost at shadow prices.
    """
    records = []
    for carrier in CAPTURE_CARRIERS:
        links = n.links[
            (n.links.carrier == carrier)
            & ~n.links.index.str.contains("-reversed", na=False)
        ].copy()
        if links.empty:
            continue

        cap_col = "p_nom_opt" if links["p_nom_opt"].abs().sum() > MIN_CAPACITY_MW else "p_nom"
        links = links[links[cap_col].abs() >= MIN_CAPACITY_MW].copy()
        if links.empty or links[cap_col].sum() < MIN_CARRIER_TOTAL_MW:
            continue

        disp_t = max(float(_dispatch_Mt(n, links).sum()) * 1e6, MIN_DISPATCH_T)
        capex  = float(_annual_capex(links, cap_col).sum())
        opex   = float(_annual_opex(n, links).sum())
        energy = float(_annual_energy_cost(n, links).sum())

        records.append({
            "planning_horizon": year,
            "name":       run,
            "carrier":    carrier,
            "disp_t":     disp_t,
            "capex_per_t":  capex  / disp_t,
            "opex_per_t":   opex   / disp_t,
            "energy_per_t": energy / disp_t,
            "total_per_t":  (capex + opex + energy) / disp_t,
        })
    return records


# ---------------------------------------------------------------------------
# Main data builder
# ---------------------------------------------------------------------------

def build_tables(
    paths: list[str],
    run_order: list[str],
    planning_horizons: list,
) -> tuple:
    """
    Load networks and return:
      pipe_df    : DataFrame – transport cost per (run, year)
      capture_df : DataFrame – capture cost per (run, year, carrier)
    """
    run_set = set(run_order)
    ph_set  = {str(ph) for ph in planning_horizons}

    pipe_records:    list[dict] = []
    capture_records: list[dict] = []
    seen: set[tuple] = set()

    for path in paths:
        run  = path.split("/")[-3]
        m    = re.search(r"(\d{4})\.nc$", path)
        year = m.group(1) if m else None
        if run not in run_set or year not in ph_set:
            continue
        if (run, year) in seen:
            logger.warning("Duplicate network run=%s year=%s; skipping %s", run, year, path)
            continue
        seen.add((run, year))

        logger.info("Loading %s %s from %s", run, year, path)
        n = pypsa.Network(path)

        # Total CO₂ captured across all CC + DAC (transport denominator)
        all_carriers = _cc_carriers(n)
        all_cc = n.links[
            n.links.carrier.isin(all_carriers)
            & ~n.links.index.str.contains("-reversed", na=False)
        ].copy()
        total_captured_t = max(float(_dispatch_Mt(n, all_cc).sum()) * 1e6, 0.0)

        rec = _pipeline_record(n, run, year, total_captured_t)
        if rec:
            pipe_records.append(rec)

        capture_records.extend(_capture_records(n, run, year))

    return (
        pd.DataFrame(pipe_records)    if pipe_records    else pd.DataFrame(),
        pd.DataFrame(capture_records) if capture_records else pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _style_ax(ax, fontsize: float) -> None:
    ax.grid(False)
    ax.axhline(0, color="black", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(axis="y", labelsize=fontsize)


def _plot_col(
    ax,
    pipe_df: pd.DataFrame,
    capture_df: pd.DataFrame,
    ph,
    run_order: list[str],
    nice_names: dict,
    cost_colors: dict,
    tech_colors: dict,
    show_process_cc: bool,
    fontsize: float,
    is_first_col: bool,
) -> float:
    """One column: transport bars + capture circles on a shared axis."""
    ph_str = str(ph)
    ax.set_xlabel(ph_str, fontsize=fontsize)
    _style_ax(ax, fontsize)

    sub_pipe = pipe_df[pipe_df.planning_horizon == ph_str].set_index("name") \
        if not pipe_df.empty else pd.DataFrame()
    sub_cap = capture_df[capture_df.planning_horizon == ph_str].set_index(["name", "carrier"]) \
        if not capture_df.empty else pd.DataFrame()

    carriers_shown = ["DAC"] + (["process emissions CC"] if show_process_cc else [])
    x_offset = {
        "DAC":                  -0.20 if show_process_cc else 0.0,
        "process emissions CC": +0.20,
    }

    bar_w = 0.8
    ymax  = 0.0

    for j, run in enumerate(run_order):
        # Transport bar
        if not sub_pipe.empty and run in sub_pipe.index:
            cpt = float(sub_pipe.loc[run, "cost_per_t"])
            if not np.isnan(cpt) and cpt > 0:
                ax.bar(j, cpt, bar_w, color=cost_colors["transport"],
                       edgecolor="none", zorder=3)
                ax.text(j, cpt + 0.5, f"{round(cpt)}",
                        ha="center", va="bottom", fontsize=fontsize)
                ymax = max(ymax, cpt)

        # Capture circles
        for carrier in carriers_shown:
            key = (run, carrier)
            if sub_cap.empty or key not in sub_cap.index:
                continue
            row   = sub_cap.loc[key]
            total = float(row["capex_per_t"] + row["opex_per_t"] + row["energy_per_t"])
            if np.isnan(total):
                continue
            color = tech_colors.get(carrier, _CAPTURE_COLORS_DEFAULT.get(carrier, "#999999"))
            ax.scatter(j + x_offset[carrier], total, s=90, color=color,
                       zorder=5, edgecolors="none", clip_on=False)
            ax.text(j + x_offset[carrier], total + 4, f"{round(total)}",
                    ha="center", va="bottom", fontsize=fontsize - 1)
            ymax = max(ymax, total)

    run_labels = [nice_names.get(r, r) for r in run_order]
    ax.set_xticks(range(len(run_order)))
    ax.set_xticklabels(run_labels, rotation=0, fontsize=fontsize)
    ax.set_xlim(-0.6, len(run_order) - 0.4)

    if is_first_col:
        ax.set_ylabel("Kosten (€/tCO₂)", fontsize=fontsize)
        ax.yaxis.set_label_coords(-0.20, 0.5)
    else:
        ax.yaxis.set_visible(False)

    return ymax


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_dashboard(
    pipe_df: pd.DataFrame,
    capture_df: pd.DataFrame,
    output_pdf: str,
    output_png: str,
    planning_horizons: list,
    run_order: list[str],
    nice_names: dict,
    cost_colors: dict,
    tech_colors: dict,
    show_process_cc: bool,
    figsize: tuple,
    dpi: int,
    font: dict,
    fontsize: float,
) -> None:

    n_ph = len(planning_horizons)

    fig, axes = plt.subplots(
        1, n_ph,
        figsize=figsize,
        dpi=dpi,
        sharey=True,
        squeeze=False,
    )
    plt.rc("font", **font)

    ymax = 0.0
    for i, ph in enumerate(planning_horizons):
        ymax = max(ymax, _plot_col(
            axes[0, i], pipe_df, capture_df, ph,
            run_order, nice_names,
            cost_colors, tech_colors,
            show_process_cc, fontsize, is_first_col=(i == 0),
        ))

    axes[0, 0].set_ylim(0, max(ymax * 1.25, 10))

    # Legend
    dac_color = tech_colors.get("DAC", _CAPTURE_COLORS_DEFAULT["DAC"])
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=cost_colors["transport"],
                       label="Transport CAPEX"),
        plt.scatter([], [], s=90, color=dac_color, label="DAC"),
    ]
    if show_process_cc:
        pcc_color = tech_colors.get(
            "process emissions CC", _CAPTURE_COLORS_DEFAULT["process emissions CC"]
        )
        legend_handles.append(
            plt.scatter([], [], s=90, color=pcc_color, label="Prozessem. CC")
        )

    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.0),
        ncol=len(legend_handles),
        fontsize=fontsize,
        frameon=False,
        handlelength=0.8,
        handleheight=0.8,
    )

    fig.subplots_adjust(
        left=0.14, right=0.97, top=0.97, bottom=0.22,
        wspace=0.10,
    )
    fig.savefig(output_pdf, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_pdf)


def _save_empty(pdf: str, png: str, figsize: tuple, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.text(0.5, 0.5, "Keine CC-Daten", ha="center", va="center",
            transform=ax.transAxes)
    ax.set_axis_off()
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake
        snakemake = mock_snakemake(
            "plot_cc_cost_dashboard",
            configfiles=["config/config.nrw.yaml"],
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    config   = snakemake.config
    plotting = snakemake.params.plotting_fig

    font             = plotting["font"]
    fontsize         = font["size"]
    figsize          = ast.literal_eval(plotting["figsize"])
    dpi              = plotting["dpi"]
    run_order        = plotting["run_order"]
    nice_names       = plotting["nice_names"]
    planning_horizons = config["scenario"]["planning_horizons"]
    show_process_cc  = plotting.get("show_process_emissions_cc", False)

    raw_cost_colors = plotting.get("cost_colors", {})
    cost_colors = {**COST_COLORS_DEFAULT, **raw_cost_colors}
    tech_colors = config["plotting"].get("tech_colors", {})

    pipe_df, capture_df = build_tables(
        list(snakemake.input.networks),
        run_order,
        planning_horizons,
    )

    if pipe_df.empty and capture_df.empty:
        _save_empty(snakemake.output.pdf, snakemake.output.png, figsize, dpi)
    else:
        plot_dashboard(
            pipe_df,
            capture_df,
            snakemake.output.pdf,
            snakemake.output.png,
            planning_horizons,
            run_order,
            nice_names,
            cost_colors,
            tech_colors,
            show_process_cc,
            figsize,
            dpi,
            font,
            fontsize,
        )
