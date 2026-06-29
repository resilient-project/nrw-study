#!/usr/bin/env python3
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-License-Identifier: MIT
"""
CO2 pipeline utilisation dashboard: utilisation duration curves + pipeline lengths.

One figure per scope (EU / Deutschland / NRW), n_planning_horizons columns.

Top row  — Utilisation Duration Curve (UDC) per pipeline category (all solid lines):
           DN700    = CO2 pipeline, onshore
           DN400    = CO2 pipeline short, onshore
           Offshore = all carriers with at least one offshore bus
           Each curve normalised by that category's own installed capacity.
           Each scenario's curves span the x-width of its bar below.
           Y-axis capped at 100 %.

Bottom row — Installed pipeline length [km] stacked by category (same colours).
             Lengths are read from the pre-computed CSVs produced by
             make_summary_nrw_study (identical source as plot_co2_pipeline_comparison).

Style follows plot_co2_pipeline_comparison.py (colours, font, figsize).
"""

import ast
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pypsa

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — aligned with plot_co2_pipeline_comparison.py
# ---------------------------------------------------------------------------

CO2_PIPELINE_CARRIERS = ["CO2 pipeline", "CO2 pipeline short"]

PIPE_CATS = ["DN700", "DN400", "Offshore"]

PIPE_CAT_LS = {
    "DN700":    "-",
    "DN400":    "-",
    "Offshore": "-",
}

# Maps (terrain, carrier) from CSV → display category
_CSV_TO_CAT = {
    ("offshore", "CO2 pipeline"):       "Offshore",
    ("offshore", "CO2 pipeline short"): "Offshore",
    ("onshore",  "CO2 pipeline"):       "DN700",
    ("onshore",  "CO2 pipeline short"): "DN400",
}

SCOPE_PREFIXES = {
    "eu":  None,
    "de":  ("DE",),
    "nrw": ("DEA",),
}
SCOPE_LABEL = {
    "eu":  "EU",
    "de":  "Deutschland",
    "nrw": "NRW",
}

# Region filter for CSV data (region column uses 2-char country codes or "DEA")
_SCOPE_REGION_FILTER = {
    "eu":  lambda r: True,
    "de":  lambda r: r.startswith("DE"),
    "nrw": lambda r: r == "DEA",
}

UDC_YLIM = 100.0


# ---------------------------------------------------------------------------
# Length data — identical loading as plot_co2_pipeline_comparison.py
# ---------------------------------------------------------------------------

def load_csvs(paths: list[str]) -> pd.DataFrame:
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df["name"] = path.split("/")[-3]
        m = re.search(r"(\d{4})\.csv$", path)
        df["planning_horizon"] = m.group(1) if m else "unknown"
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_bar_data(
    csv_paths: list[str],
    run_order: list[str],
    planning_horizons: list,
    scopes: list[str],
) -> dict[str, list]:
    """
    Aggregate pre-computed CSV length data into bar records per scope.
    Exact same source data as plot_co2_pipeline_comparison.
    """
    run_set = set(run_order)
    ph_set  = {str(ph) for ph in planning_horizons}
    bar: dict[str, list] = {s: [] for s in scopes}

    if not csv_paths:
        return bar

    data = load_csvs(csv_paths)
    data = data[data["name"].isin(run_set) & data["planning_horizon"].isin(ph_set)].copy()
    data["cat"] = data.apply(
        lambda row: _CSV_TO_CAT.get((row["terrain"], row["carrier"])), axis=1
    )
    data = data.dropna(subset=["cat"])

    for scope in scopes:
        region_ok  = _SCOPE_REGION_FILTER[scope]
        scope_data = data[data["region"].apply(region_ok)]
        for (run, ph, cat), grp in scope_data.groupby(["name", "planning_horizon", "cat"]):
            total_km = grp["length_km"].sum()
            if total_km > 1e-3:
                bar[scope].append({"run": run, "year": str(ph), "cat": cat, "value": total_km})

    return bar


# ---------------------------------------------------------------------------
# UDC data — from network files
# ---------------------------------------------------------------------------

def _filter_links(n: pypsa.Network, links: pd.DataFrame, scope: str) -> pd.DataFrame:
    prefixes = SCOPE_PREFIXES[scope]
    if prefixes is None:
        return links
    buses = n.buses.index[n.buses.index.str.startswith(prefixes)]
    return links[links.bus0.isin(buses) | links.bus1.isin(buses)]


def _active_co2_links(n: pypsa.Network) -> tuple[pd.DataFrame, str]:
    is_reversed = (
        n.links.index.str.contains("-reversed", na=False)
        | n.links.get("reversed", pd.Series(False, index=n.links.index)).fillna(False)
    )
    links = n.links[
        n.links.carrier.isin(CO2_PIPELINE_CARRIERS) & ~is_reversed
    ].copy()
    cap_col = "p_nom_opt" if links["p_nom_opt"].abs().sum() > 1e-3 else "p_nom"
    return links[links[cap_col] > 1e-3].copy(), cap_col


def _split_categories(sl: pd.DataFrame) -> dict[str, pd.DataFrame]:
    is_offshore = (
        sl["bus0"].str.contains("offshore", case=False, na=False)
        | sl["bus1"].str.contains("offshore", case=False, na=False)
    )
    onshore = sl[~is_offshore]
    return {
        "DN700":    onshore[onshore.carrier == "CO2 pipeline"],
        "DN400":    onshore[onshore.carrier == "CO2 pipeline short"],
        "Offshore": sl[is_offshore],
    }


def _build_udc(
    n: pypsa.Network,
    links: pd.DataFrame,
    cap_Mt_h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted utilisation duration curve using snapshot weights.

    Flow is the sum of |p0| for all forward links plus |p0| of their
    '-reversed' siblings (reverse-direction throughput).  Capacity uses
    only the forward links so each physical pipe is counted once.
    """
    _empty = (np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    if links.empty or cap_Mt_h <= 0:
        return _empty
    if not hasattr(n, "links_t") or "p0" not in n.links_t or n.links_t.p0.empty:
        return _empty

    p0_cols = n.links_t.p0.columns

    # Forward links that have time-series data
    valid_fwd = links.index.intersection(p0_cols)
    if valid_fwd.empty:
        return _empty

    # Reversed siblings: same index with '-reversed' suffix, if present
    rev_candidates = pd.Index([f"{idx}-reversed" for idx in valid_fwd])
    valid_rev = rev_candidates.intersection(p0_cols)

    # Build flow time-series: forward + reverse, aligned on snapshots
    flow_cols = valid_fwd.union(valid_rev)
    agg = n.links_t.p0[flow_cols].abs().sum(axis=1)

    # Snapshot weights — aligned by index (same snapshot index as p0)
    weights = n.snapshot_weightings.generators.reindex(agg.index)
    if weights.isna().any():
        raise ValueError(
            "Snapshot weightings could not be aligned with p0 index; "
            "check network snapshot consistency."
        )
    w_sum = float(weights.sum())

    order       = np.argsort(-agg.values)
    sorted_flow = agg.values[order] / 1e6
    sorted_w    = weights.values[order]

    cum_frac = np.concatenate([[0.0], np.cumsum(sorted_w) / w_sum])
    util_pct = np.concatenate([[sorted_flow[0]], sorted_flow]) / cap_Mt_h * 100.0
    return cum_frac, util_pct


def build_udc_data(
    network_paths: list[str],
    run_order: list[str],
    planning_horizons: list,
    scopes: list[str],
) -> dict[str, list]:
    run_set = set(run_order)
    ph_set  = {str(ph) for ph in planning_horizons}
    udc: dict[str, list] = {s: [] for s in scopes}
    seen: set[tuple] = set()

    for path in network_paths:
        run  = path.split("/")[-3]
        m    = re.search(r"(\d{4})\.nc$", path)
        year = m.group(1) if m else None

        if run not in run_set or year not in ph_set:
            continue
        if (run, year) in seen:
            logger.warning("Duplicate network run=%s year=%s; skipping %s", run, year, path)
            continue
        seen.add((run, year))

        try:
            n = pypsa.Network(path)
        except Exception as exc:
            logger.error("Failed to load %s: %s", path, exc)
            continue

        active, cap_col = _active_co2_links(n)
        if active.empty:
            logger.info("No active CO2 pipelines in %s / %s", run, year)
            continue

        for scope in scopes:
            sl   = _filter_links(n, active, scope)
            cats = _split_categories(sl)
            for cat, cl in cats.items():
                if cl.empty:
                    continue
                cap_Mt_h = cl[cap_col].sum() / 1e6
                if cap_Mt_h < 1e-6:
                    continue
                time_frac, util_pct = _build_udc(n, cl, cap_Mt_h)
                udc[scope].append({
                    "run": run, "year": year, "cat": cat,
                    "time_frac": time_frac, "util_pct": util_pct,
                })

    return udc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_empty(pdf: str, png: str, scope: str, figsize: tuple, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.text(
        0.5, 0.5, f"No CO2 pipeline data ({SCOPE_LABEL[scope]})",
        ha="center", va="center", transform=ax.transAxes, fontsize=12,
    )
    ax.set_axis_off()
    for out in (pdf, png):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(
    udc_data: list,
    bar_data: list,
    scope: str,
    output_pdf: str,
    output_png: str,
    planning_horizons: list,
    run_order: list[str],
    nice_names: dict,
    figsize: tuple,
    dpi: int,
    font: dict,
    fontsize: float,
    xticklabel_size: float,
    pipe_cat_colors: dict,
) -> None:
    if not udc_data and not bar_data:
        _save_empty(output_pdf, output_png, scope, figsize, dpi)
        return

    n_ph      = len(planning_horizons)
    n_runs    = len(run_order)
    bar_width = 0.8

    fig, axes = plt.subplots(
        2, n_ph,
        figsize=figsize,
        dpi=dpi,
        sharey="row",
        tight_layout=True,
        squeeze=False,
    )
    plt.rc("font", **font)

    bar_df = pd.DataFrame(bar_data) if bar_data else pd.DataFrame()

    bar_ymax = 1.0
    if not bar_df.empty:
        bar_ymax = float(bar_df.groupby(["year", "run"])["value"].sum().max()) * 1.15
        bar_ymax = max(bar_ymax, 1.0)

    bar_hw = bar_width / 2

    for i, ph in enumerate(planning_horizons):
        ax_udc = axes[0, i]
        ax_len = axes[1, i]
        ph_str = str(ph)

        # ---- top: UDC ----
        ph_udc = [d for d in udc_data if d["year"] == ph_str]
        for run in run_order:
            j = run_order.index(run)
            for cat in PIPE_CATS:
                entries = [d for d in ph_udc if d["run"] == run and d["cat"] == cat]
                if not entries:
                    continue
                d = entries[0]
                x_mapped = j - bar_hw + d["time_frac"] * bar_width
                ax_udc.step(
                    x_mapped, d["util_pct"],
                    where="post",
                    color=pipe_cat_colors[cat],
                    ls=PIPE_CAT_LS[cat],
                    lw=1.2,
                )

        ax_udc.set_xlim(-0.5, n_runs - 0.5)
        ax_udc.set_ylim(0, UDC_YLIM)
        ax_udc.set_yticks([0, 20, 40, 60, 80, int(UDC_YLIM)])
        ax_udc.set_title("", fontsize=fontsize)
        ax_udc.set_xticks(range(n_runs))
        ax_udc.tick_params(labelbottom=False)
        ax_udc.tick_params(axis="y", labelsize=fontsize)
        ax_udc.axhline(0, color="black", lw=0.5)
        ax_udc.grid(False)
        ax_udc.spines["top"].set_visible(False)
        ax_udc.spines["right"].set_visible(False)
        ax_udc.spines["left"].set_linewidth(0.5)
        ax_udc.spines["bottom"].set_linewidth(0.5)
        if i == 0:
            ax_udc.set_ylabel(f"Auslastung (%)", fontsize=fontsize)
        else:
            ax_udc.yaxis.set_visible(False)

        # ---- bottom: pipeline lengths ----
        if bar_df.empty:
            pivot = pd.DataFrame(index=pd.Index(run_order, name="run"))
        else:
            ph_bar = bar_df[bar_df.year == ph_str]
            if ph_bar.empty:
                pivot = pd.DataFrame(index=pd.Index(run_order, name="run"))
            else:
                pivot = (
                    ph_bar.pivot_table(
                        index="run", columns="cat", values="value", aggfunc="sum"
                    )
                    .reindex(run_order)
                    .fillna(0)
                )

        col_order = [c for c in PIPE_CATS if c in pivot.columns]
        if col_order:
            pivot = pivot[col_order]

        run_labels = [nice_names.get(r, r) for r in pivot.index]

        if col_order:
            pivot.plot(
                kind="bar", stacked=True, ax=ax_len, width=bar_width,
                color=[pipe_cat_colors[c] for c in col_order],
                edgecolor="none",
                legend=False,
            )
            totals = pivot[pivot > 0].sum(axis=1)
            for j, total in enumerate(totals):
                if total > 0:
                    ax_len.text(
                        j, total + bar_ymax * 0.04, f"{total:.0f}",
                        ha="center", va="bottom", fontsize=fontsize,
                    )

        ax_len.set_ylim(0, bar_ymax)
        ax_len.set_xlabel(ph_str, fontsize=fontsize)
        ax_len.set_xticklabels(run_labels, rotation=0, fontsize=xticklabel_size)
        ax_len.tick_params(axis="y", labelsize=fontsize)
        ax_len.axhline(0, color="black", lw=0.5)
        ax_len.grid(False)
        ax_len.spines["top"].set_visible(False)
        ax_len.spines["right"].set_visible(False)
        ax_len.spines["left"].set_linewidth(0.5)
        ax_len.spines["bottom"].set_linewidth(0.5)
        if i == 0:
            ax_len.set_ylabel(f"Pipelinelängen (km)", fontsize=fontsize)
        else:
            ax_len.yaxis.set_visible(False)

    fig.align_ylabels([axes[0, 0], axes[1, 0]])

    handles = [
        Patch(facecolor=pipe_cat_colors[c], label=c)
        for c in PIPE_CATS[::-1]
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(PIPE_CATS),
        fontsize=fontsize,
        frameon=False,
        handlelength=0.8,
        handleheight=0.8,
    )

    fig.subplots_adjust(wspace=0.05, hspace=0.5)

    for out in (output_pdf, output_png):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_pdf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake
        snakemake = mock_snakemake(
            "plot_co2_pipeline_utilisation_dashboard",
            configfiles=["config/config.nrw.yaml"],
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    config   = snakemake.config
    plotting = snakemake.params.plotting_fig

    font              = plotting["font"]
    fontsize          = font["size"]
    xticklabel_size   = plotting.get("xticklabel_size", fontsize)
    figsize           = ast.literal_eval(plotting["figsize"])
    dpi               = plotting["dpi"]
    run_order         = plotting["run_order"]
    nice_names        = plotting["nice_names"]
    pipe_cat_colors   = plotting["pipe_cat_colors"]
    planning_horizons = config["scenario"]["planning_horizons"]

    scopes  = ["eu", "de", "nrw"]
    outputs = {
        "eu":  (snakemake.output.eu,  snakemake.output.eu_png),
        "de":  (snakemake.output.de,  snakemake.output.de_png),
        "nrw": (snakemake.output.nrw, snakemake.output.nrw_png),
    }

    bar_by_scope = build_bar_data(
        list(snakemake.input.csvs),
        run_order,
        planning_horizons,
        scopes,
    )

    udc_by_scope = build_udc_data(
        list(snakemake.input.networks),
        run_order,
        planning_horizons,
        scopes,
    )

    for scope, (out_pdf, out_png) in outputs.items():
        plot_dashboard(
            udc_by_scope[scope],
            bar_by_scope[scope],
            scope,
            out_pdf,
            out_png,
            planning_horizons,
            run_order,
            nice_names,
            figsize,
            dpi,
            font,
            fontsize,
            xticklabel_size,
            pipe_cat_colors,
        )
