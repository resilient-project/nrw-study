#!/usr/bin/env python3
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Plot installed CO₂ capture capacity and utilisation for all CC technologies
across NRW scenarios.

Layout: one figure per scope (EU / Deutschland / NRW), each with
2 rows × n planning horizons columns.  Top row: utilisation [%] per
individual plant (scatter dots).  Bottom row: installed full-load
capacity [MtCO₂/a] as stacked bars.
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
from scipy.stats import gaussian_kde

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CC_CARRIERS = [
    "SMR CC",
    "DAC",
    "gas for industry CC",
    "process emissions CC",
    "solid biomass for industry CC",
    "urban central gas CHP CC",
    "urban central solid biomass CHP CC",
]

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

# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _cc_carriers(n: pypsa.Network) -> list[str]:
    present = set(n.links.carrier.unique())
    found = [c for c in CC_CARRIERS if c in present]
    for c in present - set(found):
        if (c.endswith(" CC") or c == "DAC" or "capture" in c.lower()) \
                and "pipeline" not in c.lower():
            found.append(c)
    return found


def _capacity_Mt(links: pd.DataFrame, cap_col: str) -> pd.Series:
    """Full-load CO₂ capture capacity [MtCO₂/a] per link."""
    factor = pd.Series(0.0, index=links.index)
    for i in range(1, 5):
        bus_col = f"bus{i}"
        eff_col = "efficiency" if i == 1 else f"efficiency{i}"
        if bus_col not in links.columns or eff_col not in links.columns:
            continue
        mask = links[bus_col].fillna("").str.contains("co2 stored", case=False)
        factor[mask] += links.loc[mask, eff_col].abs()
    return links[cap_col].fillna(0.0) * factor * 8760.0 / 1e6


def _dispatch_Mt(n: pypsa.Network, links: pd.DataFrame) -> pd.Series:
    """Actual annual CO₂ captured [MtCO₂/a] per link from dispatch time series."""
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
        result.loc[valid] += n.links_t[p_col][valid].abs().multiply(weights, axis=0).sum()
    return result / 1e6


def _filter_scope(n: pypsa.Network, links: pd.DataFrame, scope: str) -> pd.DataFrame:
    prefixes = SCOPE_PREFIXES[scope]
    if prefixes is None:
        return links
    # Only bus0 (the fuel/input bus) determines the geographic location of a CC link.
    # Including bus1 would pull in links whose output buses cross region boundaries
    # (e.g. a DAC plant outside NRW whose heat input bus is labelled as NRW).
    buses = n.buses.index[n.buses.index.str.startswith(prefixes)]
    return links[links.bus0.isin(buses)]


def build_tables(
    paths: list[str],
    run_order: list[str],
    planning_horizons: list,
    scopes: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load networks; return (cap_by_scope, disp_by_scope) dicts of DataFrames."""
    run_set = set(run_order)
    ph_set = {str(ph) for ph in planning_horizons}
    cap: dict[str, list] = {s: [] for s in scopes}
    disp: dict[str, list] = {s: [] for s in scopes}

    for path in paths:
        run = path.split("/")[-3]
        m = re.search(r"(\d{4})\.nc$", path)
        year = m.group(1) if m else None
        if run not in run_set or year not in ph_set:
            continue
        n = pypsa.Network(path)
        carriers = _cc_carriers(n)
        links = n.links[
            n.links.carrier.isin(carriers)
            & ~n.links.index.str.contains("-reversed", na=False)
        ].copy()
        if links.empty:
            continue

        cap_col = "p_nom_opt" if links["p_nom_opt"].abs().sum() > 1e-3 else "p_nom"
        links["_cap"]  = _capacity_Mt(links, cap_col)
        links["_disp"] = _dispatch_Mt(n, links)

        # Drop solver-noise links: real CC installations are orders of magnitude
        # larger than the ~1e-4 MW artifacts the solver leaves at every extendable link.
        links = links[links["_cap"] > 1e-3]  # < 1 ktCO₂/a → noise
        if links.empty:
            continue

        for scope in scopes:
            sl = _filter_scope(n, links, scope)
            if sl.empty:
                continue
            for link_id, lnk in sl[["carrier", "_cap", "_disp"]].iterrows():
                base = {"planning_horizon": year, "name": run, "carrier": lnk["carrier"], "link": link_id}
                cap[scope].append({**base, "value": lnk["_cap"]})
                disp[scope].append({**base, "value": lnk["_disp"]})

    return (
        {s: pd.DataFrame(cap[s])  for s in scopes},
        {s: pd.DataFrame(disp[s]) for s in scopes},
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _color(carrier: str, tech_colors: dict) -> str:
    return tech_colors.get(carrier, tech_colors.get("CC", "#999999"))


def plot_scope(
    cap_df: pd.DataFrame,
    disp_df: pd.DataFrame,
    scope: str,
    output_pdf: str,
    output_png: str,
    planning_horizons: list,
    run_order: list[str],
    nice_names: dict,
    legend_order: list[str],
    tech_colors: dict,
    carrier_german: dict,
    figsize: tuple,
    dpi: int,
    font: dict,
    fontsize: float,
) -> None:
    if cap_df.empty:
        _save_empty(output_pdf, output_png, scope, figsize, dpi)
        return

    n_ph = len(planning_horizons)
    fig, axes = plt.subplots(
        2, n_ph,
        figsize=figsize, dpi=dpi,
        sharey="row",
        sharex="col",
        gridspec_kw={"height_ratios": [1, 1]},
        squeeze=False,
    )
    plt.rc("font", **font)

    # Carrier-level totals for stacked bars; raw per-link data for scatter dots
    cap_agg = cap_df.groupby(["planning_horizon", "name", "carrier"])["value"].sum().reset_index()
    ymax = max(cap_agg.groupby(["planning_horizon", "name"])["value"].sum().max(), 1.0)
    cap_max = max(cap_df["value"].max(), 1.0)

    for i, ph in enumerate(planning_horizons):
        ax_dot = axes[0, i]   # top row: utilisation scatter
        ax_bar = axes[1, i]   # bottom row: capacity bars
        ph_str = str(ph)

        sub = cap_agg[cap_agg.planning_horizon == ph_str]
        if not sub.empty:
            pivot = sub.pivot(index="name", columns="carrier", values="value").fillna(0)
        else:
            pivot = pd.DataFrame(index=pd.Index(run_order, name="name"))

        pivot = pivot.reindex(run_order).fillna(0)
        col_order = [c for c in legend_order if c in pivot.columns] \
                  + [c for c in pivot.columns if c not in legend_order]
        if col_order:
            pivot = pivot[col_order]

        run_labels = [nice_names.get(r, r) for r in pivot.index]

        # --- bottom: stacked capacity bars ---
        if col_order:
            pivot.plot(
                kind="bar", stacked=True, ax=ax_bar, width=0.8,
                color=[_color(c, tech_colors) for c in pivot.columns],
                legend=False,
            )
        ax_bar.set_xlabel(ph_str, fontsize=fontsize)
        ax_bar.set_ylim(0, ymax * 1.2)
        ax_bar.set_xticklabels(run_labels, rotation=0, fontsize=fontsize)
        ax_bar.tick_params(axis="y", labelsize=fontsize)
        ax_bar.grid(False)
        ax_bar.axhline(0, color="black", lw=0.5)
        for spine in ax_bar.spines.values():
            spine.set_linewidth(0.5)
        if i == 0:
            ax_bar.set_ylabel("CO₂-Abscheideleistung (Mtpa)", fontsize=fontsize)
            ax_bar.yaxis.set_label_coords(-0.20, 0.5)
        else:
            ax_bar.yaxis.set_visible(False)

        # --- top: utilisation scatter ---
        ax_dot.set_ylim(0, 110)
        ax_dot.axhline(0, color="black", lw=0.5)
        ax_dot.tick_params(labelbottom=False)
        ax_dot.tick_params(axis="y", labelsize=fontsize)
        ax_dot.grid(False)
        for spine in ax_dot.spines.values():
            spine.set_linewidth(0.5)
        if i == 0:
            ax_dot.set_ylabel("Auslastung (%)", fontsize=fontsize)
            ax_dot.yaxis.set_label_coords(-0.20, 0.5)
        else:
            ax_dot.yaxis.set_visible(False)

        # --- per-link dots + bar totals ---
        for j, run in enumerate(pivot.index):
            cap_val = float(pivot.loc[run].sum())
            if cap_val <= 0:
                continue

            ax_bar.text(
                j, cap_val + ymax * 0.02, f"{round(cap_val)}",
                ha="center", va="bottom", fontsize=fontsize,
            )

            cap_links = cap_df[(cap_df.planning_horizon == ph_str) & (cap_df.name == run)]
            if cap_links.empty or disp_df.empty:
                continue
            disp_links = disp_df[(disp_df.planning_horizon == ph_str) & (disp_df.name == run)]
            disp_by_link = (
                disp_links.set_index("link")["value"]
                if not disp_links.empty else pd.Series(dtype=float)
            )
            # Collect valid points for this run × year
            pts_util, pts_size, pts_color, pts_carrier = [], [], [], []
            for row in cap_links.itertuples():
                cap_c = row.value
                if cap_c <= 0:
                    continue
                disp_c = float(disp_by_link.get(row.link, 0.0))
                pts_util.append(disp_c / cap_c * 100)
                pts_size.append(max(cap_c / cap_max * 400, 20))
                pts_color.append(_color(row.carrier, tech_colors))
                pts_carrier.append(row.carrier)

            if not pts_util:
                continue

            utils = np.array(pts_util)
            rng = np.random.default_rng(seed=hash(f"{ph_str}{run}") & 0xFFFFFFFF)

            # Violin-style jitter: wider where many points cluster, narrower where few
            max_jitter = 0.4
            if len(utils) > 1 and utils.std() > 1e-6:
                dens = gaussian_kde(utils)(utils)
                widths = dens / dens.max() * max_jitter
            else:
                widths = np.zeros(len(utils))

            jitters = np.array([rng.uniform(-w, w) for w in widths])
            ax_dot.scatter(
                j + jitters, utils,
                s=pts_size,
                c=pts_color,
                alpha=0.35,
                zorder=5,
                clip_on=False,
                edgecolors="none",
                linewidths=0,
            )

            # Median per carrier: circle the actual dot closest to the median,
            # using its jitter x-offset so the ring sits on top of the dot.
            carriers_arr = np.array(pts_carrier)
            sizes_arr = np.array(pts_size)
            for carrier in dict.fromkeys(pts_carrier):
                mask = carriers_arr == carrier
                grp_utils = utils[mask]
                grp_jitters = jitters[mask]
                grp_sizes = sizes_arr[mask]
                med = float(np.median(grp_utils))
                closest = int(np.argmin(np.abs(grp_utils - med)))
                med_x = float(j + grp_jitters[closest])
                ring_size = max(float(grp_sizes[closest]) * 1.5, 60)
                color = _color(carrier, tech_colors)
                ax_dot.scatter(
                    med_x, med,
                    s=80,
                    marker="X",
                    facecolors="white",
                    edgecolors=color,
                    linewidths=0.8,
                    zorder=8,
                    clip_on=False,
                )
                ax_dot.annotate(
                    f"{round(med)}%",
                    xy=(min(med_x + 0.25, j + 0.45), med),
                    fontsize=fontsize,
                    va="center",
                    color=color,
                    zorder=9,
                    clip_on=False,
                )

    # Always show all legend_order carriers so the legend is identical across scopes.
    # Extra carriers not in legend_order are appended at the end.
    all_carriers = set(cap_df.carrier.unique())
    leg_carriers = list(legend_order) \
                 + [c for c in all_carriers if c not in set(legend_order)]
    carrier_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=_color(c, tech_colors),
                       label=carrier_german.get(c, c))
        for c in leg_carriers[::-1]
    ]
    fig.legend(
        handles=carrier_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.0),
        ncol=3,
        fontsize=fontsize,
        frameon=False,
        handlelength=0.8,
        handleheight=0.8,
    )
    ref1 = max(1, round(cap_max / 4))
    ref2 = 2 * ref1
    circle_handles = [
        plt.scatter([], [], s=max(ref / cap_max * 400, 20), color="dimgrey", alpha=0.6,
                    label=f"{ref} MtCO₂/a")
        for ref in [ref1, ref2]
    ]
    fig.legend(
        handles=circle_handles,
        loc="upper right",
        bbox_to_anchor=(0.84, 0.0),
        ncol=1,
        fontsize=fontsize,
        frameon=False,
    )

    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.22, wspace=0.10, hspace=0.12)
    fig.savefig(output_pdf, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_pdf)


def _save_empty(pdf: str, png: str, scope: str, figsize: tuple, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.text(
        0.5, 0.5, f"Keine CC-Daten ({SCOPE_LABEL[scope]})",
        ha="center", va="center", transform=ax.transAxes,
    )
    ax.set_axis_off()
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake
        snakemake = mock_snakemake(
            "plot_cc_capacity_utilisation",
            configfiles=["config/config.nrw.yaml"],
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    config   = snakemake.config
    plotting = snakemake.params.plotting_fig

    tech_colors      = config["plotting"]["tech_colors"]
    carrier_german   = snakemake.params.carrier_german
    font             = plotting["font"]
    fontsize         = font["size"]
    figsize          = ast.literal_eval(plotting["figsize"])
    dpi              = plotting["dpi"]
    run_order        = plotting["run_order"]
    nice_names       = plotting["nice_names"]
    legend_order     = plotting["legend_order"]
    planning_horizons = config["scenario"]["planning_horizons"]

    scopes = ["eu", "de", "nrw"]
    outputs = {
        "eu":  (snakemake.output.eu,  snakemake.output.eu_png),
        "de":  (snakemake.output.de,  snakemake.output.de_png),
        "nrw": (snakemake.output.nrw, snakemake.output.nrw_png),
    }

    cap_by_scope, disp_by_scope = build_tables(
        list(snakemake.input.networks),
        run_order,
        planning_horizons,
        scopes,
    )

    for scope in scopes:
        pdf, png = outputs[scope]
        plot_scope(
            cap_by_scope[scope],
            disp_by_scope[scope],
            scope,
            pdf,
            png,
            planning_horizons,
            run_order,
            nice_names,
            legend_order,
            tech_colors,
            carrier_german,
            figsize,
            dpi,
            font,
            fontsize,
        )
