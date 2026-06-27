# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Plot CO2 pipeline volume and length comparison across all NRW scenarios.

Layout: 2 rows (volume_mtpakm, length_km) × n planning horizons columns.
Bars are stacked by segment: Offshore / Onshore pipeline DE / Onshore short DE /
Onshore pipeline DEA / Onshore short DEA.
"""

import ast
import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from scripts._helpers import configure_logging, set_scenario_config

# Each segment: (label, terrain, list-of-regions, carrier-or-None)
SEGMENTS = [
    ("Offshore (DE)",       "offshore", ["DE", "DEA"], None),
    ("Onshore DN700 (DE)", "onshore",  ["DE"],        "CO2 pipeline"),
    ("Onshore DN400 (DE)", "onshore",  ["DE"],        "CO2 pipeline short"),
    ("Onshore DN700 (NRW)","onshore",  ["DEA"],       "CO2 pipeline"),
    ("Onshore DN400 (NRW)","onshore",  ["DEA"],       "CO2 pipeline short"),
]


def load_csvs(paths: list[str]) -> pd.DataFrame:
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df["name"] = path.split("/")[-3]
        match = re.search(r"(\d{4})\.csv$", path)
        df["planning_horizon"] = match.group(1) if match else "unknown"
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_segment_table(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return DataFrame with (name, planning_horizon) as index and segments as columns."""
    cols = []
    for label, terrain, regions, carrier in SEGMENTS:
        mask = (data["terrain"] == terrain) & data["region"].isin(regions)
        if carrier is not None:
            mask &= data["carrier"] == carrier
        agg = data[mask].groupby(["name", "planning_horizon"])[metric].sum().rename(label)
        cols.append(agg)
    return pd.concat(cols, axis=1).fillna(0)


def plot_metric(axes, seg_table, planning_horizons, run_order, nice_names,
                segment_order, segment_colors, ylabel, fontsize,
                xticklabel_size=None, show_xlabels=True):
    if xticklabel_size is None:
        xticklabel_size = fontsize
    ymax = 0
    plot_data = {}
    for i, ph in enumerate(planning_horizons):
        ax = axes[i]
        ph = str(ph)
        data = seg_table.xs(ph, level="planning_horizon")
        data = data.reindex([r for r in run_order if r in data.index])
        data = data.rename(index=nice_names)
        data = data[[s for s in segment_order if s in data.columns]]
        plot_data[ph] = data

        data.plot(
            kind="bar", stacked=True, ax=ax, width=0.8,
            color=[segment_colors[s] for s in data.columns],
            edgecolor="none",
            legend=False,
        )
        n_bars = len(data)
        nrw_cols = [j for j, col in enumerate(data.columns) if "(NRW)" in col]
        for col_idx in nrw_cols:
            for bar_idx in range(n_bars):
                patch = ax.patches[col_idx * n_bars + bar_idx]
                patch.set_hatch("//////")
                patch.set_edgecolor("white")
                patch.set_linewidth(0)
        ymax = max(ymax, data[data > 0].sum(axis=1).max())

        ax.set_xlabel(ph if show_xlabels else "", fontsize=fontsize)
        if show_xlabels:
            ax.set_xticklabels(data.index, rotation=0, fontsize=xticklabel_size)
        else:
            ax.tick_params(labelbottom=False)
        ax.grid(False)
        ax.axhline(0, color="black", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)

        if i == 0:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        else:
            ax.yaxis.set_visible(False)

    for i, ph in enumerate(planning_horizons):
        ph = str(ph)
        axes[i].set_ylim(0, ymax * 1.15)
        axes[i].tick_params(axis="y", labelsize=fontsize)
        totals = plot_data[ph][plot_data[ph] > 0].sum(axis=1)
        for j, total in enumerate(totals):
            if total > 0:
                axes[i].text(j, total + ymax * 0.04, f"{total:.0f}", ha="center", va="bottom",
                             fontsize=fontsize)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_co2_pipeline_comparison",
            configfiles=["config/config.nrw.yaml"],
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    plotting = snakemake.params.plotting_fig
    font = plotting["font"]
    fontsize = font["size"]
    xticklabel_size = plotting.get("xticklabel_size", fontsize)
    figsize = ast.literal_eval(plotting["figsize"])
    dpi = plotting["dpi"]
    run_order = plotting["run_order"]
    nice_names = plotting["nice_names"]
    segment_order = plotting["segment_order"]
    segment_colors = plotting["segment_colors"]

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    n_ph = len(planning_horizons)

    data = load_csvs(snakemake.input.csvs)

    vol_table = build_segment_table(data, "volume_mtpakm") / 1e6  # Mtpa·km → Mtpa·Mkm
    len_table = build_segment_table(data, "length_km")

    fig, axes = plt.subplots(
        nrows=2, ncols=n_ph,
        figsize=figsize, dpi=dpi,
        sharey="row",
        tight_layout=True,
        gridspec_kw={"height_ratios": [1.022, 1]},
    )
    plt.rc("font", **font)
    plt.rcParams["hatch.linewidth"] = 0.8

    plot_metric(axes[0], vol_table, planning_horizons, run_order, nice_names,
                segment_order, segment_colors, "Transportleist. (Mtpa·Mkm)", fontsize,
                xticklabel_size=xticklabel_size, show_xlabels=False)
    plot_metric(axes[1], len_table, planning_horizons, run_order, nice_names,
                segment_order, segment_colors, "Pipelinelängen (km)", fontsize,
                xticklabel_size=xticklabel_size, show_xlabels=True)
    fig.align_ylabels([axes[0, 0], axes[1, 0]])

    DE_LEGEND = {
        "Offshore (DE)": "Offshore",
        "Onshore DN700 (DE)": "Onshore DN700",
        "Onshore DN400 (DE)": "Onshore DN400",
    }
    de_segments = [s for s in segment_order if s in DE_LEGEND]
    handles = [
        Patch(facecolor=segment_colors[s], label=DE_LEGEND[s])
        for s in de_segments[::-1]
    ]
    handles.append(
        Patch(facecolor=segment_colors["Onshore DN400 (NRW)"], hatch="//////",
              edgecolor="white", label="NRW")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(handles),
        fontsize=fontsize,
        frameon=False,
        handlelength=0.8,
        handleheight=0.8,
    )

    fig.subplots_adjust(wspace=0.05, hspace=0.5)

    fig.savefig(snakemake.output.plot, dpi=dpi, bbox_inches="tight")
    fig.savefig(snakemake.output.png, dpi=300, bbox_inches="tight")
    plt.close(fig)
