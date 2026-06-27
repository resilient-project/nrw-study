# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plot absolute and delta system costs stacked vertically (2 rows) for all scenarios.
"""

import logging
import ast
import matplotlib.pyplot as plt
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

NOISE_THRESHOLD_BN_EUR = 1e-4


def import_csvs(df: pd.DataFrame) -> pd.DataFrame:
    data_col = "cost"
    data_list = []
    for i, path in enumerate(df["path"]):
        data = pd.read_csv(path, index_col=list(range(3)), header=list(range(3)))
        data.columns = data.columns.get_level_values("planning_horizon")
        planning_horizons = data.columns
        data.reset_index(inplace=True)
        data = data.melt(
            id_vars=[data_col, "component", "carrier"],
            value_vars=planning_horizons,
            var_name="planning_horizon",
            value_name="value",
        )
        data["name"] = df.loc[i, "name"]
        data["planning_horizon"] = data["planning_horizon"].astype(str)
        data_list.append(data)
    return pd.concat(data_list)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_costs_overview_stacked",
            configfiles=["config/config.nrw.yaml"],
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    config = snakemake.config
    plotting = snakemake.params.plotting_fig
    tech_colors = config["plotting"]["tech_colors"]

    figsize = ast.literal_eval(plotting["figsize"])
    fontsize = plotting["font"]["size"]
    dpi = plotting["dpi"]
    font = plotting["font"]
    legend_order = plotting["legend_order"]

    if "Load shedding" in legend_order:
        legend_order.remove("Load shedding")

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    lt_order = list(plotting["run_order"])
    lt_order_nice_names = plotting["nice_names"]
    main_scenario = plotting["main_scenario"]

    carrier_groups = dict(config["grouping"])
    group_colors = config["group_colors"]

    if plotting.get("aggregate_industry_emissions", False):
        aggregate_label = plotting.get("aggregate_label", "Emissionen Industrie")
        carrier_groups["gas for industry CC"] = aggregate_label
        carrier_groups["process emissions CC"] = aggregate_label
        aggregate_color = plotting.get("aggregate_color")
        if aggregate_color:
            group_colors = dict(group_colors)
            group_colors[aggregate_label] = aggregate_color

    for carrier, group in plotting.get("carrier_group_overrides", {}).items():
        carrier_groups[carrier] = group

    group_merge = plotting.get("group_merge", {})
    if group_merge:
        carrier_groups = {k: group_merge.get(v, v) for k, v in carrier_groups.items()}

    group_merge_colors = plotting.get("group_merge_colors", {})
    if group_merge_colors:
        group_colors = dict(group_colors)
        group_colors.update(group_merge_colors)

    costs = pd.DataFrame()
    costs["path"] = snakemake.input.costs
    costs["prefix"] = costs["path"].apply(lambda x: x.split("/")[-4])
    costs["name"] = costs["path"].apply(lambda x: x.split("/")[-3])

    costs = import_csvs(costs).fillna(0)
    costs["group"] = costs["carrier"].map(carrier_groups)
    costs["group_color"] = costs["group"].map(group_colors)

    costs = (
        costs.groupby(["planning_horizon", "group", "name", "group_color"], observed=True)
        .agg(value=("value", "sum"))
        .div(1e9)
    )
    costs.reset_index(inplace=True)
    costs.loc[costs["value"].abs() < NOISE_THRESHOLD_BN_EUR, "value"] = 0.0

    if "Load shedding" in costs.group.values:
        costs = costs[costs["group"] != "Load shedding"]

    n_planning_horizons = len(planning_horizons)

    ymax_abs = (
        costs.groupby(["planning_horizon", "name"], observed=True)
        .sum(numeric_only=True)
        .max()
        .max()
    )

    handlelength = 1
    handleheight = 1.1

    delta_pos_max = 0.0
    delta_neg_min = 0.0

    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_planning_horizons,
        figsize=figsize,
        dpi=dpi,
        sharey="row",
        tight_layout=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    plt.rc("font", **font)

    for i, planning_horizon in enumerate(planning_horizons):
        ph = str(planning_horizon)
        ax_delta = axes[0, i]  # delta: top, 2/3 height
        ax_abs = axes[1, i]    # absolute: bottom, 1/3 height

        pivot = costs.query("planning_horizon == @ph").copy().pivot(
            index="name", columns="group", values="value"
        )
        data_order = [col for col in legend_order if col in pivot.columns]
        pivot = pivot[data_order]
        pivot = pivot.reindex([n for n in lt_order if n in pivot.index])

        # --- top row: delta vs. main scenario ---
        data_main = pivot.loc[main_scenario]
        data_delta = (pivot - data_main).rename(index=lt_order_nice_names)
        delta_pos_max = max(delta_pos_max, data_delta.clip(lower=0).sum(axis=1).max())
        delta_neg_min = min(delta_neg_min, data_delta.clip(upper=0).sum(axis=1).min())

        data_delta.plot(
            kind="bar",
            stacked=True,
            ax=ax_delta,
            width=0.8,
            color=[group_colors.get(col, "yellow") for col in data_delta.columns],
        )
        ax_delta.legend().remove()
        ax_delta.set_xlabel("")
        ax_delta.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_delta.grid(False)
        ax_delta.axhline(0, color="black", lw=0.5)

        nets = data_delta.sum(axis=1)
        for j, net in enumerate(nets):
            net_int = int(round(net))
            sign = "+" if net_int > 0 else ""
            ax_delta.plot(j, net, marker="o", color="black", markersize=3, zorder=5)
            ax_delta.text(j, net, f"{sign}{net_int}", ha="center", va="bottom", fontsize=fontsize)

        if i == 0:
            ax_delta.set_ylabel("Kostendifferenz zu 1.a\n(Mrd. € p.a.)", fontsize=fontsize)
        else:
            ax_delta.yaxis.set_visible(False)

        # --- bottom row: absolute costs ---
        data_abs = pivot.rename(index=lt_order_nice_names)

        data_abs.plot(
            kind="bar",
            stacked=True,
            ax=ax_abs,
            width=0.8,
            color=[group_colors.get(col, "yellow") for col in data_abs.columns],
        )
        ax_abs.legend().remove()
        ax_abs.set_xlabel(f"{planning_horizon}", fontsize=fontsize)
        ax_abs.set_xticklabels(data_abs.index, rotation=0, fontsize=fontsize)
        ax_abs.set_ylim(0, ymax_abs * 1.1)
        ax_abs.grid(False)
        ax_abs.axhline(0, color="black", lw=0.5)

        totals = data_abs[data_abs > 0].sum(axis=1)
        for j, total in enumerate(totals):
            if total > 0:
                ax_abs.text(j, total, f"{total:.0f}", ha="center", va="bottom", fontsize=fontsize)

        if i == 0:
            ax_abs.set_ylabel("Gesamtsystemkosten\n(Mrd. € p.a.)", fontsize=fontsize)
        else:
            ax_abs.yaxis.set_visible(False)

    # Symmetric y limits for delta row (top, axes[0]) — data-driven, single 1.1 padding
    ymax_delta = max(delta_pos_max, abs(delta_neg_min)) * 1.1
    for ax in axes[0]:
        ax.set_ylim(-ymax_delta, ymax_delta)

    for ax in axes.flat:
        ax.tick_params(axis="y", labelsize=fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("black")

    group_exceptions = set(plotting.get("group_exceptions", []))
    active_groups = set(costs["group"].dropna().unique()) | group_exceptions
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=group_colors[c], label=c)
        for c in legend_order[::-1]
        if c in group_colors and c in active_groups
    ]

    # Legend to the left of the plots, 2 columns
    legend = fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        ncol=1,
        fontsize=fontsize,
        title="",
        title_fontsize=fontsize,
        frameon=False,
        handlelength=handlelength,
        handleheight=handleheight,
    )
    legend.get_title().set_fontweight("bold")
    legend._legend_box.align = "left"

    plt.tight_layout()
    fig.subplots_adjust(left=0.30, hspace=0.05, wspace=0.05)

    fig.savefig(snakemake.output.plot, dpi=dpi, bbox_inches="tight")
    fig.savefig(snakemake.output.png, dpi=150, bbox_inches="tight")
