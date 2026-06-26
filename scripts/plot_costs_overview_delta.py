# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plot costs for all scenarios side-by-side for certain carrier.
"""

import logging
import ast
import matplotlib.pyplot as plt
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def import_csvs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Import costs from long-term and short-term runs.
    """

    data_col = "cost"

    data_list = []
    for i, path in enumerate(df["path"]):
        data = pd.read_csv(
            path, index_col=list(range(3)), header=list(range(3))
        )
        # Rename three columns to
        data.columns = data.columns.get_level_values('planning_horizon')
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

        # Append to cost
        data_list.append(data)

    data = pd.concat(data_list)

    return data


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_costs_overview_delta",
            configfiles=["config/config.nrw.yaml"],
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    config = snakemake.config
    plotting = snakemake.params.plotting_fig
    nice_names = config["plotting"]["nice_names"]
    tech_colors = config["plotting"]["tech_colors"]

    figsize = ast.literal_eval(plotting["figsize"])
    fontsize = plotting["font"]["size"]
    subfontsize = fontsize
    dpi = plotting["dpi"]

    opts = config["scenario"]["opts"][0]
    sector_opts = config["scenario"]["sector_opts"][0]
    font = plotting["font"]
    legend_order = plotting["legend_order"]

    # Drop load shedding if in legend_order
    if "Load shedding" in legend_order:
        legend_order.remove("Load shedding")

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    lt_order = [col for col in plotting["run_order"]]
    lt_order_nice_names = plotting["nice_names"]

    # [
    #     plotting["nice_names"][col] for col in plotting["run_order"]
    # ]

    carrier_groups = config["grouping"]
    group_colors = config["group_colors"]

    # Create df of all runs (rows)
    costs = pd.DataFrame()
    costs["path"] = snakemake.input.costs
    costs["prefix"] = costs["path"].apply(lambda x: x.split("/")[-4])
    costs["name"] = costs["path"].apply(lambda x: x.split("/")[-3])

    costs = import_csvs(costs).fillna(0)
    costs["group"] = costs["carrier"].map(carrier_groups)
    costs["group_color"] = costs["group"].map(group_colors)

    # to_drop = costs.index[(costs.value.abs()<10)] # Drop small values
    # costs = costs.drop(to_drop, axis=0)
    # costs.reset_index(drop=True, inplace=True)

    # Group by group
    costs = costs.groupby(["planning_horizon", "group", "name", "group_color"], observed=True).agg(
        value=("value", "sum"),
    ).div(1e9) # EUR to bn. EUR p.a.
    costs.reset_index(inplace=True)

    # Nice names for lt_run
    costs["nice_name"] = costs["name"].map(plotting["nice_names"])

    # # Move name column values to columns
    # costs = costs.pivot(
    #     index=["planning_horizon", "group", "group_color"],
    #     columns="nice_name",
    #     values="value",
    # ).reset_index()

    # Drop load shedding after debugging
    if "Load shedding" in costs.group.values:
        costs = costs[costs["group"] != "Load shedding"]

    # First plot
    n_planning_horizons = len(planning_horizons)

    # ymax = (costs.groupby(["planning_horizon", "name"], observed=True).sum().max(numeric_only=True)).max()
    # ymin = 0

    x_anchor = 0
    ncol = 4
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
        ax = axes[i]
        planning_horizon = str(planning_horizon)
        data = costs.query("planning_horizon == @planning_horizon").copy().pivot(
            index="name",
            columns="group",
            values="value",
        )

        main_scenario = plotting["main_scenario"]
        other_scenarios = [s for s in lt_order if s != main_scenario and s in data.index]
        data_main = data.loc[main_scenario]

        # Deltas
        data = data.loc[other_scenarios] - data_main

        data_order = [col for col in legend_order if col in data.columns]
        data = data[data_order]

        # Apply run_order, then rename to nice names
        data = data.reindex([name for name in lt_order if name in data.index])
        data = data.rename(index=lt_order_nice_names)

        data.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=0.8,
            color=[group_colors.get(col, "yellow") for col in data.columns],
        )

        # Turn off legend
        ax.legend().remove()

        # Set title and labels
        ax.set_xlabel(f"{planning_horizon}", fontsize=fontsize)
        ax.set_ylabel(f"Delta system costs (bn. € p.a.)", fontsize=fontsize)

        ax.set_xticklabels(
            data.index,
            rotation=90,
            fontsize=subfontsize,
        )

        # Remove all grid lines
        ax.grid(False)

        # Remove y ticks in all but the first plot
        if i > 0:
            ax.yaxis.set_visible(False)

        # Net sum label and dot per column
        nets = data.sum(axis=1)
        for j, net in enumerate(nets):
            sign = "+" if net > 0 else ""
            ax.plot(j, net, marker="o", color="black", markersize=3, zorder=5)
            ax.text(
                x=j,
                y=net,
                s=f"{sign}{net:.0f}",
                ha="center",
                va="bottom",
                fontsize=subfontsize,
            )

        # Add 0 axis line
        ax.axhline(0, color="black", lw=0.5)

    # Set ylim by using the maximum absolute value across all axes
    ymax = 0
    for ax in axes:
        ax_ymax = ax.get_ylim()[1]
        ax_ymin = ax.get_ylim()[0]
        ax_max = max(abs(ax_ymax), abs(ax_ymin))
        if ax_max > ymax:
            ymax = ax_max
            ymin = -ymax
    for ax in axes:
        ax.set_ylim(ymin*1.1, ymax*1.1)

    # Change font size of major sharey ticks
    for ax in axes:
        ax.tick_params(axis="y", labelsize=subfontsize)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=group_colors[c], label=c)
        for c in legend_order[::-1]
    ]

    # Add the production legend (left side, 2 columns)
    legend = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(x_anchor+xpad, 0.03),  # fixed at 0 (left-aligned)
        ncol=ncol,
        fontsize=subfontsize,
        title="",
        title_fontsize=subfontsize,
        frameon=False,
        handlelength=handlelength,
        handleheight=handleheight,
    )
    legend.get_title().set_fontweight('bold')
    legend._legend_box.align = "left"

    # All borders to 0.5 thickness
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("black")

    # Tight layout
    plt.tight_layout()

    fig.subplots_adjust(wspace=0.05)

    fig.savefig(
        snakemake.output.plot,
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(snakemake.output.png, dpi=150, bbox_inches="tight")
