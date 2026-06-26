# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Plot installed capacities for all scenarios side-by-side.
"""

import logging
import ast
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def import_csvs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    data_list = []
    for i, path in enumerate(df["path"]):
        data = pd.read_csv(
            path, index_col=list(range(2)), header=list(range(3))
        )
        data.columns = data.columns.get_level_values('planning_horizon')
        planning_horizons = data.columns

        data.reset_index(inplace=True)
        data = data.melt(
            id_vars=["component", "carrier"],
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
            "plot_capacities_overview",
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

    if "Load shedding" in legend_order:
        legend_order.remove("Load shedding")

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    lt_order = [col for col in plotting["run_order"]]
    lt_order_nice_names = plotting["nice_names"]

    carrier_groups = config["grouping"]
    group_colors = config["group_colors"]

    caps = pd.DataFrame()
    caps["path"] = snakemake.input.capacities
    caps["prefix"] = caps["path"].apply(lambda x: x.split("/")[-4])
    caps["name"] = caps["path"].apply(lambda x: x.split("/")[-3])

    caps = import_csvs(caps).fillna(0)
    caps["group"] = caps["carrier"].map(carrier_groups)
    caps["group_color"] = caps["group"].map(group_colors)

    # Derive valid (component, carrier) pairs from AC supply statistics
    n = pypsa.Network(snakemake.input.network)
    supply = n.statistics.supply(bus_carrier="AC")
    nice_to_raw = (
        n.carriers["nice_name"][n.carriers["nice_name"].str.strip() != ""]
        .reset_index()
        .set_index("nice_name")["name"]
        .to_dict()
    )
    valid_pairs = {
        (comp, nice_to_raw.get(carrier, carrier))
        for comp, carrier in supply.index
    }
    mask = pd.MultiIndex.from_arrays([caps["component"], caps["carrier"]]).isin(valid_pairs)
    caps = caps[mask]

    # Drop AC, DC transmission
    caps = caps[~caps["component"].isin(["Line"])]
    caps = caps[~caps["carrier"].isin(["DC", "electricity distribution grid"])]

    caps = caps.groupby(["planning_horizon", "group", "name", "group_color"], observed=True).agg(
        value=("value", "sum"),
    ).div(1e3)  # MW to GW
    caps.reset_index(inplace=True)

    caps["nice_name"] = caps["name"].map(plotting["nice_names"])

    if "Load shedding" in caps.group.values:
        caps = caps[caps["group"] != "Load shedding"]

    n_planning_horizons = len(planning_horizons)

    ymax = caps[caps["value"] > 0].groupby(["planning_horizon", "name"], observed=True)["value"].sum().max()
    ymin = 0

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
        data = caps.query("planning_horizon == @planning_horizon").copy().pivot(
            index="name",
            columns="group",
            values="value",
        )

        data_order = [col for col in legend_order if col in data.columns]
        data = data[data_order]

        data = data.reindex([name for name in lt_order if name in data.index])
        data = data.rename(index=lt_order_nice_names)

        data.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=0.8,
            color=[group_colors.get(col, "yellow") for col in data.columns],
        )

        ax.legend().remove()

        ax.set_xlabel(f"{planning_horizon}", fontsize=fontsize)
        ax.set_ylabel(f"Capacities (GW)", fontsize=fontsize)

        ax.set_ylim(ymin, ymax * 1.1)

        ax.set_xticklabels(
            data.index,
            rotation=90,
            fontsize=subfontsize,
        )

        ax.grid(False)

        if i > 0:
            ax.yaxis.set_visible(False)

        totals = data[data > 0].sum(axis=1)
        for j, total in enumerate(totals):
            if total > 0:
                ax.text(
                    x=j,
                    y=total,
                    s=f"{total:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=subfontsize,
                )

        ax.axhline(0, color="black", lw=0.5)

    for ax in axes:
        ax.tick_params(axis="y", labelsize=subfontsize)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=group_colors[c], label=c)
        for c in legend_order if c in list(data.columns)
    ]
    handles = handles[::-1]

    legend = fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(x_anchor + xpad, 0.055),
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

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("black")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05)

    fig.savefig(
        snakemake.output.plot,
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(snakemake.output.png, dpi=150, bbox_inches="tight")
