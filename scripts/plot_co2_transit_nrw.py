#!/usr/bin/env python3
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
CO₂ transit through NRW (NUTS-1 DEA) derived from solved networks.

Scope is the ``co2 stored`` pipeline network. Per snapshot t and per NRW node:

  S  local injections into NRW ``co2 stored`` buses (all CC, DAC, buffer
     store discharge) – every non-pipeline component counts here
  D  local withdrawals (methanolisation, Fischer-Tropsch, Sabatier,
     sequestration, buffer store charge) plus pipeline losses inside NRW
  I  pipeline inflow delivered into NRW across the DEA border
  O  pipeline outflow sent out of NRW across the DEA border

with S + I = D + O. Uncaptured emissions never enter the pipeline network
and are not part of the balance. Transit (CO₂ entering and leaving NRW) is
not uniquely defined by the balance; three numbers are reported:

  transit_min    Σ_t max(0, I_t − D_t)    imports NRW sinks cannot absorb
  transit_max    Σ_t min(I_t, O_t)        all imports that could leave again
  transit_trace  proportional sharing (perfect mixing at every NRW node),
                 cf. Bialek (1996), Hörsch et al. (2018)

All quantities are evaluated per snapshot and summed with the snapshot
weightings; annual-netting bounds are kept in the CSV for comparison.

Outputs: summary CSV, origin-destination CSV (tracing), a dashboard across
scenarios and planning horizons, and one Sankey per scenario/year.
"""

import ast
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.path import Path as MplPath

from scripts._helpers import configure_logging, set_scenario_config
from scripts._poster import get_poster, merge_settings, translate_figure

logger = logging.getLogger(__name__)

REGION_PREFIX = "DEA"
BUS_CARRIER = "co2 stored"
PIPE_CARRIERS = frozenset({"CO2 pipeline", "CO2 pipeline short"})
LOSS_LABEL = "Leitungsverluste"
TRANSIT_COLOR = "#f2385a"
LOCAL_COLOR = "#503530"
IMPORT_LOCAL_COLOR = "#b0a49f"
MIN_FLOW_MT = 0.05  # Sankey ribbons below this are dropped [MtCO₂/a]
MIN_IMPORT_MT = 0.1  # below this, the transit share is not shown [MtCO₂/a]


# ---------------------------------------------------------------------------
# Network data
# ---------------------------------------------------------------------------


def _neighbour(location: str, neighbour_names: dict) -> str:
    """Label of an outside node: NUTS-1 for Germany, country otherwise."""
    code = location[:3] if location.startswith("DE") else location[:2]
    return neighbour_names.get(code, code)


def _local_injections(n: pypsa.Network, buses: pd.Index, rename: dict) -> pd.DataFrame:
    """
    Injection into each bus by non-pipeline carrier [t/h], positive = into bus.

    Columns are a MultiIndex (bus, carrier); sign may change over time
    (buffer stores), so sources and sinks are split later.
    """
    parts = []
    links = n.links[~n.links.carrier.isin(PIPE_CARRIERS)]
    for port in [c[3:] for c in links.columns if re.fullmatch(r"bus\d", c)]:
        sel = links.index[links[f"bus{port}"].isin(buses)]
        if sel.empty or f"p{port}" not in n.links_t:
            continue
        p = -n.links_t[f"p{port}"].reindex(columns=sel, fill_value=0.0)
        p.columns = pd.MultiIndex.from_arrays(
            [links.loc[sel, f"bus{port}"], links.loc[sel, "carrier"].replace(rename)]
        )
        parts.append(p)
    stores = n.stores.index[n.stores.bus.isin(buses)]
    if not stores.empty:
        p = n.stores_t.p.reindex(columns=stores, fill_value=0.0)
        p.columns = pd.MultiIndex.from_arrays(
            [n.stores.loc[stores, "bus"], n.stores.loc[stores, "carrier"]]
        )
        parts.append(p)
    if not parts:
        return pd.DataFrame(
            index=n.snapshots, columns=pd.MultiIndex.from_tuples([], names=[None, None])
        )
    inj = pd.concat(parts, axis=1)
    return inj.T.groupby(level=[0, 1]).sum().T


def compute_transit(
    n: pypsa.Network,
    neighbour_names: dict,
    rename: dict,
    region_prefix: str = REGION_PREFIX,
) -> tuple[dict, pd.DataFrame]:
    """
    Return (summary record, origin-destination table) in MtCO₂/a.

    The OD table traces every origin (local capture by technology, imports
    by neighbour) to every destination (local use by technology, exports by
    neighbour, pipeline losses) under proportional sharing.
    """
    w = n.snapshot_weightings.generators
    bus_loc = n.buses.location
    co2_buses = n.buses.index[n.buses.carrier == BUS_CARRIER]
    nodes = co2_buses[bus_loc[co2_buses].str.startswith(region_prefix)]
    idx = {b: i for i, b in enumerate(nodes)}
    n_nodes, n_snap = len(nodes), len(n.snapshots)

    # --- pipelines -------------------------------------------------------
    pipes = n.links[n.links.carrier.isin(PIPE_CARRIERS)]
    in0 = pipes.bus0.isin(nodes)
    in1 = pipes.bus1.isin(nodes)
    sent = n.links_t.p0.reindex(columns=pipes.index, fill_value=0.0)
    delivered = -n.links_t.p1.reindex(columns=pipes.index, fill_value=0.0)

    internal = pipes.index[in0 & in1]
    border_in = pipes.index[~in0 & in1]
    border_out = pipes.index[in0 & ~in1]

    # Internal delivered flow matrix F[t, j, i] (j → i) and losses per sender
    F = np.zeros((n_snap, n_nodes, n_nodes))
    loss = np.zeros((n_snap, n_nodes))
    for link in internal:
        j, i = idx[pipes.at[link, "bus0"]], idx[pipes.at[link, "bus1"]]
        F[:, j, i] += delivered[link].to_numpy()
        loss[:, j] += (sent[link] - delivered[link]).to_numpy()

    # --- origins (local sources + imports) and destinations (local sinks + exports)
    inj = _local_injections(n, nodes, rename)
    sources = inj.clip(lower=0)
    sinks = -inj.clip(upper=0)

    def _to_cube(df_cols: dict[str, pd.DataFrame]) -> tuple[list[str], np.ndarray]:
        """{label: DataFrame(time × bus)} → labels, array[t, node, label]."""
        labels = list(df_cols)
        cube = np.zeros((n_snap, n_nodes, len(labels)))
        for k, label in enumerate(labels):
            df = df_cols[label]
            for bus in df.columns:
                cube[:, idx[bus], k] += df[bus].to_numpy()
        return labels, cube

    def _by_carrier(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {
            c: df.xs(c, axis=1, level=1)
            for c in df.columns.get_level_values(1).unique()
            if df.xs(c, axis=1, level=1).to_numpy().any()
        }

    def _border(links: pd.Index, flow: pd.DataFrame, inside: str, outside: str):
        out: dict[str, pd.DataFrame] = {}
        for link in links:
            label = _neighbour(bus_loc[pipes.at[link, outside]], neighbour_names)
            bus = pipes.at[link, inside]
            df = out.setdefault(
                label, pd.DataFrame(0.0, index=n.snapshots, columns=[bus])
            )
            df[bus] = df.get(bus, 0.0) + flow[link]
        return out

    imports = _border(border_in, delivered, inside="bus1", outside="bus0")
    exports = _border(border_out, sent, inside="bus0", outside="bus1")

    src_labels, src = _to_cube(
        {**_by_carrier(sources), **{f"Import {k}": v for k, v in imports.items()}}
    )
    snk_labels, snk = _to_cube(
        {**_by_carrier(sinks), **{f"Export {k}": v for k, v in exports.items()}}
    )
    snk_labels.append(LOSS_LABEL)
    snk = np.concatenate([snk, loss[:, :, None]], axis=2)

    is_import = np.array([s.startswith("Import ") for s in src_labels])
    is_export = np.array([s.startswith("Export ") for s in snk_labels])

    # --- proportional sharing per snapshot ---------------------------------
    # Throughput P_i = local/import injection + internal inflow; composition
    # x_i solves diag(P) x − Fᵀ x = L.
    od = np.zeros((len(src_labels), len(snk_labels)))
    throughput = src.sum(axis=2) + F.sum(axis=1)
    for t in range(n_snap):
        P = throughput[t]
        A = np.diag(np.where(P > 1e-9, P, 1.0)) - F[t].T
        X = np.linalg.solve(A, src[t])  # node × origin
        od += w.iloc[t] * X.T @ snk[t]

    # --- balance and bounds -----------------------------------------------
    S = src[:, :, ~is_import].sum(axis=(1, 2))
    I = src[:, :, is_import].sum(axis=(1, 2))
    O = snk[:, :, is_export].sum(axis=(1, 2))
    D = snk[:, :, ~is_export].sum(axis=(1, 2))  # incl. internal losses
    wv = w.to_numpy()

    def tot(x):
        return float((x * wv).sum()) / 1e6

    imbalance = tot(np.abs(S + I - D - O))
    if imbalance > 1e-3 * max(tot(S + I), 1e-9):
        logger.warning("NRW CO₂ balance does not close: %.4f Mt", imbalance)

    od = pd.DataFrame(od / 1e6, index=src_labels, columns=snk_labels)
    record = {
        "S": tot(S),
        "D": tot(D),
        "I": tot(I),
        "O": tot(O),
        "losses": tot(loss.sum(axis=1)),
        "transit_min": tot(np.maximum(0.0, I - D)),
        "transit_max": tot(np.minimum(I, O)),
        "transit_trace": float(od.loc[is_import, is_export].to_numpy().sum()),
    }
    record["transit_min_annual"] = max(0.0, record["I"] - record["D"])
    record["transit_max_annual"] = min(record["I"], record["O"])
    for key in ["min", "max", "trace"]:
        t_ = record[f"transit_{key}"]
        record[f"share_of_imports_{key}"] = t_ / record["I"] if record["I"] else np.nan
        record[f"share_of_exports_{key}"] = t_ / record["O"] if record["O"] else np.nan
    return record, od


def build_tables(
    paths,
    run_order,
    planning_horizons,
    neighbour_names,
    rename,
    region_prefix=REGION_PREFIX,
):
    run_set = set(run_order)
    ph_set = {str(ph) for ph in planning_horizons}
    records, ods = [], []
    for path in paths:
        run = path.split("/")[-3]
        m = re.search(r"(\d{4})\.nc$", path)
        year = m.group(1) if m else None
        if run not in run_set or year not in ph_set:
            continue
        logger.info("Loading %s %s from %s", run, year, path)
        n = pypsa.Network(path)
        record, od = compute_transit(n, neighbour_names, rename, region_prefix)
        records.append({"run": run, "planning_horizon": year, **record})
        od = od.stack().rename("value").rename_axis(["origin", "destination"])
        ods.append(od.reset_index().assign(run=run, planning_horizon=year))
        logger.info(
            "%s %s: I=%.2f O=%.2f transit min/trace/max = %.2f / %.2f / %.2f Mt",
            run,
            year,
            record["I"],
            record["O"],
            record["transit_min"],
            record["transit_trace"],
            record["transit_max"],
        )
    if not records:
        raise ValueError(
            f"No networks matched run_order={sorted(run_set)} and planning_horizons={sorted(ph_set)}"
        )
    summary = pd.DataFrame(records)
    od = pd.concat(ods, ignore_index=True)[
        ["run", "planning_horizon", "origin", "destination", "value"]
    ]
    return summary, od


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def plot_dashboard(
    summary,
    planning_horizons,
    run_order,
    nice_names,
    fontsize,
    figsize,
    dpi,
    out_pdf,
    out_png,
    poster=None,
    legend_ncol=4,
):
    """
    One bar per scenario splitting all CO₂ crossing the NRW border into
    non-overlapping parts: transit (traced), imports used in NRW (I − T) and
    exports of NRW-captured CO₂ (O − T). Import = red + grey, export = red +
    brown. The whisker on the transit segment shows the [min, max] bounds.
    """
    fig, axes = plt.subplots(
        1, len(planning_horizons), figsize=figsize, dpi=dpi, sharey=True, squeeze=False
    )
    axes = axes[0]
    bw = 0.6
    ymax = summary.eval("I + O - transit_trace").max()
    for ax, ph in zip(axes, planning_horizons):
        sub = summary[summary.planning_horizon == str(ph)].set_index("run")
        for j, run in enumerate(run_order):
            if run not in sub.index:
                continue
            r = sub.loc[run]
            parts = [
                (r["transit_trace"], TRANSIT_COLOR, "white"),
                (r["I"] - r["transit_trace"], IMPORT_LOCAL_COLOR, "#333333"),
                (r["O"] - r["transit_trace"], LOCAL_COLOR, "white"),
            ]
            bottom = 0.0
            for value, color, text_color in parts:
                ax.bar(
                    j, value, bw, bottom=bottom, color=color, edgecolor="none", zorder=3
                )
                if value >= 0.06 * ymax:
                    ax.text(
                        j,
                        bottom + value / 2,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=fontsize - 2,
                        color=text_color,
                        zorder=5,
                    )
                bottom += value
            ax.errorbar(
                j + bw / 2 + 0.07,
                r["transit_trace"],
                yerr=[
                    [r["transit_trace"] - r["transit_min"]],
                    [r["transit_max"] - r["transit_trace"]],
                ],
                fmt="none",
                ecolor="black",
                elinewidth=0.8,
                capsize=2.5,
                zorder=4,
            )
            share = (
                f"{r['share_of_imports_trace']:.0%}" if r["I"] >= MIN_IMPORT_MT else "–"
            )
            ax.annotate(
                f"{bottom:.1f}",
                (j, bottom),
                xytext=(0, 1),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize - 2,
            )
            ax.annotate(
                share,
                (j, bottom),
                xytext=(0, fontsize + 1),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize - 2,
                color=TRANSIT_COLOR,
                fontweight="bold",
            )
        ax.set_xticks(range(len(run_order)))
        ax.set_xticklabels([nice_names.get(r, r) for r in run_order], fontsize=fontsize)
        ax.set_xlabel(str(ph), fontsize=fontsize)
        ax.set_xlim(-0.6, len(run_order) - 0.4)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", labelsize=fontsize)
    axes[0].set_ylabel("CO$_2$-Fluss über NRW-Grenze (Mt/a)", fontsize=fontsize)
    axes[0].set_ylim(0, max(ymax * 1.15, 1))

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TRANSIT_COLOR, label="Transit (durch NRW)"),
        plt.Rectangle(
            (0, 0), 1, 1, color=IMPORT_LOCAL_COLOR, label="Import (Nutzung in NRW)"
        ),
        plt.Rectangle(
            (0, 0), 1, 1, color=LOCAL_COLOR, label="Export (aus Abscheidung in NRW)"
        ),
        plt.Line2D(
            [],
            [],
            color="black",
            lw=0.8,
            marker="_",
            markersize=6,
            label="Bandbreite Transit",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.06, 0.02),
        ncol=legend_ncol,
        fontsize=fontsize,
        frameon=False,
        handlelength=0.9,
    )
    fig.text(
        0.06,
        -0.06 - 0.09 * (-(-4 // legend_ncol) - 1),
        "Prozentwert: Transitanteil an CO$_2$-Importmenge",
        fontsize=fontsize - 2,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.2, wspace=0.08)
    translate_figure(fig, poster)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sankey
# ---------------------------------------------------------------------------


def _ribbon(ax, x0, x1, y0, y1, h, color, alpha):
    dx = (x1 - x0) / 2
    verts = [
        (x0, y0),
        (x0 + dx, y0),
        (x1 - dx, y1),
        (x1, y1),
        (x1, y1 + h),
        (x1 - dx, y1 + h),
        (x0 + dx, y0 + h),
        (x0, y0 + h),
        (x0, y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(
        patches.PathPatch(
            MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha
        )
    )


def plot_sankey(od, record, title, colors, labels, fontsize, out):
    """Origins (left) → destinations (right); import → export ribbons are transit."""
    # buffer tanks charge and discharge (~0.1 Mt/a, irrelevant for transit) – not drawn
    od = od[(od.origin != "co2 stored") & (od.destination != "co2 stored")]
    # node totals from the full table, so labels match the CSV; tiny ribbons and nodes are hidden
    left_tot = od.groupby("origin").value.sum()
    right_tot = od.groupby("destination").value.sum()
    left_tot, right_tot = (
        left_tot[left_tot > MIN_FLOW_MT],
        right_tot[right_tot > MIN_FLOW_MT],
    )
    od = od[
        (od.value > MIN_FLOW_MT)
        & od.origin.isin(left_tot.index)
        & od.destination.isin(right_tot.index)
    ]
    if od.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Keine CO$_2$-Flüsse in NRW", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    def order(tot, prefix):
        """Border nodes first (alphabetical), then local carriers by size."""
        border = sorted(s for s in tot.index if s.startswith(prefix))
        local = tot.drop(border).sort_values(ascending=False).index.tolist()
        return border + local

    left = order(left_tot, "Import ")
    right = order(right_tot, "Export ")
    total = left_tot.sum()
    gap = 0.03 * total
    min_slot = 0.065 * total  # room for the rotated value label of small nodes

    def positions(names, tot):
        pos, y = {}, 0.0
        for s in names:
            pos[s] = y
            y += max(tot[s], min_slot) + gap
        return pos, y - gap

    lpos, lh = positions(left, left_tot)
    rpos, rh = positions(right, right_tot)
    height = max(lh, rh)
    # centre the shorter column
    lpos = {k: v + (height - lh) / 2 for k, v in lpos.items()}
    rpos = {k: v + (height - rh) / 2 for k, v in rpos.items()}

    def color(s):
        if s.startswith(("Import ", "Export ")):
            return "#9e9e9e"
        return colors.get(s, "#cccccc")

    def label(s):
        return labels.get(s, s)

    def text_color(s):
        """Node colour, darkened if too light to read as text."""
        rgb = np.array(matplotlib.colors.to_rgb(color(s)))
        luminance = rgb @ [0.299, 0.587, 0.114]
        return tuple(rgb * 0.6) if luminance > 0.55 else tuple(rgb)

    fig, ax = plt.subplots(figsize=(9, 6))
    x0, x1, nw = 0.0, 1.0, 0.03
    lofs = {s: 0.0 for s in left}
    rofs = {s: 0.0 for s in right}
    od = od.assign(
        o=od.origin.map(left.index), d=od.destination.map(right.index)
    ).sort_values(["o", "d"])
    for _, f in od.iterrows():
        transit = f.origin.startswith("Import ") and f.destination.startswith("Export ")
        c = TRANSIT_COLOR if transit else color(f.origin)
        _ribbon(
            ax,
            x0 + nw,
            x1,
            lpos[f.origin] + lofs[f.origin],
            rpos[f.destination] + rofs[f.destination],
            f.value,
            c,
            0.75 if transit else 0.35,
        )
        lofs[f.origin] += f.value
        rofs[f.destination] += f.value

    # value rotated next to the bar (in the node colour), name further out
    for names, pos, tot, x_val, x_name, ha in [
        (left, lpos, left_tot, x0 - 0.008, x0 - 0.065, "right"),
        (right, rpos, right_tot, x1 + nw + 0.008, x1 + nw + 0.065, "left"),
    ]:
        for s in names:
            x_bar = x0 if ha == "right" else x1
            y = pos[s] + tot[s] / 2
            ax.add_patch(
                patches.Rectangle(
                    (x_bar, pos[s]), nw, tot[s], facecolor=color(s), edgecolor="none"
                )
            )
            ax.text(
                x_val,
                y,
                f"{tot[s]:.1f}",
                rotation=90,
                ha=ha,
                va="center",
                fontsize=fontsize,
                color=text_color(s),
                fontweight="bold",
            )
            ax.text(x_name, y, label(s), ha=ha, va="center", fontsize=fontsize)

    ax.text(
        x0 + nw / 2,
        -gap,
        "Von",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
    )
    ax.text(
        x1 + nw / 2,
        -gap,
        "Nach",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
    )
    ax.set_title(
        f"{title}\nTransit: {record['transit_trace']:.1f} Mt/a (Flow Tracing), "
        f"Bandbreite {record['transit_min']:.1f}–{record['transit_max']:.1f} Mt/a",
        fontsize=fontsize + 1,
    )
    ax.set_xlim(-0.6, 1.65)
    ax.set_ylim(height + gap, -3 * gap)  # top-down
    ax.set_axis_off()
    ax.text(
        0.5,
        -0.02,
        "MtCO$_2$/a",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize - 1,
        color="#555555",
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if out.endswith(".png"):
        fig.savefig(out[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_co2_transit_nrw",
            configfiles=["config/config.nrw.yaml"],
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    poster = get_poster(snakemake)
    settings = merge_settings(snakemake.params.plotting_fig, poster, "co2_transit_nrw")
    tech_colors = snakemake.params.tech_colors
    carrier_german = snakemake.params.carrier_german
    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    run_order = [
        r for r in settings["run_order"] if r in snakemake.config["run"]["name"]
    ]
    nice_names = settings["nice_names"]
    fontsize = settings["font"]["size"]
    plt.rc("font", **settings["font"])

    agg_label = settings["aggregate_label"]
    rename = (
        {c: agg_label for c in ["gas for industry CC", "process emissions CC"]}
        if settings.get("aggregate_industry_emissions", False)
        else {}
    )
    colors = {
        **tech_colors,
        agg_label: settings["aggregate_color"],
        "co2 stored": "#d9d9d9",
        LOSS_LABEL: "#d9d9d9",
    }
    labels = {**carrier_german, "co2 stored": "Zwischenspeicher"}

    if "summary" in snakemake.input.keys():
        # poster variant: re-plot from the existing summary, no network reads
        summary = pd.read_csv(snakemake.input.summary, dtype={"planning_horizon": str})
        od = None
    else:
        summary, od = build_tables(
            snakemake.input.networks,
            run_order,
            planning_horizons,
            settings["neighbour_names"],
            rename,
            settings.get("region_prefix", REGION_PREFIX),
        )
        summary.to_csv(snakemake.output.summary, index=False)
        od.to_csv(snakemake.output.od, index=False)

    plot_dashboard(
        summary,
        planning_horizons,
        run_order,
        nice_names,
        fontsize,
        ast.literal_eval(settings["figsize"]),
        settings["dpi"],
        snakemake.output.dashboard,
        snakemake.output.dashboard_png,
        poster,
        settings.get("legend_ncol", 4),
    )

    for out in snakemake.output.get("sankey", []):
        run, year = re.search(
            r"co2_transit_nrw_sankey_(.+)_(\d{4})\.png$", Path(out).name
        ).groups()
        sub = od[(od.run == run) & (od.planning_horizon == year)]
        record = summary.set_index(["run", "planning_horizon"]).loc[(run, year)]
        title = f"CO$_2$-Bilanz NRW – Szenario {nice_names.get(run, run)}, {year}"
        plot_sankey(sub, record, title, colors, labels, fontsize, out)
