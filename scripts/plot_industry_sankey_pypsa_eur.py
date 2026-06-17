# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from scripts._helpers import configure_logging
from scripts.plot_industry_sankey_forecast import COLOR_MAPPING as BASE_COLOR_MAPPING
from scripts.plot_industry_sankey_forecast import nice_labels

product_mapper = {
    "Electric arc": "Metallerzeugung ",
    "DRI + Electric arc": "Metallerzeugung ",
    "Integrated steelworks": "Metallerzeugung ",
    "HVC": "Basic chemicals",
    "HVC (mechanical recycling)": "Basic chemicals",
    "HVC (chemical recycling)": "Basic chemicals",
    "Ammonia": "Basic chemicals",
    "Chlorine": "Basic chemicals",
    "Methanol": "Basic chemicals",
    "Other chemicals": "Other chemical industry",
    "Pharmaceutical products etc.": "Other chemical industry",
    "Cement": "Processing of stone and earth (non-metallic mineral processing)",
    "Ceramics & other NMM": "Glass and ceramics",
    "Glass production": "Glass and ceramics",
    "Pulp production": "Paper industry",
    "Paper production": "Paper industry",
    "Printing and media reproduction": "Paper industry",
    "Food, beverages and tobacco": "Food and tobacco",
    "Alumina production": "Non-ferrous metals and foundries",
    "Aluminium - primary production": "Non-ferrous metals and foundries",
    "Aluminium - secondary production": "Non-ferrous metals and foundries",
    "Other non-ferrous metals": "Non-ferrous metals and foundries",
    "Transport equipment": "Vehicle manufacturing (motor vehicles and transport equipment)",
    "Machinery equipment": "Machinery and equipment (mechanical engineering)",
    "Textiles and leather": "Other economic sectors",
    "Wood and wood products": "Other economic sectors",
    "Other industrial sectors": "Other economic sectors",
}

CARRIER_LABELS = {
    "elec": "Electricity",
    "coal": "Coal",
    "coke": "Coal",
    "biomass": "Biomass",
    "methane": "Natural gas",
    "hydrogen": "Hydrogen",
    "heat": "Heat",
    "naphtha": "Naphtha",
    "ammonia": "Ammonia",
    "methanol": "Methanol",
}

EXCLUDED_CARRIERS = {"process emission", "process emission from feedstock"}


def load_production_data(path, country):
    return pd.read_csv(path, index_col=0).loc[country]


def load_ratio_data(path, country):
    idx = pd.IndexSlice
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df = df.loc[:, idx[country, :]].drop(EXCLUDED_CARRIERS)
    df.columns = df.columns.get_level_values(1)
    df.index = df.index.map(lambda x: CARRIER_LABELS.get(x, x))
    return df


def compute_flows(prod, ratios):
    records = [pd.Series(ratios.loc[:, product] * kton / 1e3, name=product)
               for product, kton in prod.items()]
    flows = pd.concat(records, axis=1)
    flows = flows.T.groupby(product_mapper).sum().T
    return flows


def assign_colors(nodes):
    color_mapping = dict(BASE_COLOR_MAPPING)
    color_list = [mcolors.to_hex(c) for cmap in [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c] for c in cmap.colors]
    if len(nodes) > len(color_list):
        additional = len(nodes) - len(color_list)
        color_list.extend([
            mcolors.to_hex(mcolors.hsv_to_rgb((x / additional, 0.7, 0.9)))
            for x in range(additional)
        ])
    idx = 0
    for node in nodes:
        if node not in color_mapping:
            color_mapping[node] = color_list[idx % len(color_list)]
            idx += 1
    return color_mapping


def calc_positions(nodes, values):
    positions = {}
    y = 0.0
    total = sum(values.get(node, 0) for node in nodes)
    gap = total * 0.02 if len(nodes) > 1 else 0.0
    for node in nodes:
        h = values.get(node, 0)
        positions[node] = (y, h)
        y += h + gap
    return positions, y


def draw_nodes(ax, positions, x, align, width, colors):
    for node, (y, h) in positions.items():
        color = colors.get(node, "#cccccc")
        ax.bar(x, h, width=width, bottom=y, align="center", color=color, edgecolor="black", alpha=0.85)
        label = nice_labels.get(node, node)
        if align == "right":
            ax.text(x - 0.04, y + h / 2, label, ha="right", va="center", fontsize=10)
        else:
            ax.text(x + 0.04, y + h / 2, label, ha="left", va="center", fontsize=10)


def draw_links(ax, flows, pos_src, pos_tgt, x_src, x_tgt, width, colors):
    offsets_src = {node: 0.0 for node in pos_src}
    offsets_tgt = {node: 0.0 for node in pos_tgt}
    for _, flow in flows.sort_values(["source", "target"]).iterrows():
        src, tgt, val = flow["source"], flow["target"], flow["value"]
        if src not in pos_src or tgt not in pos_tgt:
            continue
        y_src, _ = pos_src[src]
        y_tgt, _ = pos_tgt[tgt]
        y_s = y_src + offsets_src[src]
        y_t = y_tgt + offsets_tgt[tgt]
        offsets_src[src] += val
        offsets_tgt[tgt] += val
        color = colors.get(src, "#999999")
        dx = x_tgt - x_src - width
        p1 = (x_src + width / 2, y_s + val)
        p4 = (x_tgt - width / 2, y_t + val)
        p2 = (p1[0] + dx / 2, p1[1])
        p3 = (p4[0] - dx / 2, p4[1])
        p5 = (x_tgt - width / 2, y_t)
        p8 = (x_src + width / 2, y_s)
        p6 = (p5[0] - dx / 2, p5[1])
        p7 = (p8[0] + dx / 2, p8[1])
        codes = [
            MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY,
        ]
        ax.add_patch(patches.PathPatch(MplPath([p1, p2, p3, p4, p5, p6, p7, p8, p1], codes),
                                       facecolor=color, alpha=0.4, edgecolor=None))


def add_legend(ax, max_height):
    if max_height <= 0:
        return
    magnitude = 10 ** int(np.log10(max_height) - 0.5)
    if max_height / magnitude > 15:
        magnitude *= 5
    elif max_height / magnitude > 8:
        magnitude *= 2
    ax.set_ylim(-max_height * 0.25, max_height * 1.05)
    legend_y = -max_height * 0.2
    ax.add_patch(patches.Rectangle(
        (0.45, legend_y), 0.05, magnitude,
        facecolor="gray", edgecolor="black", alpha=0.6, clip_on=False,
    ))
    ax.text(0.56, legend_y + magnitude / 2, f"{magnitude:.1f} TWh",
            ha="left", va="center", fontsize=12, fontweight="bold", clip_on=False)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_industry_sankey_pypsa_eur",
            planning_horizons="2045",
            configfiles=["config/config.nrw.yaml"],
            run="oge-grid___Ref___offshore-co2",
        )
    configure_logging(snakemake)

    country = snakemake.params.country
    planning_horizons = snakemake.wildcards.planning_horizons

    production = load_production_data(snakemake.input.production, country)
    ratios = load_ratio_data(snakemake.input.ratios, country)

    flows = compute_flows(production, ratios)
    flows = (
        flows.stack()
        .reset_index()
        .rename(columns={0: "value", "MWh/tMaterial": "source", "level_1": "target"})
    )
    flows = flows[flows["value"] > 0]

    carriers = sorted(flows["source"].unique())
    products = sorted(flows["target"].unique())

    pos_carriers, height_carriers = calc_positions(carriers, flows.groupby("source")["value"].sum().to_dict())
    pos_products, height_products = calc_positions(products, flows.groupby("target")["value"].sum().to_dict())

    max_height = max(height_carriers, height_products)
    colors = assign_colors(set(carriers) | set(products))

    fig, ax = plt.subplots(figsize=(18, 9))
    width = 0.05

    draw_nodes(ax, pos_carriers, 0, "right", width, colors)
    draw_nodes(ax, pos_products, 1, "left", width, colors)
    draw_links(ax, flows, pos_carriers, pos_products, 0, 1, width, colors)
    add_legend(ax, max_height)

    ax.set_xlim(-0.5, 1.5)
    ax.axis("off")
    ax.set_title(f"PyPSA-EUR Industry Energy Flow ({country}, {planning_horizons})", fontsize=16, pad=20)

    plt.tight_layout()
    fig.savefig(snakemake.output.sankey, bbox_inches="tight")
