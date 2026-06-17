# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path

from scripts._helpers import configure_logging

COLOR_MAPPING = {
    # --- Energy carriers (left column) ---
    "Waste non-RES": "#c97b7b",
    "Other fossil": "#7f7f7f",
    "Other RES": "#66c2a5",
    "Natural gas": "#9e9e9e",
    "Naphtha": "#984ea3",
    "Hydrogen": "#1abc9c",
    "Fuel oil": "#8c564b",
    "Electricity": "#377eb8",
    "District heating": "#e6550d",
    "Heat": "#e6550d",
    "Coal": "#4d4d4d",
    "Biomass": "#4daf4a",
    "Ambient heat": "#76c7c0",
    # --- Uses / demand types (middle column) ---
    "Space heating": "#ffe680",
    "Space cooling": "#c6dbef",
    "Raw material (feedstock) demand": "#fdd0a2",
    "Process heat (steam)": "#ffb347",
    "Process heat (industrial furnaces)": "#d73027",
    "Process cooling": "#9ecae1",
    "Mechanical and other electricity use": "#4a6fdc",
    "Energy balance calibration": "#bdbdbd",
    "Electrolysis (aluminium smelting)": "#2b8cbe",
    "Carbon capture and storage": "#525252",
    # --- Industrial sectors (right column) ---
    "Vehicle manufacturing (motor vehicles and transport equipment)": "#e41a1c",
    "Rubber and plastic products": "#ff7f00",
    "Quarrying of stone and earth; other mining": "#a65628",
    "Processing of stone and earth (non-metallic mineral processing)": "#d9b38c",
    "Paper industry": "#31a354",
    "Other economic sectors": "#969696",
    "Other chemical industry": "#984ea3",
    "Non-ferrous metals and foundries": "#6baed6",
    "Metallerzeugung ": "#ffd92f",
    "Machinery and equipment (mechanical engineering)": "#636363",
    "Glass and ceramics": "#fdbf6f",
    "Food and tobacco": "#b15928",
    "Fabricated metal products (metalworking)": "#cab2d6",
    "Basic chemicals": "#a6cee3",
}

nice_labels = {
    "Metallerzeugung ": "Metal production",
}


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_industry_sankey_forecast",
            forecast_scenario="Industrie-CCS",
            year="2045",
            configfiles=["config/config.nrw.yaml"],
            run="oge-grid___Ref___offshore-co2",
        )
    configure_logging(snakemake)

    mapping_file = snakemake.input.mapping
    demand_file = snakemake.input.demand
    forecast_scenario = snakemake.wildcards.forecast_scenario
    year = snakemake.wildcards.year
    output_file = snakemake.output.sankey

    mapping_df = pd.read_csv(mapping_file)
    demand_df = pd.read_csv(demand_file)

    link1 = demand_df.groupby(["Energy_carrier", "Application"])[year].sum().reset_index()
    link1.columns = ["source", "target", "value"]
    link1 = link1[link1["value"] > 0]

    link2 = demand_df.groupby(["Application", "Subsector"])[year].sum().reset_index()
    link2.columns = ["source", "target", "value"]
    link2 = link2[link2["value"] > 0]

    l0_nodes = sorted(link1["source"].unique())
    l1_nodes = sorted(pd.concat([link1["target"], link2["source"]]).unique())
    l2_nodes = sorted(link2["target"].unique())

    all_nodes = sorted(set(l0_nodes) | set(l1_nodes) | set(l2_nodes))

    available_cmaps = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c]
    color_list = []
    for cmap in available_cmaps:
        color_list.extend([mcolors.to_hex(c) for c in cmap.colors])

    if len(all_nodes) > len(color_list):
        import colorsys

        additional_needed = len(all_nodes) - len(color_list)
        hsv_colors = [
            colorsys.hsv_to_rgb(x / additional_needed, 0.7, 0.9)
            for x in range(additional_needed)
        ]
        color_list.extend([mcolors.to_hex(c) for c in hsv_colors])

    color_idx = 0
    for node in all_nodes:
        if node not in COLOR_MAPPING:
            COLOR_MAPPING[node] = color_list[color_idx % len(color_list)]
            color_idx += 1

    def calc_y(nodes, values_dict):
        pos = {}
        y = 0
        total = sum(values_dict.get(n, 0) for n in nodes)
        gap = total * 0.02 if len(nodes) > 1 else 0
        for node in nodes:
            h = values_dict.get(node, 0)
            pos[node] = (y, h)
            y += h + gap
        return pos, y

    l0_vals = link1.groupby("source")["value"].sum().to_dict()
    l1_in = link1.groupby("target")["value"].sum()
    l1_out = link2.groupby("source")["value"].sum()
    l1_vals = {
        node: max(l1_in.get(node, 0), l1_out.get(node, 0)) for node in l1_nodes
    }
    l2_vals = link2.groupby("target")["value"].sum().to_dict()

    pos0, max_y0 = calc_y(l0_nodes, l0_vals)
    pos1, max_y1 = calc_y(l1_nodes, l1_vals)
    pos2, max_y2 = calc_y(l2_nodes, l2_vals)
    max_h = max(max_y0, max_y1, max_y2)

    fig, ax = plt.subplots(figsize=(20, 10))
    x0, x1, x2 = 0, 1, 2
    width = 0.05

    def draw_nodes(pos, x, align):
        for node, (y, h) in pos.items():
            color = COLOR_MAPPING.get(node, "#cccccc")
            print_label = nice_labels.get(node, node)
            ax.bar(x, h, width=width, bottom=y, align="center", color=color, edgecolor="black", alpha=0.8)
            if align == "right":
                ax.text(x - 0.04, y + h / 2, print_label, ha="right", va="center", fontsize=10)
            elif align == "center":
                ax.text(
                    x, y + h / 2, print_label, ha="center", va="center", fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
                )
            elif align == "left":
                ax.text(x + 0.04, y + h / 2, print_label, ha="left", va="center", fontsize=10)

    def draw_links(links, pos_src, pos_tgt, x_src, x_tgt, width, color_by="source"):
        y_offsets_src = {n: 0.0 for n in pos_src}
        y_offsets_tgt = {n: 0.0 for n in pos_tgt}
        for _, row in links.sort_values(["source", "target"]).iterrows():
            src, tgt, val = row["source"], row["target"], row["value"]
            if src not in pos_src or tgt not in pos_tgt:
                continue
            y_src, _ = pos_src[src]
            y_tgt, _ = pos_tgt[tgt]
            y_s = y_src + y_offsets_src[src]
            y_t = y_tgt + y_offsets_tgt[tgt]
            y_offsets_src[src] += val
            y_offsets_tgt[tgt] += val
            color = COLOR_MAPPING.get(src if color_by == "source" else tgt, "#999999")
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
                Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY,
            ]
            path = Path([p1, p2, p3, p4, p5, p6, p7, p8, p1], codes)
            ax.add_patch(patches.PathPatch(path, facecolor=color, alpha=0.4, edgecolor=None))

    draw_nodes(pos0, x0, "right")
    draw_nodes(pos1, x1, "center")
    draw_nodes(pos2, x2, "left")
    draw_links(link1, pos0, pos1, x0, x1, width, color_by="source")
    draw_links(link2, pos1, pos2, x1, x2, width, color_by="target")

    magnitude = 10 ** int(np.log10(max_h) - 0.5)
    if magnitude == 0:
        magnitude = 1
    if max_h / magnitude > 15:
        magnitude *= 5
    elif max_h / magnitude > 8:
        magnitude *= 2

    magnitude /= 2

    ax.set_ylim(-max_h * 0.25, max_h * 1.05)
    legend_y = -max_h * 0.2
    ax.add_patch(
        patches.Rectangle(
            (x1 - 0.05, legend_y), 0.05, magnitude,
            facecolor="gray", edgecolor="black", alpha=0.6, clip_on=False,
        )
    )
    ax.text(
        x1 + 0.06, legend_y + magnitude / 2, f" {magnitude} TWh",
        ha="left", va="center", fontsize=12, fontweight="bold", clip_on=False,
    )

    ax.set_xlim(x0 - 0.5, x2 + 0.5)
    ax.axis("off")
    ax.set_title(f"FORECAST scenario: {forecast_scenario}, {year}", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
