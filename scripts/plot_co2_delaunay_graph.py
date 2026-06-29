# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Three-panel CO₂ Delaunay graph output:
  1. Europe – sequestration potential only
  2. Europe – sequestration potential + CO₂ links
  3. NRW zoom – CO₂ links (Mercator, same as other NRW map plots)
"""

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

from scripts._helpers import configure_logging, retry, set_scenario_config
from scripts.make_summary import assign_locations

FIGURE_SIZE = (4.5, 7)
NRW_BOUNDS = [5.75, 9.55, 50.2, 52.65]
EUROPE_BOUNDS = [-8.8, 26.0, 38.7, 67.0]
SEQ_COLS = [
    "conservative estimate Mt",
    "conservative estimate GAS Mt",
    "conservative estimate OIL Mt",
    "conservative estimate aquifer Mt",
]


def load_projection(plotting_params: dict) -> ccrs.CRS:
    proj_kwargs = dict(plotting_params.get("projection", {"name": "EqualEarth"}))
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)


def _base_map_opts(boundaries):
    opts = {k: v for k, v in map_opts.items() if k != "boundaries"}
    opts["boundaries"] = boundaries
    return opts


def _new_fig(proj_crs):
    return plt.subplots(figsize=FIGURE_SIZE, subplot_kw={"projection": proj_crs})


def _draw_regions(ax, proj_crs, shade_outside_nrw=False):
    regions_proj = regions.to_crs(proj_crs.proj4_init)
    if shade_outside_nrw:
        nrw_mask = regions_proj.index.str.startswith("DEA")
        regions_proj[~nrw_mask].plot(
            ax=ax, facecolor="#bbbbbb", edgecolor="#999999", linewidth=0.4, zorder=1
        )
        regions_proj[nrw_mask].plot(
            ax=ax, facecolor="#f5f5f5", edgecolor="#aaaaaa", linewidth=0.4, zorder=1
        )
    else:
        regions_proj.plot(
            ax=ax, facecolor="#f5f5f5", edgecolor="#aaaaaa", linewidth=0.4, zorder=1
        )


def _draw_seq(ax, proj_crs, seq_color, seq_alpha):
    seq_gdf.to_crs(proj_crs.proj4_init).plot(
        ax=ax, color=seq_color, alpha=seq_alpha, edgecolor="none", zorder=2
    )


def _draw_links(ax, plot_network, plot_links, link_colors, linewidth, bounds):
    plot_network.plot(
        geomap=True,
        geomap_color=False,
        bus_size=0,
        link_color=plot_links.carrier.map(link_colors),
        link_width=linewidth,
        branch_components=["Link"],
        ax=ax,
        **_base_map_opts(bounds),
    )
    for col in ax.collections:
        col.set_capstyle("round")
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def _finalise(ax, add_border=False):
    ax.set_facecolor("white")
    if add_border:
        ax.spines["geo"].set_linewidth(0.8)
        ax.spines["geo"].set_edgecolor("#bbbbbb")


def plot_europe_seq(seq_color, seq_alpha):
    """Europe map: sequestration layer only, no links."""
    fig, ax = _new_fig(proj_europe)
    _draw_regions(ax, proj_europe)
    _draw_seq(ax, proj_europe, seq_color, seq_alpha)
    # apply extent via a dummy n.plot call — use map_opts directly
    ax.set_extent(
        [EUROPE_BOUNDS[0], EUROPE_BOUNDS[1], EUROPE_BOUNDS[2], EUROPE_BOUNDS[3]],
        crs=ccrs.PlateCarree(),
    )
    _finalise(ax, add_border=True)
    return fig


def plot_europe_full(plot_network, plot_links, link_colors, linewidth, seq_color, seq_alpha):
    """Europe map: sequestration layer + CO₂ links."""
    fig, ax = _new_fig(proj_europe)
    _draw_regions(ax, proj_europe)
    _draw_seq(ax, proj_europe, seq_color, seq_alpha)
    _draw_links(ax, plot_network, plot_links, link_colors, linewidth, EUROPE_BOUNDS)
    _finalise(ax, add_border=True)
    return fig


def plot_nrw(plot_network, plot_links, link_colors, linewidth):
    """NRW zoom: CO₂ links, darker surrounding regions, subtle border."""
    fig, ax = _new_fig(proj_nrw)
    _draw_regions(ax, proj_nrw, shade_outside_nrw=True)
    _draw_links(ax, plot_network, plot_links, link_colors, linewidth, NRW_BOUNDS)
    _finalise(ax, add_border=True)
    return fig


@retry
def make_all_plots(n):
    plot_network = n.copy()
    assign_locations(plot_network)

    tech_colors = snakemake.params.plotting["tech_colors"]
    settings = snakemake.params.plotting["co2_delaunay_graph"]

    link_colors = {
        "CO2 pipeline": settings.get(
            "co2_pipeline_color", tech_colors.get("CO2 pipeline", "#f5627f")
        ),
        "CO2 pipeline short": settings.get(
            "co2_pipeline_short_color", tech_colors.get("CO2 pipeline short", "#c084fc")
        ),
    }
    linewidth = settings.get("linewidth", 1.5)
    seq_color = settings.get("sequestration_color", "#4a1870")
    seq_alpha = settings.get("sequestration_alpha", 0.4)

    is_reversed = plot_network.links.get(
        "reversed", pd.Series(False, index=plot_network.links.index)
    ).fillna(False)
    plot_links = plot_network.links.loc[
        plot_network.links.carrier.isin(link_colors) & ~is_reversed
    ].copy()

    relevant_buses = set(plot_links.bus0) | set(plot_links.bus1)
    plot_network.buses = plot_network.buses.loc[
        plot_network.buses.index.isin(relevant_buses)
    ].copy()
    plot_network.links = plot_links

    fig_seq = plot_europe_seq(seq_color, seq_alpha)
    fig_europe = plot_europe_full(plot_network, plot_links, link_colors, linewidth, seq_color, seq_alpha)
    fig_nrw = plot_nrw(plot_network, plot_links, link_colors, linewidth)

    return fig_seq, fig_europe, fig_nrw


def _save(fig, path_pdf, path_png):
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, bbox_inches="tight", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        _cfg = __import__(
            "yaml", fromlist=["safe_load"]
        ).safe_load(open("config/plotting.default.yaml"))
        _settings = _cfg["plotting"]["co2_delaunay_graph"]

        snakemake = mock_snakemake(
            "plot_co2_delaunay_graph",
            opts="",
            clusters="adm",
            sector_opts="",
            planning_horizons=str(_settings["planning_horizons"]),
            configfiles=["config/config.nrw.yaml"],
            run=_settings["run"],
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)
    regions = gpd.read_file(snakemake.input.regions).set_index("name")

    _seq_raw = gpd.read_file(snakemake.input.sequestration)
    _seq_sum = _seq_raw[SEQ_COLS].sum(axis=1, min_count=1)
    seq_gdf = _seq_raw[_seq_sum.notna() & (_seq_sum > 0)]

    map_opts = snakemake.params.plotting["map"]
    map_opts.pop("geomap_colors", None)

    proj_europe = load_projection(snakemake.params.plotting)
    proj_nrw = ccrs.Mercator()

    fig_seq, fig_europe, fig_nrw = make_all_plots(n)

    _save(fig_seq, snakemake.output.europe_seq_map, snakemake.output.europe_seq_png)
    _save(fig_europe, snakemake.output.europe_map, snakemake.output.europe_png)
    _save(fig_nrw, snakemake.output.nrw_map, snakemake.output.nrw_png)
