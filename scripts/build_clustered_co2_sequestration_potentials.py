# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build regionalised geological sequestration potential for carbon dioxide using
data from `CO2Stop <https://setis.ec.europa.eu/european-co2-storage-
database_en>`_.
"""

import logging

import geopandas as gpd
import pandas as pd
from shapely.algorithms.polylabel import polylabel

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def area(gdf):
    """
    Returns area of GeoDataFrame geometries in square kilometers.
    """
    return gdf.to_crs(epsg=3035).area.div(1e6)


def allocate_sequestration_potential(
    gdf, regions, attr="conservative estimate Mt", threshold=3
):
    if isinstance(attr, str):
        attr = [attr]
    gdf = gdf.loc[gdf[attr].sum(axis=1) > threshold, attr + ["geometry"]]
    gdf["area_sqkm"] = area(gdf)
    overlay = gpd.overlay(regions, gdf, keep_geom_type=True)
    overlay["share"] = area(overlay) / overlay["area_sqkm"]
    adjust_cols = overlay.columns.difference(
        {"name", "offshore", "area_sqkm", "geometry", "share"}
    )
    overlay[adjust_cols] = overlay[adjust_cols].multiply(overlay["share"], axis=0)

    result = (
        overlay.dissolve(["name", "offshore"], aggfunc="sum")[attr]
        .sum(axis=1)
        .to_frame("potential")
    )

    regions_indexed = regions.set_index(["name", "offshore"])
    coordinates = regions_indexed.apply(
        lambda x: polylabel(x.geometry, tolerance=10), axis=1
    )
    result["x"] = coordinates.apply(lambda p: p.x)
    result["y"] = coordinates.apply(lambda p: p.y)
    return result


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_clustered_co2_sequestration_potentials",
            clusters="adm",
            configfiles=["config/config.nrw.yaml"],
            run="test",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    cf = snakemake.params.sequestration_potential

    gdf = gpd.read_file(snakemake.input.sequestration_potential)

    regions = gpd.read_file(snakemake.input.regions_offshore)
    regions["offshore"] = True

    if cf["include_onshore"]:
        onregions = gpd.read_file(snakemake.input.regions_onshore)
        onregions["offshore"] = False
        regions = (
            pd.concat([regions, onregions])
            .dissolve(by=["name", "offshore"])
            .reset_index()
        )

    s = allocate_sequestration_potential(
        gdf, regions, attr=cf["attribute"], threshold=cf["min_size"]
    )

    s = s[s["potential"] > cf["min_size"]]
    s.to_csv(snakemake.output.sequestration_potential)
