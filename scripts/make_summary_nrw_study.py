# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Calculates CO2 pipeline lengths per country and terrain (onshore/offshore)
for the NRW study, clipped by the respective region shapes.

Output columns: country, terrain, carrier, length_km
"""

import geopandas as gpd
import pandas as pd
import pypsa
from shapely.geometry import LineString

from scripts._helpers import configure_logging, set_scenario_config


def co2_pipeline_lengths(
    n: pypsa.Network,
    regions_onshore: gpd.GeoDataFrame,
    regions_offshore: gpd.GeoDataFrame,
) -> pd.DataFrame:
    co2_carriers = ["CO2 pipeline", "CO2 pipeline short"]
    pipe_links = n.links[
        n.links.carrier.isin(co2_carriers) & (n.links.p_nom_opt > 0)
    ].copy()

    if pipe_links.empty:
        return pd.DataFrame(columns=["country", "terrain", "carrier", "length_km", "volume_mtpakm"])

    pipe_links["geometry"] = [
        LineString([
            (n.buses.at[row.bus0, "x"], n.buses.at[row.bus0, "y"]),
            (n.buses.at[row.bus1, "x"], n.buses.at[row.bus1, "y"]),
        ])
        for _, row in pipe_links.iterrows()
    ]
    # p_nom_opt is kept so volume can be allocated proportionally to clipped length
    pipes_gdf = gpd.GeoDataFrame(pipe_links[["carrier", "p_nom_opt", "geometry"]], crs="EPSG:4326")

    records = []
    for terrain, regions in [("onshore", regions_onshore), ("offshore", regions_offshore)]:
        by_region = regions.copy()
        # Keep DEA (NRW) as its own region; group everything else by 2-char country code.
        by_region["region"] = by_region.index.map(
            lambda x: "DEA" if x.startswith("DEA") else x[:2]
        )
        by_region = by_region.dissolve(by="region").to_crs("EPSG:4326")

        for region, row in by_region.iterrows():
            clipped = gpd.clip(pipes_gdf, row.geometry)
            if clipped.empty:
                continue
            clipped_m = clipped.to_crs("EPSG:3035")
            clipped_m["length_km"] = clipped_m.geometry.length / 1000
            clipped_m["volume_mtpakm"] = clipped_m["length_km"] * clipped_m["p_nom_opt"]
            for carrier, grp in clipped_m.groupby("carrier"):
                records.append({
                    "region": region,
                    "terrain": terrain,
                    "carrier": carrier,
                    "volume_mtpakm": round(grp["volume_mtpakm"].sum(), 3),
                    "length_km": round(grp["length_km"].sum(), 3),
                })

    df = pd.DataFrame(records, columns=["region", "terrain", "carrier", "volume_mtpakm", "length_km"])
    return df.sort_values(
        ["region", "terrain", "carrier", "volume_mtpakm", "length_km"],
        ascending=[True, True, True, False, False],
    ).reset_index(drop=True)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "make_summary_nrw_study",
            opts="",
            clusters="adm",
            sector_opts="",
            planning_horizons="2045",
            configfiles=["config/config.nrw.yaml"],
            run="endo-grid___Ref___offshore-co2",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)

    regions_onshore = gpd.read_file(snakemake.input.regions_onshore).set_index("name")
    regions_offshore = gpd.read_file(snakemake.input.regions_offshore).set_index("name")

    df = co2_pipeline_lengths(n, regions_onshore, regions_offshore)
    df.to_csv(snakemake.output.co2_pipeline_length, index=False)
