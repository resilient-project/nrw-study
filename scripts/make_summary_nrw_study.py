# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Calculates CO2 pipeline lengths per country and terrain (onshore/offshore)
for the NRW study, clipped by the respective region shapes.

Output columns: region, terrain, carrier, volume_mtpakm, length_km

Length is computed as haversine distance (bus0 → bus1) multiplied by
length_factor from config, then scaled by the fraction of the pipeline
geometry that falls within each clipped region.
"""

import geopandas as gpd
import pandas as pd
import pypsa
from pypsa.geo import haversine_pts
from shapely.geometry import LineString

from scripts._helpers import configure_logging, set_scenario_config


def co2_pipeline_lengths(
    n: pypsa.Network,
    regions_onshore: gpd.GeoDataFrame,
    regions_offshore: gpd.GeoDataFrame,
    length_factor: float = 1.0,
) -> pd.DataFrame:
    co2_carriers = ["CO2 pipeline", "CO2 pipeline short"]
    is_reversed = n.links.get("reversed", pd.Series(False, index=n.links.index)).fillna(False)
    pipe_links = n.links[
        n.links.carrier.isin(co2_carriers) & (n.links.p_nom_opt > 0) & ~is_reversed
    ].copy()

    if pipe_links.empty:
        return pd.DataFrame(columns=["region", "terrain", "carrier", "length_km", "volume_mtpakm"])

    pipe_links["geometry"] = [
        LineString([
            (n.buses.at[row.bus0, "x"], n.buses.at[row.bus0, "y"]),
            (n.buses.at[row.bus1, "x"], n.buses.at[row.bus1, "y"]),
        ])
        for _, row in pipe_links.iterrows()
    ]

    # Haversine distances as the reference length (same formula as prepare_sector_network)
    pipe_links["haversine_km"] = haversine_pts(
        n.buses.loc[pipe_links["bus0"].values, ["x", "y"]].values,
        n.buses.loc[pipe_links["bus1"].values, ["x", "y"]].values,
    )

    pipes_gdf = gpd.GeoDataFrame(
        pipe_links[["carrier", "p_nom_opt", "haversine_km", "geometry"]], crs="EPSG:4326"
    )
    # Full projected lengths for computing clip fractions
    pipes_gdf_m = pipes_gdf.to_crs("EPSG:3035")
    pipes_gdf_m["full_length_m"] = pipes_gdf_m.geometry.length

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
            # Proportion of each pipe that falls within this region (geometric)
            clip_frac = (
                clipped_m.geometry.length
                / pipes_gdf_m.loc[clipped_m.index, "full_length_m"]
            ).clip(0, 1)
            # Apply haversine × fraction × length_factor
            clipped_m["length_km"] = (
                pipes_gdf.loc[clipped_m.index, "haversine_km"] * clip_frac * length_factor
            )
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

    df = co2_pipeline_lengths(
        n, regions_onshore, regions_offshore,
        length_factor=snakemake.params.length_factor or 1.25,
    )
    df.to_csv(snakemake.output.co2_pipeline_length, index=False)
