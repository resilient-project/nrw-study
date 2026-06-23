# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Create summary CSV files for all scenario runs including costs, capacities,
capacity factors, curtailment, energy balances, prices and other metrics.
Restricted to components whose location starts with "DE".
"""

import logging

import pandas as pd
import pypsa

from scripts._helpers import configure_logging, set_scenario_config

idx = pd.IndexSlice
logger = logging.getLogger(__name__)

OUTPUTS = [
    "costs",
    "capacities",
    "energy",
    "energy_balance",
    "capacity_factors",
    "metrics",
    "curtailment",
    "prices",
    "weighted_prices",
    "market_values",
    "nodal_costs",
    "nodal_capacities",
    "nodal_energy_balance",
    "nodal_capacity_factors",
]


def assign_carriers(n: pypsa.Network) -> None:
    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"


def assign_locations(n: pypsa.Network) -> None:
    for c in n.components[n.one_port_components]:
        if c.static.empty:
            continue
        c.static["location"] = c.static.bus.map(n.buses.location)

    for c in n.components[n.branch_components]:
        if c.static.empty:
            continue
        c_bus_cols = c.static.filter(regex="^bus")
        locs = c_bus_cols.apply(lambda c: c.map(n.buses.location)).sort_index(axis=1)
        # Use first location that is not "EU"; take "EU" if nothing else available
        c.static["location"] = locs.apply(
            lambda row: next(
                (loc for loc in row.dropna() if loc != "EU"),
                "EU",
            ),
            axis=1,
        )


def to_de(result: pd.Series) -> pd.Series:
    """Filter a statistics result to DE locations and sum over the location level."""
    loc = result.index.get_level_values("location")
    result = result[loc.str.startswith("DE", na=False)]
    remaining = [l for l in result.index.names if l != "location"]
    return result.groupby(level=remaining).sum()


def calculate_nodal_capacity_factors(n: pypsa.Network) -> pd.Series:
    comps = n.one_port_components ^ {"Store"} | n.passive_branch_components
    result = n.statistics.capacity_factor(comps=comps, groupby=["location", "carrier"])
    loc = result.index.get_level_values("location")
    return result[loc.str.startswith("DE", na=False)]


def calculate_capacity_factors(n: pypsa.Network) -> pd.Series:
    comps = n.one_port_components ^ {"Store"} | n.passive_branch_components
    cf = n.statistics.capacity_factor(comps=comps, groupby=["location", "carrier"])
    cap = n.statistics.optimal_capacity(comps=comps, groupby=["location", "carrier"])

    # Restrict to the intersection — don't fill missing CFs with 0
    common = cf.index.intersection(cap.index)
    cf_de = cf.loc[common]
    cap_de = cap.loc[common]
    de_mask = cf_de.index.get_level_values("location").str.startswith("DE", na=False)
    cf_de, cap_de = cf_de[de_mask], cap_de[de_mask]
    remaining = [l for l in cf_de.index.names if l != "location"]

    # Capacity-weighted mean over DE locations
    return (
        (cf_de * cap_de).groupby(level=remaining).sum()
        / cap_de.groupby(level=remaining).sum()
    ).sort_index()


def calculate_nodal_costs(n: pypsa.Network) -> pd.Series:
    grouper = ["location", "carrier"]
    costs = pd.concat(
        {
            "capital": n.statistics.capex(groupby=grouper),
            "marginal": n.statistics.opex(groupby=grouper),
        }
    )
    costs.index.names = ["cost", "component", "location", "carrier"]
    loc = costs.index.get_level_values("location")
    return costs[loc.str.startswith("DE", na=False)]


def calculate_costs(n: pypsa.Network) -> pd.Series:
    grouper = ["location", "carrier"]
    costs = pd.concat(
        {
            "capital": n.statistics.capex(groupby=grouper),
            "marginal": n.statistics.opex(groupby=grouper),
        }
    )
    costs.index.names = ["cost", "component", "location", "carrier"]
    return to_de(costs)


def calculate_nodal_capacities(n: pypsa.Network) -> pd.Series:
    result = n.statistics.optimal_capacity(groupby=["location", "carrier"])
    loc = result.index.get_level_values("location")
    return result[loc.str.startswith("DE", na=False)]


def calculate_capacities(n: pypsa.Network) -> pd.Series:
    return to_de(n.statistics.optimal_capacity(groupby=["location", "carrier"]))


def calculate_curtailment(n: pypsa.Network) -> pd.Series:
    carriers = ["AC", "low voltage"]
    duration = n.snapshot_weightings.generators.sum()

    curtailed_abs = n.statistics.curtailment(
        bus_carrier=carriers,
        aggregate_across_components=True,
        groupby=["location", "carrier"],
    )
    available = (
        n.statistics.optimal_capacity(
            "Generator", bus_carrier=carriers, groupby=["location", "carrier"]
        )
        * duration
    )

    curtailed_abs, available = curtailed_abs.align(available, fill_value=0)
    return (to_de(curtailed_abs) / to_de(available) * 100).sort_index()


def calculate_energy(n: pypsa.Network) -> pd.Series:
    return to_de(
        n.statistics.energy_balance(groupby=["carrier", "location"])
    ).sort_values(ascending=False)


def calculate_energy_balance(n: pypsa.Network) -> pd.Series:
    return to_de(
        n.statistics.energy_balance(groupby=["carrier", "location", "bus_carrier"])
    ).sort_values(ascending=False)


def calculate_nodal_energy_balance(n: pypsa.Network) -> pd.Series:
    result = n.statistics.energy_balance(groupby=["carrier", "location", "bus_carrier"])
    loc = result.index.get_level_values("location")
    return result[loc.str.startswith("DE", na=False)]


def calculate_metrics(n: pypsa.Network) -> pd.Series:
    metrics = {}

    dc_links = n.links.query("carrier == 'DC'")
    if "location" in dc_links.columns:
        dc_links = dc_links[dc_links["location"].str.startswith("DE", na=False)]
    metrics["line_volume_DC"] = dc_links.eval("length * p_nom_opt").sum()

    lines = n.lines
    if "location" in lines.columns:
        lines = lines[lines["location"].str.startswith("DE", na=False)]
    metrics["line_volume_AC"] = lines.eval("length * s_nom_opt").sum()
    metrics["line_volume"] = metrics["line_volume_AC"] + metrics["line_volume_DC"]

    metrics["total costs"] = (
        to_de(n.statistics.capex(groupby=["location", "carrier"])).sum()
        + to_de(n.statistics.opex(groupby=["location", "carrier"])).sum()
    )

    ac_de_buses = n.buses.query("carrier == 'AC'").index
    ac_de_buses = ac_de_buses[
        n.buses.loc[ac_de_buses, "location"].str.startswith("DE", na=False)
    ]
    prices = n.buses_t.marginal_price[ac_de_buses]

    zero_hours = prices.where(prices < 0.1).count().sum()
    metrics["electricity_price_zero_hours"] = zero_hours / prices.size
    metrics["electricity_price_mean"] = prices.unstack().mean()
    metrics["electricity_price_std"] = prices.unstack().std()

    if "lv_limit" in n.global_constraints.index:
        metrics["line_volume_limit"] = n.global_constraints.at["lv_limit", "constant"]
        metrics["line_volume_shadow"] = n.global_constraints.at["lv_limit", "mu"]

    if "CO2Limit" in n.global_constraints.index:
        metrics["co2_shadow"] = n.global_constraints.at["CO2Limit", "mu"]

    if "co2_sequestration_limit" in n.global_constraints.index:
        metrics["co2_storage_shadow"] = n.global_constraints.at[
            "co2_sequestration_limit", "mu"
        ]

    return pd.Series(metrics).sort_index()


def calculate_prices(n: pypsa.Network) -> pd.Series:
    de = n.buses.index[n.buses.location.str.startswith("DE", na=False)]
    return (
        n.buses_t.marginal_price[de]
        .mean()
        .groupby(n.buses.loc[de].carrier)
        .mean()
        .sort_index()
    )


def calculate_weighted_prices(n: pypsa.Network) -> pd.Series:
    de = n.buses.index[n.buses.location.str.startswith("DE", na=False)]
    carriers = n.buses.loc[de].carrier.unique()

    weighted_prices = {}

    for carrier in carriers:
        de_carrier = de[n.buses.loc[de].carrier == carrier]

        load = n.statistics.withdrawal(
            groupby="bus",
            aggregate_time=False,
            bus_carrier=carrier,
            aggregate_across_components=True,
        ).T

        # Filter to DE buses
        de_cols = load.columns.intersection(de_carrier)
        if de_cols.empty:
            continue
        load = load[de_cols]

        if not load.empty and load.sum().sum() > 0:
            price = n.buses_t.marginal_price.loc[:, de_carrier]
            price = price.reindex(columns=load.columns, fill_value=1)

            weights = n.snapshot_weightings.generators
            a = weights @ (load * price).sum(axis=1)
            b = weights @ load.sum(axis=1)
            weighted_prices[carrier] = a / b

    return pd.Series(weighted_prices).sort_index()


def calculate_market_values(n: pypsa.Network) -> pd.Series:
    result = n.statistics.market_value(
        bus_carrier="AC",
        aggregate_across_components=True,
        groupby=["location", "carrier"],
    )
    energy = n.statistics.supply(
        bus_carrier="AC",
        aggregate_across_components=True,
        groupby=["location", "carrier"],
    )

    result, energy = result.align(energy, fill_value=0)
    de_mask = result.index.get_level_values("location").str.startswith("DE", na=False)
    result_de, energy_de = result[de_mask], energy[de_mask]
    remaining = [l for l in result_de.index.names if l != "location"]

    # Energy-weighted average market value over DE locations
    return (
        (result_de * energy_de).groupby(level=remaining).sum()
        / energy_de.groupby(level=remaining).sum()
    ).sort_values().dropna()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "make_summary_de",
            clusters="adm",
            opts="",
            sector_opts="",
            planning_horizons="2035",
            configfiles="config/config.nrw.yaml",
            run="endo-grid___Ref___offshore-co2"
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)
    assign_carriers(n)
    assign_locations(n)

    pypsa.set_option("params.statistics.nice_names", False)
    pypsa.set_option("params.statistics.drop_zero", False)

    for output in OUTPUTS:
        globals()["calculate_" + output](n).to_csv(snakemake.output[output])
