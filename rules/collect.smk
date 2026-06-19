# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT


localrules:
    all,
    cluster_networks,
    prepare_elec_networks,
    prepare_sector_networks,
    solve_elec_networks,
    solve_sector_networks,


rule process_costs:
    input:
        lambda w: (
            expand(
                resources(
                    f"costs_{config_provider('costs', 'year')(w)}_processed.csv"
                ),
                run=config["run"]["name"],
            )
            if config_provider("foresight")(w) == "overnight"
            else expand(
                resources("costs_{planning_horizons}_processed.csv"),
                **config["scenario"],
                run=config["run"]["name"],
            )
        ),


rule cluster_networks:
    input:
        expand(
            resources("networks/base_s_{clusters}.nc"),
            **config["scenario"],
            run=config["run"]["name"],
        ),
    message:
        "Collecting clustered network files"


rule prepare_elec_networks:
    input:
        expand(
            resources("networks/base_s_{clusters}_elec_{opts}.nc"),
            **config["scenario"],
            run=config["run"]["name"],
        ),
    message:
        "Collecting prepared electricity network files"


rule prepare_sector_networks:
    input:
        expand(
            resources(
                "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc"
            ),
            **config["scenario"],
            run=config["run"]["name"],
        ),
    message:
        "Collecting prepared sector-coupled network files"


rule solve_elec_networks:
    input:
        expand(
            RESULTS + "networks/base_s_{clusters}_elec_{opts}.nc",
            **config["scenario"],
            run=config["run"]["name"],
        ),
    message:
        "Collecting solved electricity network files"


rule solve_sector_networks:
    input:
        expand(
            RESULTS
            + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"],
            run=config["run"]["name"],
        ),
    message:
        "Collecting solved sector-coupled network files"


rule solve_sector_networks_perfect:
    input:
        expand(
            RESULTS
            + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-costs-all_{planning_horizons}.pdf",
            **config["scenario"],
            run=config["run"]["name"],
        ),
    message:
        "Collecting solved sector-coupled network files with perfect foresight"


def balance_map_paths(kind, w):
    """
    kind = "static" or "interactive"
    """
    cfg_key = "balance_map" if kind == "static" else "balance_map_interactive"

    return expand(
        RESULTS
        + f"maps/{kind}/base_s_{{clusters}}_{{opts}}_{{sector_opts}}_{{planning_horizons}}"
        f"-balance_map_{{carrier}}.{'pdf'if kind== 'static' else 'html'}",
        **config["scenario"],
        run=config["run"]["name"],
        carrier=config_provider("plotting", cfg_key, "bus_carriers")(w),
    )


RUN_PREFIX = config["run"].get("prefix", "")
SUMMARY_RESULTS = f"results/{RUN_PREFIX}/" if RUN_PREFIX else "results/"
CCS_FOCUS_MAP_SUMMARY = (
    SUMMARY_RESULTS
    + "nrw-study-summary/{run}/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
    + "-balance_map_co2_stored_focus_{focus}{variant}.pdf"
    if config["run"].get("scenarios", {}).get("enable")
    else SUMMARY_RESULTS
    + "nrw-study-summary/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
    + "-balance_map_co2_stored_focus_{focus}{variant}.pdf"
)


rule plot_balance_maps:
    input:
        static=lambda w: balance_map_paths("static", w),
        interactive=lambda w: balance_map_paths("interactive", w),
    message:
        "Plotting energy balance maps"


rule plot_balance_maps_static:
    input:
        lambda w: balance_map_paths("static", w),


rule plot_ccs_balance_maps_static:
    input:
        expand(
            RESULTS
            + "maps/static/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
            + "-balance_map_co2_stored.pdf",
            **config["scenario"],
            run=config["run"]["name"],
        ),
    message:
        "Plotting static CO2 stored balance maps for all configured NRW scenarios"


rule plot_ccs_balance_maps_focus_static:
    input:
        expand(
            CCS_FOCUS_MAP_SUMMARY,
            **config["scenario"],
            run=config["run"]["name"],
            focus=["nrw", "de"],
            variant=["", "_network_only"],
        ),
    message:
        "Plotting NRW and Germany focused CO2 stored balance maps for all configured NRW scenarios"


rule plot_balance_maps_interactive:
    input:
        lambda w: balance_map_paths("interactive", w),


rule plot_power_networks_clustered:
    input:
        expand(
            resources("maps/power-network-s-{clusters}.pdf"),
            **config["scenario"],
            run=config["run"]["name"],
        ),
    message:
        "Plotting clustered power network topology"
