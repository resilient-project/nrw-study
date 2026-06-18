# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

FORECAST_INDUSTRY_CFG = config["industry"]["forecast_industry"]
NRW_RESULTS = "results/" + config["run"]["prefix"] + "/"


rule plot_industry_sankey_pypsa_eur:
    input:
        production=resources("industrial_production_per_country_tomorrow_{planning_horizons}.csv"),
        ratios=resources("industry_sector_ratios_{planning_horizons}.csv"),
    output:
        sankey=RESULTS + "nrw-study/industry_sankey_pypsa_eur_{planning_horizons}.pdf",
    params:
        country="DE",
    log:
        RESULTS + "logs/plot_industry_sankey_pypsa_eur_{planning_horizons}.log",
    benchmark:
        benchmarks("plot_industry_sankey_pypsa_eur_{planning_horizons}.json"),
    threads: 1
    resources:
        mem_mb=4000,
    message:
        "Plotting PyPSA-EUR industry sankey for {wildcards.planning_horizons}"
    script:
        scripts("plot_industry_sankey_pypsa_eur.py")


rule plot_industry_sankey_forecast:
    input:
        mapping="data/forecast_industry/mapping.csv",
        demand="data/forecast_industry/{forecast_scenario}/energy_demand.csv",
    output:
        sankey=RESULTS+"nrw-study/industry_sankey_forecast_{forecast_scenario}_{year}.pdf"
    log:
        RESULTS
        + "logs/plot_industry_sankey_forecast_{forecast_scenario}_{year}.log"
    benchmark:
        benchmarks("plot_industry_sankey_forecast_{forecast_scenario}_{year}.json")
    threads: 1
    resources:
        mem_mb=4000,
    message:
        "Plotting industry sankey diagram for {wildcards.forecast_scenario}, {wildcards.year}"
    script:
        scripts("plot_industry_sankey_forecast.py")


rule plot_industry_sankey_forecast_all:
    input:
        [
            expand(
                RESULTS + "nrw-study/industry_sankey_forecast_{forecast_scenario}_{year}.pdf",
                run=run_name,
                forecast_scenario=FORECAST_INDUSTRY_CFG["scenario_mapping"][run_name],
                year=config["scenario"]["planning_horizons"],
            )
            for run_name in config["run"]["name"]
            if run_name in FORECAST_INDUSTRY_CFG["scenario_mapping"]
        ]
    log:
        RESULTS + "logs/plot_industry_sankey_forecast_all.log"
    message:
        "Plotting industry sankey diagrams for all forecast scenarios and planning horizons"


rule plot_costs_overview:
    params:
        plotting_fig=config_provider("plotting", "nrw-study", "costs_overview"),
    input:
        costs=expand(
            RESULTS + "csvs/costs.csv",
            run=config["run"]["name"],
        ),
    output:
        plot=NRW_RESULTS + "nrw-study-summary/costs_overview.pdf",
    log:
        NRW_RESULTS + "logs/plot_costs_overview.log",
    benchmark:
        NRW_RESULTS + "benchmark/plot_costs_overview",
    threads: 1
    resources:
        mem_mb=4000,
    message:
        "Plotting costs overview across all NRW scenarios"
    script:
        scripts("plot_costs_overview.py")


rule plot_costs_overview_delta:
    params:
        plotting_fig=config_provider("plotting", "nrw-study", "costs_overview"),
    input:
        costs=expand(
            RESULTS + "csvs/costs.csv",
            run=config["run"]["name"],
        ),
    output:
        plot=NRW_RESULTS + "nrw-study-summary/costs_overview_delta.pdf",
    log:
        NRW_RESULTS + "logs/plot_costs_overview_delta.log",
    benchmark:
        NRW_RESULTS + "benchmark/plot_costs_overview_delta",
    threads: 1
    resources:
        mem_mb=4000,
    message:
        "Plotting delta costs overview across all NRW scenarios"
    script:
        scripts("plot_costs_overview_delta.py")


rule plot_nrw_study:
    input:
        expand(
            RESULTS + "nrw-study/industry_sankey_pypsa_eur_{planning_horizons}.pdf",
            run=config["run"]["name"],
            planning_horizons=config["scenario"]["planning_horizons"],
        ),
        rules.plot_industry_sankey_forecast_all.input if FORECAST_INDUSTRY_CFG["enable"] else [],
        rules.plot_costs_overview.output.plot,
        rules.plot_costs_overview_delta.output.plot,
    message:
        "Plotting all NRW study outputs"
