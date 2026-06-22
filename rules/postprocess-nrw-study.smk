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
        plotting_fig=config_provider("plotting", "nrw-study", "costs_overview_delta"),
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


rule make_summary_nrw_study:
    input:
        network=RESULTS
        + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        regions_onshore=resources("regions_onshore_base_s_{clusters}.geojson"),
        regions_offshore=resources("regions_offshore_base_s_{clusters}.geojson"),
    output:
        co2_pipeline_length=RESULTS
        + "nrw-study/co2_pipeline_length_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
    log:
        RESULTS
        + "logs/make_summary_nrw_study/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.log",
    benchmark:
        (
            RESULTS
            + "benchmarks/make_summary_nrw_study/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        )
    threads: 1
    resources:
        mem_mb=4000,
    message:
        "Making NRW study CO2 pipeline summary for {wildcards.clusters} clusters, {wildcards.opts}, {wildcards.sector_opts}, {wildcards.planning_horizons}"
    script:
        scripts("make_summary_nrw_study.py")


rule plot_carbon_dioxide_network:
    input:
        network=RESULTS
        + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        regions=resources("regions_onshore_base_s_{clusters}.geojson"),
        lengths=RESULTS
        + "nrw-study/co2_pipeline_length_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
    output:
        map=RESULTS
        + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-co2_network_{planning_horizons}.pdf",
        png=RESULTS
        + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-co2_network_{planning_horizons}.png",
    log:
        RESULTS
        + "logs/plot_carbon_dioxide_network/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.log",
    benchmark:
        (
            RESULTS
            + "benchmarks/plot_carbon_dioxide_network/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        )
    threads: 2
    resources:
        mem_mb=10000,
    params:
        plotting=config_provider("plotting"),
    message:
        "Plotting carbon dioxide network for {wildcards.clusters} clusters, {wildcards.opts} electric options, {wildcards.sector_opts} sector options and {wildcards.planning_horizons} planning horizons"
    script:
        scripts("plot_carbon_dioxide_network.py")


rule plot_co2_pipeline_comparison:
    params:
        plotting_fig=config_provider("plotting", "nrw-study", "co2_pipeline_comparison"),
    input:
        csvs=expand(
            RESULTS + "nrw-study/co2_pipeline_length_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            run=config["run"]["name"],
            clusters=config["scenario"]["clusters"],
            opts=config["scenario"]["opts"],
            sector_opts=config["scenario"]["sector_opts"],
            planning_horizons=config["scenario"]["planning_horizons"],
        ),
    output:
        plot=NRW_RESULTS + "nrw-study-summary/co2_pipeline_comparison.pdf",
    log:
        NRW_RESULTS + "logs/plot_co2_pipeline_comparison.log",
    benchmark:
        NRW_RESULTS + "benchmark/plot_co2_pipeline_comparison",
    threads: 1
    resources:
        mem_mb=4000,
    message:
        "Plotting CO2 pipeline comparison across all NRW scenarios"
    script:
        scripts("plot_co2_pipeline_comparison.py")


rule animate_carbon_dioxide_network:
    input:
        maps=lambda wc: expand(
            RESULTS + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-co2_network_{planning_horizons}.png",
            run=wc.run,
            clusters=wc.clusters,
            opts=wc.opts,
            sector_opts=wc.sector_opts,
            planning_horizons=config["scenario"]["planning_horizons"],
        ),
    output:
        gif=RESULTS
        + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-co2_network_animation.gif",
    params:
        plotting=config_provider("plotting"),
    log:
        RESULTS
        + "logs/animate_carbon_dioxide_network/base_s_{clusters}_{opts}_{sector_opts}.log",
    benchmark:
        RESULTS
        + "benchmarks/animate_carbon_dioxide_network/base_s_{clusters}_{opts}_{sector_opts}",
    threads: 1
    resources:
        mem_mb=4000,
    message:
        "Animating CO2 network maps for {wildcards.clusters}, {wildcards.opts}, {wildcards.sector_opts}"
    script:
        scripts("animate_carbon_dioxide_network.py")


rule plot_carbon_dioxide_network_nrw:
    input:
        network=RESULTS
        + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        regions=resources("regions_onshore_base_s_{clusters}.geojson"),
        nuts3_shapes=resources("nuts3_shapes.geojson"),
    output:
        map=RESULTS
        + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-co2_network_nrw_{planning_horizons}.pdf",
        png=RESULTS
        + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-co2_network_nrw_{planning_horizons}.png",
    log:
        RESULTS
        + "logs/plot_carbon_dioxide_network_nrw/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.log",
    benchmark:
        (
            RESULTS
            + "benchmarks/plot_carbon_dioxide_network_nrw/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        )
    threads: 2
    resources:
        mem_mb=10000,
    params:
        plotting=config_provider("plotting"),
    message:
        "Plotting NRW carbon dioxide network for {wildcards.clusters} clusters, {wildcards.opts} electric options, {wildcards.sector_opts} sector options and {wildcards.planning_horizons} planning horizons"
    script:
        scripts("plot_carbon_dioxide_network_nrw.py")


rule plot_process_emissions_nrw:
    input:
        process_emissions=lambda wc: (
            "data/forecast_industry/"
            + config["industry"]["forecast_industry"]["scenario_mapping"][wc.run]
            + "/process_emissions.csv"
        ),
        regions=resources("regions_onshore_base_s_{clusters}.geojson"),
        nuts3_shapes=resources("nuts3_shapes.geojson"),
    output:
        map=RESULTS
        + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-process_emissions_nrw_{planning_horizons}.pdf",
        png=RESULTS
        + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-process_emissions_nrw_{planning_horizons}.png",
    params:
        plotting=config_provider("plotting"),
    log:
        RESULTS
        + "logs/plot_process_emissions_nrw/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.log",
    benchmark:
        (
            RESULTS
            + "benchmarks/plot_process_emissions_nrw/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        )
    threads: 1
    resources:
        mem_mb=4000,
    message:
        "Plotting process emissions NRW for {wildcards.planning_horizons}"
    script:
        scripts("plot_process_emissions_nrw.py")


rule plot_energy_demand_nrw:
    input:
        energy_demand=lambda wc: (
            "data/forecast_industry/"
            + config["industry"]["forecast_industry"]["scenario_mapping"][wc.run]
            + "/energy_demand.csv"
        ),
        regions=resources("regions_onshore_base_s_{clusters}.geojson"),
        nuts3_shapes=resources("nuts3_shapes.geojson"),
    output:
        map=RESULTS
        + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-energy_demand_nrw_{planning_horizons}.pdf",
        png=RESULTS
        + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-energy_demand_nrw_{planning_horizons}.png",
    params:
        plotting=config_provider("plotting"),
    log:
        RESULTS
        + "logs/plot_energy_demand_nrw/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.log",
    benchmark:
        (
            RESULTS
            + "benchmarks/plot_energy_demand_nrw/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        )
    threads: 1
    resources:
        mem_mb=8000,
    message:
        "Plotting energy demand NRW for {wildcards.planning_horizons}"
    script:
        scripts("plot_energy_demand_nrw.py")


rule animate_energy_demand_nrw:
    input:
        maps=lambda wc: expand(
            RESULTS + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-energy_demand_nrw_{planning_horizons}.png",
            run=wc.run,
            clusters=wc.clusters,
            opts=wc.opts,
            sector_opts=wc.sector_opts,
            planning_horizons=config["scenario"]["planning_horizons"],
        ),
    output:
        gif=RESULTS
        + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-energy_demand_nrw_animation.gif",
    params:
        plotting=config_provider("plotting"),
    log:
        RESULTS
        + "logs/animate_energy_demand_nrw/base_s_{clusters}_{opts}_{sector_opts}.log",
    benchmark:
        RESULTS
        + "benchmarks/animate_energy_demand_nrw/base_s_{clusters}_{opts}_{sector_opts}",
    threads: 1
    resources:
        mem_mb=4000,
    message:
        "Animating energy demand NRW for {wildcards.clusters}, {wildcards.opts}, {wildcards.sector_opts}"
    script:
        scripts("animate_carbon_dioxide_network.py")


rule animate_process_emissions_nrw:
    input:
        maps=lambda wc: expand(
            RESULTS + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-process_emissions_nrw_{planning_horizons}.png",
            run=wc.run,
            clusters=wc.clusters,
            opts=wc.opts,
            sector_opts=wc.sector_opts,
            planning_horizons=config["scenario"]["planning_horizons"],
        ),
    output:
        gif=RESULTS
        + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-process_emissions_nrw_animation.gif",
    params:
        plotting=config_provider("plotting"),
    log:
        RESULTS
        + "logs/animate_process_emissions_nrw/base_s_{clusters}_{opts}_{sector_opts}.log",
    benchmark:
        RESULTS
        + "benchmarks/animate_process_emissions_nrw/base_s_{clusters}_{opts}_{sector_opts}",
    threads: 1
    resources:
        mem_mb=4000,
    message:
        "Animating process emissions NRW for {wildcards.clusters}, {wildcards.opts}, {wildcards.sector_opts}"
    script:
        scripts("animate_carbon_dioxide_network.py")


rule animate_carbon_dioxide_network_nrw:
    input:
        maps=lambda wc: expand(
            RESULTS + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-co2_network_nrw_{planning_horizons}.png",
            run=wc.run,
            clusters=wc.clusters,
            opts=wc.opts,
            sector_opts=wc.sector_opts,
            planning_horizons=config["scenario"]["planning_horizons"],
        ),
    output:
        gif=RESULTS
        + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-co2_network_nrw_animation.gif",
    params:
        plotting=config_provider("plotting"),
    log:
        RESULTS
        + "logs/animate_carbon_dioxide_network_nrw/base_s_{clusters}_{opts}_{sector_opts}.log",
    benchmark:
        RESULTS
        + "benchmarks/animate_carbon_dioxide_network_nrw/base_s_{clusters}_{opts}_{sector_opts}",
    threads: 1
    resources:
        mem_mb=4000,
    message:
        "Animating NRW CO2 network maps for {wildcards.clusters}, {wildcards.opts}, {wildcards.sector_opts}"
    script:
        scripts("animate_carbon_dioxide_network.py")


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
        rules.plot_co2_pipeline_comparison.output.plot,
        expand(
            RESULTS
            + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-co2_network_{planning_horizons}.pdf",
            run=config["run"]["name"],
            clusters=config["scenario"]["clusters"],
            opts=config["scenario"]["opts"],
            sector_opts=config["scenario"]["sector_opts"],
            planning_horizons=config["scenario"]["planning_horizons"],
        ),
        expand(
            RESULTS
            + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-co2_network_nrw_{planning_horizons}.pdf",
            run=config["run"]["name"],
            clusters=config["scenario"]["clusters"],
            opts=config["scenario"]["opts"],
            sector_opts=config["scenario"]["sector_opts"],
            planning_horizons=config["scenario"]["planning_horizons"],
        ),
        expand(
            RESULTS
            + "nrw-study/co2_pipeline_length_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
            run=config["run"]["name"],
            clusters=config["scenario"]["clusters"],
            opts=config["scenario"]["opts"],
            sector_opts=config["scenario"]["sector_opts"],
            planning_horizons=config["scenario"]["planning_horizons"],
        ),
        expand(
            RESULTS
            + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-co2_network_animation.gif",
            run=config["run"]["name"],
            clusters=config["scenario"]["clusters"],
            opts=config["scenario"]["opts"],
            sector_opts=config["scenario"]["sector_opts"],
        ),
        expand(
            RESULTS
            + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-co2_network_nrw_animation.gif",
            run=config["run"]["name"],
            clusters=config["scenario"]["clusters"],
            opts=config["scenario"]["opts"],
            sector_opts=config["scenario"]["sector_opts"],
        ),
        expand(
            RESULTS
            + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-process_emissions_nrw_{planning_horizons}.pdf",
            run=config["run"]["name"],
            clusters=config["scenario"]["clusters"],
            opts=config["scenario"]["opts"],
            sector_opts=config["scenario"]["sector_opts"],
            planning_horizons=config["scenario"]["planning_horizons"],
        ),
        expand(
            RESULTS
            + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-process_emissions_nrw_animation.gif",
            run=config["run"]["name"],
            clusters=config["scenario"]["clusters"],
            opts=config["scenario"]["opts"],
            sector_opts=config["scenario"]["sector_opts"],
        ),
        expand(
            RESULTS
            + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-energy_demand_nrw_{planning_horizons}.pdf",
            run=config["run"]["name"],
            clusters=config["scenario"]["clusters"],
            opts=config["scenario"]["opts"],
            sector_opts=config["scenario"]["sector_opts"],
            planning_horizons=config["scenario"]["planning_horizons"],
        ),
        expand(
            RESULTS
            + "maps/static/base_s_{clusters}_{opts}_{sector_opts}-energy_demand_nrw_animation.gif",
            run=config["run"]["name"],
            clusters=config["scenario"]["clusters"],
            opts=config["scenario"]["opts"],
            sector_opts=config["scenario"]["sector_opts"],
        ),
    message:
        "Plotting all NRW study outputs"
