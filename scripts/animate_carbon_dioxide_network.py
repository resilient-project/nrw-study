# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Assemble per-planning-horizon CO2 network PNGs into an animated GIF.

Frame order is determined by the year extracted from each filename.
Frame duration is set via plotting.carbon_dioxide_network.gif_frame_duration (ms).
"""

import re

from PIL import Image

from scripts._helpers import configure_logging, set_scenario_config


def extract_year(path: str) -> int:
    m = re.search(r"(\d{4})\.png$", path)
    return int(m.group(1)) if m else 0


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "animate_carbon_dioxide_network",
            opts="",
            clusters="adm",
            sector_opts="",
            configfiles=["config/config.nrw.yaml"],
            run="endo-grid___Ref___offshore-co2",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    settings = snakemake.params.plotting["carbon_dioxide_network"]
    frame_duration = settings.get("gif_frame_duration", 700)

    sorted_paths = sorted(snakemake.input.maps, key=extract_year)
    frames = [Image.open(p) for p in sorted_paths]

    frames[0].save(
        snakemake.output.gif,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=True,
    )
