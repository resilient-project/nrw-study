# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Helpers for poster variants of the NRW study figures.

Poster rules (``*_poster`` in rules/postprocess-nrw-study.smk) pass a
``poster`` param taken from ``plotting: nrw-study: poster`` in the config.
Scripts use it to

* override figure settings (font size, figsize, ...) per figure via
  :func:`merge_settings`, and
* translate all German figure text to English right before saving via
  :func:`translate_figure`.

Without a ``poster`` param every function here is a no-op, so the regular
(German) outputs are unchanged.
"""

import copy
import logging
import re

from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib.transforms import Bbox

logger = logging.getLogger(__name__)

# German -> English. Keys are matched against the full text of a matplotlib
# Text object (or one line of a multi-line text).
EN = {
    # carriers / legend groups
    "CO$_2$-Abscheidung": "CO$_2$ capture",
    "Nutzung": "Use",
    "Pipelines": "Pipelines",
    "PCI-PMI Projekte": "PCI-PMI projects",
    "Emissionen Industrie": "Industry emissions",
    "Industrie CO$_2$-Abscheidung": "Industry CO$_2$ capture",
    "Biomasse für Industrie": "Biomass for industry",
    "Biomasse-KWK": "Biomass CHP",
    "Dampfreformierung": "Steam methane reforming",
    "Direct Air Capture": "Direct air capture",
    "Methanolisierung": "Methanolisation",
    "Methanisierung": "Methanation",
    "Fischer-Tropsch": "Fischer-Tropsch",
    "Sabatier": "Sabatier",
    "CO$_2$-Sequestrierung": "CO$_2$ sequestration",
    "Gas für Industrie": "Gas for industry",
    "Prozessemissionen": "Process emissions",
    "Zwischenspeicher": "Buffer storage",
    "Leitungsverluste": "Pipeline losses",
    # cost groups
    "Sonstige": "Other",
    "Abfall": "Waste",
    "Übertragungsnetz Strom": "Electricity grid",
    "Therm. KW": "Thermal power plants",
    "Hydro & Laufwasser": "Hydro & run-of-river",
    "Biomasse": "Biomass",
    "Biomasse & -gas": "Biomass & biogas",
    "PtH": "Power-to-heat",
    "KWK": "CHP",
    "Solar": "Solar",
    "Wind": "Wind",
    "Onshore-Wind": "Onshore wind",
    "Offshore-Wind": "Offshore wind",
    "Batterie": "Battery",
    "Elektrolyse": "Electrolysis",
    "Methanol": "Methanol",
    "Ammoniak": "Ammonia",
    "Ammoniak (Haber-Bosch)": "Ammonia",
    "H$_2$-Infrastruktur": "H$_2$ infrastructure",
    "CO$_2$-Infrastruktur": "CO$_2$ infrastructure",
    "Lastabwurf": "Load shedding",
    "Kernkraft": "Nuclear",
    "Öl": "Oil",
    "Kohle": "Coal",
    "Punktquellen CC": "Point-source CC",
    # axis labels
    "Kostendifferenz zu 1.a": "Cost difference to 1.a",
    "(Mrd. € p.a.)": "(bn € p.a.)",
    "Gesamtsystemkosten": "Total system costs",
    "Auslastung (%)": "Utilisation (%)",
    "Pipelinelängen (km)": "Pipeline length (km)",
    "CO$_2$-Fluss über NRW-Grenze (Mt/a)": "CO$_2$ across NRW border (Mt/a)",
    # transit dashboard
    "Transit (durch NRW)": "Transit (through NRW)",
    "Import (Nutzung in NRW)": "Import (used in NRW)",
    "Export (aus Abscheidung in NRW)": "Export (captured in NRW)",
    "Bandbreite Transit": "Transit range",
    "Prozentwert: Transitanteil an CO$_2$-Importmenge": "Percentage: transit share of CO$_2$ imports",
    # NRW map neighbours
    "NIEDERLANDE": "NETHERLANDS",
    "BELGIEN": "BELGIUM",
    "RHEINLAND-PFALZ": "RHINELAND-PALATINATE",
    "HESSEN": "HESSE",
    "NIEDERSACHSEN": "LOWER SAXONY",
}

_IGNORE = re.compile(r"^[\s\d.,:%+\-–−km/$_{}()^\\a-zA-Z]{0,4}$")


def get_poster(snakemake) -> dict:
    """Return the poster settings of a rule, or an empty dict."""
    return dict(getattr(snakemake.params, "poster", None) or {})


def merge_settings(settings: dict, poster: dict, key: str) -> dict:
    """Deep-merge ``poster[key]`` into a copy of ``settings``."""
    out = copy.deepcopy(dict(settings))
    for k, v in (poster.get(key) or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def _translate(s: str, mapping: dict) -> str | None:
    if s in mapping:
        return mapping[s]
    if "\n" in s:
        lines = s.split("\n")
        new = [mapping.get(line, line) for line in lines]
        if new != lines:
            return "\n".join(new)
    return None


def translate_figure(fig, poster: dict) -> None:
    """Translate all German text of ``fig`` in place (poster only)."""
    if not poster or poster.get("language", "de") != "en":
        return
    mapping = {**EN, **(poster.get("translations") or {})}
    untranslated = set()
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s.strip():
            continue
        new = _translate(s, mapping)
        if new is not None:
            t.set_text(new)
        elif s not in mapping.values() and not _IGNORE.match(s):
            untranslated.add(s)
    if untranslated:
        logger.info("Poster: texts kept as is: %s", sorted(untranslated))


def save_split(fig, ax, output) -> None:
    """
    Save a map without the legends drawn below the axes to ``output.map`` /
    ``output.png`` and, if ``output`` has ``legend`` / ``legend_png``, those
    legends alone to separate files (poster layout with a shared legend).
    """
    fig.canvas.draw()
    # freeze the layout (matplotlibrc sets figure.autolayout), otherwise hiding
    # artists below moves the axes between the two saves
    fig.set_layout_engine("none")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bottom = ax.get_window_extent(renderer).y0
    # pypsa's legend helpers add legends to the figure, others sit on the axes
    legends = {
        id(c): c
        for c in [*ax.get_children(), *fig.get_children()]
        if isinstance(c, Legend)
    }.values()
    below = [
        leg for leg in legends if leg.get_window_extent(renderer).y1 <= ax_bottom + 1
    ]

    if "legend" in output.keys():
        bbox = Bbox.union([leg.get_window_extent(renderer) for leg in below])
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted()).padded(0.05)
        # hide everything else that might reach into the legend area
        # (keep the axes itself visible: legends placed on it are its children)
        others = [
            a
            for a in [*ax.get_children(), *fig.get_children()]
            if a is not ax and a not in below
        ]
        visible = [a.get_visible() for a in others]
        for a in others:
            a.set_visible(False)
        fig.savefig(output.legend, bbox_inches=bbox)
        fig.savefig(output.legend_png, bbox_inches=bbox, dpi=150)
        for a, v in zip(others, visible):
            a.set_visible(v)

    for leg in below:
        leg.set_visible(False)
    fig.savefig(output.map, bbox_inches="tight")
    fig.savefig(output.png, bbox_inches="tight", dpi=150)
