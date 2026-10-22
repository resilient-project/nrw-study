# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Hand-calculated toy networks for scripts/plot_co2_transit_nrw.py.

Region nodes are prefixed "DEA", everything else is outside. Flows are set
directly in links_t (no optimisation), values in t/h; results are in Mt,
so expectations are scaled by 1e6.
"""

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.plot_co2_transit_nrw import compute_transit


def _network(nodes, links, flows, weights, efficiency=None, stores=None):
    """
    Build a network from (bus0, bus1, carrier) links and a dict of p0 series.

    Nodes in ``nodes`` get a "co2 stored" bus; any other link endpoint is a
    plain bus at the location given by its first word.
    """
    n = pypsa.Network()
    n.set_snapshots(range(len(weights)))
    n.snapshot_weightings.loc[:, :] = np.repeat(
        np.asarray(weights, float)[:, None], 3, axis=1
    )
    for node in nodes:
        n.add("Bus", f"{node} co2 stored", carrier="co2 stored", location=node)

    def bus(name):
        if name in nodes:
            return f"{name} co2 stored"
        if name not in n.buses.index:
            n.add("Bus", name, location=name.split()[0])
        return name

    for name, (b0, b1, carrier) in zip(flows, links):
        n.add(
            "Link",
            name,
            bus0=bus(b0),
            bus1=bus(b1),
            carrier=carrier,
            efficiency=(efficiency or {}).get(name, 1.0),
        )
    p0 = pd.DataFrame(flows, dtype=float)
    n.links_t.p0 = p0
    n.links_t.p1 = -p0.mul(n.links.efficiency[p0.columns], axis=1)
    for name, (node, p) in (stores or {}).items():
        n.add("Store", name, bus=f"{node} co2 stored", carrier="co2 stored")
        n.stores_t.p[name] = pd.Series(p, index=n.snapshots, dtype=float)
    record, od = compute_transit(n, {}, {})
    return {k: v * 1e6 for k, v in record.items() if not k.startswith("share")}, od


PIPE = "CO2 pipeline"


def _check(record, expected):
    for key, value in expected.items():
        assert record[key] == pytest.approx(value, abs=1e-6), key


def test_diamond_parallel_paths():
    # import 10 → A; A→B 5, A→C 5; B captures 5; B→D 10, C→D 5; D exports 12, uses 3
    # D mixes 10 import / 5 local → export 12 carries 8 import
    record, _ = _network(
        ["DEB", "DEA1", "DEA2", "DEA3", "DEA4", "NL"],
        [
            ("DEB", "DEA1", PIPE),
            ("DEA1", "DEA2", PIPE),
            ("DEA1", "DEA3", PIPE),
            ("DEA2", "DEA4", PIPE),
            ("DEA3", "DEA4", PIPE),
            ("DEA4", "NL", PIPE),
            ("DEA2 gas", "DEA2", "process emissions CC"),
            ("DEA4", "DEA4 meoh", "methanolisation"),
        ],
        {
            "i": [10],
            "ab": [5],
            "ac": [5],
            "bd": [10],
            "cd": [5],
            "x": [12],
            "cc": [5],
            "m": [3],
        },
        [1],
    )
    _check(
        record,
        {
            "I": 10,
            "O": 12,
            "S": 5,
            "D": 3,
            "transit_trace": 8,
            "transit_min": 7,
            "transit_max": 10,
        },
    )


def test_internal_loop():
    # A→B 10, B→A 4; import 10 → A, A uses 4; B captures 2, exports 8
    # 14a = 10 + 4b, 12b = 10a → b = 0.78125 → transit 8b = 6.25
    record, _ = _network(
        ["DEB", "DEA1", "DEA2", "NL"],
        [
            ("DEB", "DEA1", PIPE),
            ("DEA1", "DEA2", PIPE),
            ("DEA2", "DEA1", PIPE),
            ("DEA2", "NL", PIPE),
            ("DEA2 gas", "DEA2", "process emissions CC"),
            ("DEA1", "DEA1 meoh", "methanolisation"),
        ],
        {"i": [10], "ab": [10], "ba": [4], "x": [8], "cc": [2], "m": [4]},
        [1],
    )
    _check(
        record,
        {"I": 10, "O": 8, "transit_trace": 6.25, "transit_min": 6, "transit_max": 8},
    )


def test_internal_pipeline_losses():
    # internal pipe sends 10, delivers 9; losses count as regional withdrawal
    record, od = _network(
        ["DEB", "DEA1", "DEA2", "NL"],
        [("DEB", "DEA1", PIPE), ("DEA1", "DEA2", PIPE), ("DEA2", "NL", PIPE)],
        {"i": [10], "ab": [10], "x": [9]},
        [1],
        efficiency={"ab": 0.9},
    )
    _check(
        record,
        {
            "I": 10,
            "O": 9,
            "D": 1,
            "losses": 1,
            "transit_trace": 9,
            "transit_min": 9,
            "transit_max": 9,
        },
    )
    assert od.to_numpy().sum() * 1e6 == pytest.approx(10)


def test_store_and_snapshot_weights():
    # t0 (w=2): capture 5, store charges 2, export 3; t1 (w=1): store discharges 2, export 2
    record, _ = _network(
        ["DEA1", "NL"],
        [("DEA1", "NL", PIPE), ("DEA1 gas", "DEA1", "process emissions CC")],
        {"x": [3, 2], "cc": [5, 0]},
        [2, 1],
        stores={"st": ("DEA1", [-2, 2])},
    )
    _check(
        record, {"S": 12, "D": 4, "O": 8, "I": 0, "transit_trace": 0, "transit_max": 0}
    )


def test_bidirectional_border():
    # same neighbour, two lines in opposite directions: in 6, out 4, region uses 2
    record, od = _network(
        ["DEB", "DEA1"],
        [
            ("DEB", "DEA1", PIPE),
            ("DEA1", "DEB", PIPE),
            ("DEA1", "DEA1 meoh", "methanolisation"),
        ],
        {"a": [6], "b": [4], "m": [2]},
        [1],
    )
    _check(
        record, {"I": 6, "O": 4, "transit_trace": 4, "transit_min": 4, "transit_max": 4}
    )
    assert od.at["Import DEB", "Export DEB"] * 1e6 == pytest.approx(4)
