"""Build docs/case_study_base_stations.html: cislunar servicing base-station case study.

    python docs/build_case_study.py            # uses cached computations if present
    python docs/build_case_study.py --rebuild  # recompute the route matrix (~20-30 min, parallel)
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

import cislunar as cl
from cislunar import cr3bp, servicing as SV

HERE = Path(__file__).parent
CACHE = HERE / "_cache" / "case_study.json"
TEMPLATE = HERE / "case_study_template.html"
OUT = HERE / "case_study_base_stations.html"
S = cl.EARTH_MOON
NS = (10, 100, 1000)


def log(m):
    print(m, flush=True)


def rnd(a, nd=0):
    return np.round(np.asarray(a, float), nd).tolist()


def xyz(a, nd=0):
    a = np.asarray(a)
    return {"x": rnd(a[:, 0], nd), "y": rnd(a[:, 1], nd), "z": rnd(a[:, 2], nd)}


def compute():
    t0 = time.time()
    nodes = SV.default_nodes()
    log("route matrix (parallel) ...")
    costs = SV.cost_matrix(cache=str(HERE / "_cache" / "routes.json"), verbose=True)
    log(f"  {len(costs)} routes in {time.time() - t0:.0f}s")

    for c in SV.CLIENT_KEYS:  # direct-architecture leg (client's own plane for Earth orbits)
        costs[f"direct->{c}"] = {"total_mps": SV.direct_dv_mps(c, costs), "method": "direct"}
    log("symmetry spot checks (ADR-22) ...")
    checks = []
    for a, b in (("dro", "nrho"), ("nrho", "geo"), ("l2h", "l1h")):  # matrix values are mirrors
        direct = SV.node_transfer(nodes[a], nodes[b])
        checks.append({"route": f"{a}->{b}", "direct_mps": direct.total_mps,
                       "matrix_mps": costs[f"{a}->{b}"]["total_mps"],
                       "matrix_note": costs[f"{a}->{b}"].get("note", "")})
        log(f"  {a}->{b}: direct {direct.total_mps:.1f} vs matrix {costs[f'{a}->{b}']['total_mps']:.1f}")

    log("route trajectories ...")
    routes = {}
    for k, r in costs.items():
        if r["method"] in ("cr3bp", "coorbital") and r.get("x_post"):
            tr = cr3bp.propagate(np.array(r["x_post"]), r["tof"], S, n_points=400)
            routes[k] = {"earth": xyz(S.km(tr.r - S.pos1)), "moon": xyz(S.km(tr.r - S.pos2)),
                         "t_d": rnd(tr.t_days, 3)}

    node_out = {}
    for k, n in nodes.items():
        node_out[k] = {"label": n.label, "category": n.category, "period_d": n.period_days,
                       "periodic": n.periodic, "earth": xyz(n.trajectory_km(500, "earth")),
                       "moon": xyz(n.trajectory_km(500, "moon"))}

    log("default campaign tables ...")
    mm = SV.MassModel()
    table = {}
    for c in SV.CLIENT_KEYS:
        row = {"direct": {str(n): SV.campaign(n, c, None, costs, mm) for n in NS}}
        for st in SV.STATION_KEYS:
            row[st] = {str(n): SV.campaign(n, c, st, costs, mm) for n in NS}
            row[st]["break_even"] = SV.break_even(c, st, costs, mm)
        table[c] = row
    log(f"done in {time.time() - t0:.0f}s")
    return {"costs": costs, "routes": routes, "nodes": node_out, "checks": checks,
            "stations": SV.STATION_KEYS, "clients": SV.CLIENT_KEYS, "table": table,
            "mass_model": vars(mm), "eps_deg": SV.EPS_DEG, "ns": list(NS)}


def main():
    if CACHE.exists() and "--rebuild" not in sys.argv:
        log(f"using cached data {CACHE}")
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        if "--tables" in sys.argv:  # refresh direct legs + default tables without recomputing routes
            costs = data["costs"]
            for c in SV.CLIENT_KEYS:
                costs.pop(f"direct->{c}", None)
                costs[f"direct->{c}"] = {"total_mps": SV.direct_dv_mps(c, costs), "method": "direct"}
            mm = SV.MassModel()
            data["table"] = {c: dict({"direct": {str(n): SV.campaign(n, c, None, costs, mm) for n in NS}},
                                     **{st: dict({str(n): SV.campaign(n, c, st, costs, mm) for n in NS},
                                                 break_even=SV.break_even(c, st, costs, mm))
                                        for st in SV.STATION_KEYS}) for c in SV.CLIENT_KEYS}
            CACHE.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    else:
        data = compute()
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    if not TEMPLATE.exists():
        log("no template yet; data cached")
        return
    from plotly.offline import get_plotlyjs

    tpl = TEMPLATE.read_text(encoding="utf-8").replace("__VERSION__", cl.__version__)
    manual = (HERE / "manual_template.html").read_text(encoding="utf-8")
    css = manual[manual.index("<style>") + len("<style>"):manual.index("</style>")]  # shared look
    tpl = tpl.replace("/*__CSS__*/", css)
    tpl = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    OUT.write_text(tpl.replace("/*__PLOTLY__*/", get_plotlyjs()), encoding="utf-8")
    log(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
