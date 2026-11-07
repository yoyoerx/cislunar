"""Build docs/manual.html: run the cislunar tools, embed the results, plotly.js,
example figures and an auto-generated API reference into one offline HTML file.

    python docs/build_manual.py            # uses cached computations if present
    python docs/build_manual.py --rebuild  # recompute everything (~5 min)
    python docs/build_manual.py --families # recompute only the orbit families
    python docs/build_manual.py --rosettes # recompute only the rosette orbits
    python docs/build_manual.py --transient # recompute manifolds/fates/flyby/TCM/SK data
"""
from __future__ import annotations

import base64
import html
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

import cislunar as cl
from cislunar import cr3bp, frames, periodic as P, relative as R, targeting as T
from cislunar.constants import R_EARTH, SYNODIC_MONTH_DAYS

HERE = Path(__file__).parent
ROOT = HERE.parent
CACHE = HERE / "_cache" / "manual_data.json"
TEMPLATE = HERE / "manual_template.html"
OUT = HERE / "manual.html"
FIGS = ROOT / "examples" / "output"
S = cl.EARTH_MOON


def log(msg):
    print(msg, flush=True)


def rnd(a, nd=1):
    return np.round(np.asarray(a, float), nd).tolist()


def km_moon(traj_r):
    """Rotating-frame positions -> Moon-centred km."""
    return S.km(np.asarray(traj_r) - S.pos2)


def xyz(arr_km, nd=0):
    a = np.asarray(arr_km)
    return {"x": rnd(a[:, 0], nd), "y": rnd(a[:, 1], nd), "z": rnd(a[:, 2], nd)}


def orbit_record(o: P.PeriodicOrbit, n=200):
    tr = o.trajectory(n)
    rp, ra = o.perilune_apolune_km()
    return {
        "state0": o.state0.tolist(), "period": o.period, "period_d": o.period_days,
        "rp": rp, "ra": ra, "nu": float(o.stability_index()), "C": o.jacobi,
        "path": xyz(km_moon(tr.r)),
    }


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def compute_families():
    fams = {}
    t = time.time()
    start = P.halo(2, 5000.0, "south")
    fam = P.continue_arclength(start, 0.01, stop=lambda o: o.period_days < 6.0,
                               direction=(0.0, 1.0, 0.0))
    fams["l2s"] = {"name": "L2 southern halo -> NRHO", "members": [orbit_record(o) for o in fam]}
    log(f"  L2 south halo family: {len(fam)} ({time.time() - t:.0f}s)")

    t = time.time()
    start = P.halo(1, 5000.0, "north")
    fam = P.continue_arclength(start, 0.01, stop=lambda o: o.period_days < 8.0
                               or o.perilune_apolune_km()[0] < 2500.0,
                               direction=(0.0, 1.0, 0.0), max_orbits=90)
    fams["l1n"] = {"name": "L1 northern halo -> NRHO", "members": [orbit_record(o) for o in fam]}
    log(f"  L1 north halo family: {len(fam)} ({time.time() - t:.0f}s)")

    t = time.time()
    xl1 = cr3bp.lagrange_points(S.mu)["L1"][0]
    # one continuous sweep (secant predictor throughout) so the family is not lost
    path = P.continue_to(P.lyapunov(1, 2000.0), P.IX, xl1 + S.nd_length(40000.0),
                         S.nd_length(500.0), free=(P.IVY,), targets=(P.IVX,), return_path=True)
    xs = np.array([o.state0[0] for o in path])
    picks = sorted({int(np.argmin(np.abs(xs - (xl1 + S.nd_length(a)))))
                    for a in np.arange(2000.0, 40001.0, 2000.0)})
    mem = [path[k] for k in picks]
    for o in mem:
        o.family = "L1 Lyapunov"
    fams["lyap"] = {"name": "L1 planar Lyapunov", "members": [orbit_record(o) for o in mem]}
    log(f"  L1 Lyapunov family: {len(mem)} ({time.time() - t:.0f}s)")

    t = time.time()
    orb = P.dro(10000.0)
    mem = [orb]
    for d in np.arange(15000.0, 100001.0, 5000.0):
        orb = P.continue_to(orb, P.IX, 1 - S.mu - S.nd_length(d), S.nd_length(3000.0),
                            free=(P.IVY,), targets=(P.IVX,))
        orb.family = "DRO"
        mem.append(orb)
    fams["dro"] = {"name": "Distant Retrograde Orbits", "members": [orbit_record(o) for o in mem]}
    log(f"  DRO family: {len(mem)} ({time.time() - t:.0f}s)")
    return fams


def compute_sandbox_starts():
    out = {}
    named = [
        ("nrho", "9:2 NRHO (Gateway)", P.nrho()),
        ("dro", "DRO, 70,000 km", P.dro(70000.0)),
        ("l1h", "L1 northern halo", P.halo(1, 15000.0, "north")),
        ("l2h", "L2 southern halo", P.halo(2, 20000.0, "south")),
        ("lyap", "L1 Lyapunov", P.lyapunov(1, 20000.0)),
    ]
    for key, label, o in named:
        out[key] = {"label": label, "state0": o.state0.tolist(), "period": o.period}
    # low lunar orbit: 100 km circular, polar to the Earth-Moon plane
    llo = T.keplerian_state_fn(S.r2 + 100.0, 0.0, np.radians(90.0), np.radians(90.0),
                               0.0, 0.0, S, center="moon")
    per = 2 * np.pi * np.sqrt((S.r2 + 100.0) ** 3 / S.gm2)
    out["llo"] = {"label": "Low lunar orbit, 100 km polar", "state0": llo(0.0).tolist(),
                  "period": float(S.nd_time_seconds(per))}
    leo = T.keplerian_state_fn(R_EARTH + 400.0, 0.0, np.radians(28.5), 0.0, 0.0, 0.0, S)
    per = 2 * np.pi * np.sqrt((R_EARTH + 400.0) ** 3 / S.gm1)
    out["leo"] = {"label": "Low Earth orbit, 400 km", "state0": leo(0.0).tolist(),
                  "period": float(S.nd_time_seconds(per))}
    return out


def compute_frames_demo():
    out = {}
    for key, o, revs in (("nrho", P.nrho(), 4), ("dro", P.dro(70000.0), 2),
                         ("l2h", P.halo(2, 20000.0, "south"), 2)):
        tf = o.period * revs
        tr = cr3bp.propagate(o.state0, tf, S, n_points=500)
        mi = frames.trajectory_to_inertial(tr, "moon")
        ei = frames.trajectory_to_inertial(tr, "earth")
        moon_e = frames.primary_track_inertial(tr.t, S, "moon", "earth")
        out[key] = {"t_d": rnd(tr.t_days, 3), "rot": xyz(km_moon(tr.r)),
                    "in_moon": xyz(S.km(mi.r)), "in_earth": xyz(S.km(ei.r)),
                    "moon_earth": xyz(S.km(moon_e))}
    return out


def compute_leo_transfer():
    nrho = P.nrho()
    inc = np.radians(28.5)
    tof0 = S.nd_time_days(4.5)
    r_arr = frames.rotating_to_inertial(tof0, nrho.state_at(tof0), S, "earth")[:3]
    raan = T.raan_for_target(r_arr, inc)[0]
    leo = T.keplerian_state_fn(R_EARTH + 400.0, 0.0, inc, raan, 0.0, 0.0, S)
    per = 2 * np.pi * np.sqrt((R_EARTH + 400.0) ** 3 / S.gm1)
    tr = T.optimize_transfer(leo, nrho, (0.0, S.nd_time_seconds(per)),
                             (S.nd_time_days(3.5), S.nd_time_days(5.5)), S, verbose=True)
    arc = tr.trajectory(1500)
    t_end = tr.t_depart + tr.tof
    nr = cr3bp.propagate(nrho.state_at(t_end), t_end + nrho.period, S, t0=t_end, n_points=300)
    arc_i = frames.trajectory_to_inertial(arc, "earth")
    nr_i = frames.trajectory_to_inertial(nr, "earth")
    tt = np.linspace(tr.t_depart, t_end + nrho.period, 300)
    moon_i = frames.primary_track_inertial(tt, S, "moon", "earth")
    # Earth-only (two-body) propagation of the same post-TLI state, inertial
    x_post = tr.x_chaser.copy()
    x_post[3:] += tr.dv1
    xi0 = S.state_to_dim(frames.rotating_to_inertial(tr.t_depart, x_post, S, "earth"))
    kep = []
    ts = np.linspace(0, S.seconds(tr.tof + nrho.period), 600)
    for dt in ts:
        r, _ = T.twobody.kepler_propagate(xi0[:3], xi0[3:], float(dt), S.gm1)
        kep.append(r)
    kep = np.array(kep)
    full = cr3bp.propagate(x_post, tr.t_depart + tr.tof + nrho.period, S, t0=tr.t_depart,
                           n_points=1500)
    full_i = frames.trajectory_to_inertial(full, "earth")
    tm = np.linspace(tr.t_depart, tr.t_depart + tr.tof + nrho.period, 300)
    moon_i2 = frames.primary_track_inertial(tm, S, "moon", "earth")
    e0 = -S.pos1
    return {
        "t_depart": tr.t_depart, "tof": tr.tof, "x_chaser": tr.x_chaser.tolist(),
        "dv1": tr.dv1.tolist(), "dv1_mps": tr.dv1_mps, "dv2_mps": tr.dv2_mps,
        "target_arrive": nrho.state_at(t_end).tolist(),
        "arc_rot": xyz(S.km(arc.r + e0)), "nrho_rot": xyz(S.km(nr.r + e0)),
        "arc_in": xyz(S.km(arc_i.r)), "nrho_in": xyz(S.km(nr_i.r)),
        "moon_in": xyz(S.km(moon_i)),
        "kep_in": xyz(kep), "cr3bp_in": xyz(S.km(full_i.r)), "moon_in2": xyz(S.km(moon_i2)),
        "nrho_state0": nrho.state0.tolist(), "nrho_period": nrho.period,
    }


def compute_porkchop():
    dro, nrho = P.dro(70000.0), P.nrho()
    t_dep = np.linspace(0.0, S.nd_time_days(14.0), 22)
    tofs = np.linspace(S.nd_time_days(2.0), S.nd_time_days(9.0), 12)
    pc = T.porkchop(dro, nrho, t_dep, tofs, S, keep_transfers=True, refine=3, verbose=True)
    cells = []
    for i in range(len(tofs)):
        row = []
        for j in range(len(t_dep)):
            tr = pc.transfers[i][j]
            if tr is None:
                row.append(None)
            else:
                x = tr.x_chaser.copy()
                x[3:] += tr.dv1
                row.append({"x": x.tolist(), "tof": tr.tof, "td": tr.t_depart,
                            "dv1": round(tr.dv1_mps, 1), "dv2": round(tr.dv2_mps, 1)})
        cells.append(row)
    b = pc.best
    bi = int(np.argmin(np.abs(tofs - b.tof)))
    bj = int(np.argmin(np.abs(t_dep - b.t_depart)))
    return {
        "dep_d": rnd(S.days(t_dep), 3), "tof_d": rnd(S.days(tofs), 3),
        "total": [[None if not np.isfinite(v) else round(float(v), 1) for v in row]
                  for row in pc.total_mps],
        "cells": cells, "best": [bi, bj],
        "dro": xyz(km_moon(dro.trajectory(300).r)), "nrho": xyz(km_moon(nrho.trajectory(300).r)),
    }


def compute_rpod():
    nrho = P.nrho()
    xt0 = nrho.state_at(0.0)
    rel0 = np.concatenate((S.nd_length(np.array([-20.0, 0.0, 1.0])), np.zeros(3)))
    xc0 = frames.from_lvlh(xt0, rel0, S, "moon")
    wps = [R.Waypoint(8.0, (-5.0, 0.0, 0.0), "5 km"),
           R.Waypoint(14.0, (-1.0, 0.0, 0.0), "1 km"),
           R.Waypoint(18.0, (-0.2, 0.0, 0.0), "200 m"),
           R.Waypoint(20.0, (-0.03, 0.0, 0.0), "30 m"),
           R.Waypoint(21.0, (0.0, 0.0, 0.0), "dock")]
    plan = R.plan_waypoints(xt0, xc0, wps, S, "moon", final_rhodot_mps=(0.03, 0.0, 0.0),
                            n_points_per_leg=150)
    aborts = R.passive_abort(plan, 24.0, n_points=300)
    rel = plan.rel
    t0h = rel.t_hours[0]
    return {
        "t_h": rnd(rel.t_hours - t0h, 3), "rho_m": {k: rnd(rel.rho_km[:, n] * 1000, 2)
                                                    for n, k in enumerate("xyz")},
        "range_m": rnd(rel.range_km * 1000, 3),
        "burns": [{"t_h": float(S.seconds(m.t - plan.t0) / 3600), "dv": m.dv_lvlh_mps.tolist(),
                   "mag": m.mag_mps, "label": m.label} for m in plan.maneuvers],
        "waypoints": [{"t_h": w.t_hours, "rho_m": [1000 * v for v in w.rho_km], "label": w.label}
                      for w in wps],
        "aborts": [{"burn": a.burn_index + 1, "min_m": a.min_range_km * 1000,
                    "t_min_h": a.t_min_hours, "t0_h": float(a.rel.t_hours[0] - t0h),
                    "rho_m": {k: rnd(a.rel.rho_km[:, n] * 1000, 1) for n, k in enumerate("xyz")}}
                   for a in aborts[:-1]],
        "total": plan.total_dv_mps, "start_m": [-20000, 0, 1000],
        "n_moon_apolune": float(np.sqrt(S.mu / np.linalg.norm(xt0[:3] - S.pos2) ** 3)
                                / S.tstar),
    }


ROSETTE_CASES = [
    (2, 1, 0.55, "TESS-like: apogees phased 90 deg from the Moon"),
    (3, 1, None, "IBEX has flown a 3:1 lunar-resonant orbit since 2011"),
    (4, 1, None, ""),
    (5, 2, 0.3, "closes after two lunar months"),
    (3, 2, 0.2, "passes within 30,000 km of the Moon: strongly unstable"),
    (1, 2, None, "exterior: outside the Moon's orbit"),
    (1, 3, 0.3, "exterior"),
    (2, 3, 0.5, "exterior, swings past the Moon"),
]


def compute_rosettes():
    res = []
    for p, q, e, note in ROSETTE_CASES:
        t0 = time.time()
        o = P.resonant(p, q, e)
        guess, _ = P.resonant_guess(p, q, e if e is not None else (0.5 if p > q else 0.2), S)
        tr = o.trajectory(1600)
        ti = frames.trajectory_to_inertial(tr, "earth")
        moon = frames.primary_track_inertial(tr.t, S, "moon", "earth")
        d_e = np.linalg.norm(tr.r - S.pos1, axis=1)
        d_m = np.linalg.norm(tr.r - S.pos2, axis=1)
        e0 = -S.pos1
        res.append({
            "key": f"{p}:{q}", "p": p, "q": q, "e": e if e is not None else (0.5 if p > q else 0.2),
            "note": note, "state0": o.state0.tolist(), "guess0": guess.tolist(), "period": o.period,
            "period_d": o.period_days, "perigee": float(S.km(d_e.min())), "apogee": float(S.km(d_e.max())),
            "min_moon": float(S.km(d_m.min())), "nu": float(o.stability_index()), "C": o.jacobi,
            "t_d": rnd(tr.t_days, 3),
            "rot": {"x": rnd(S.km(tr.r[:, 0] + e0[0]), 0), "y": rnd(S.km(tr.r[:, 1]), 0)},
            "inr": {"x": rnd(S.km(ti.r[:, 0]), 0), "y": rnd(S.km(ti.r[:, 1]), 0)},
            "moon": {"x": rnd(S.km(moon[:, 0]), 0), "y": rnd(S.km(moon[:, 1]), 0)},
        })
        log(f"  {p}:{q} resonant ({time.time() - t0:.0f}s)")
    lunar = {}
    for key, label, o, revs in (("nrho", "9:2 NRHO", P.nrho(), 18),
                                ("dro", "70,000 km DRO", P.dro(70000.0), 8),
                                ("l2h", "L2 southern halo", P.halo(2, 20000.0, "south"), 6)):
        rot = o.trajectory(400)
        full = cr3bp.propagate(o.state0, revs * o.period, S, n_points=300 * revs + 1)
        mi = frames.trajectory_to_inertial(full, "moon")
        lunar[key] = {"label": label, "revs": revs, "per_rev": 300, "period_d": o.period_days,
                      "rot": xyz(km_moon(rot.r)), "inr": xyz(S.km(mi.r)), "t_d": rnd(full.t_days, 3)}
    return {"resonant": res, "lunar": lunar}


def compute_transient():
    """Data for manual section 12: manifolds, fate maps, flybys, TCMs, SK, budgets."""
    from cislunar import manifolds as MF, navigation as NV

    out = {}
    t0 = time.time()
    # --- manifold tubes -----------------------------------------------------
    tubes = {}
    for key, label, o in (("l1lyap", "L1 Lyapunov (20,000 km)", P.lyapunov(1, 20000.0)),
                          ("l2lyap", "L2 Lyapunov (20,000 km)", P.lyapunov(2, 20000.0)),
                          ("l2halo", "L2 southern halo (20,000 km)", P.halo(2, 20000.0, "south"))):
        rec = {"label": label, "efold_d": float(S.days(MF.efolding_time(o))),
               "orbit": xyz(km_moon(o.trajectory(300).r)), "tubes": {}}
        for kind in ("unstable", "stable"):
            for br in ("interior", "exterior"):
                tb = MF.manifold(o, kind, br, n=24, eps_km=50.0, tf_days=30.0, n_points=220)
                rec["tubes"][f"{kind}-{br}"] = {
                    "paths": [xyz(km_moon(tr.r)) for tr in tb.trajectories],
                    "fates": [f.kind for f in tb.fates],
                    "t_d": [round(float(abs(S.days(f.t))), 2) for f in tb.fates],
                    "counts": tb.fate_counts()}
        tubes[key] = rec
        log(f"  manifolds {key} ({time.time() - t0:.0f}s)")
    out["tubes"] = tubes
    # --- fate maps ------------------------------------------------------------
    d = np.linspace(-50000.0, 55000.0, 70)
    C = np.linspace(2.98, 3.20, 45)
    maps = {}
    for sense in ("retrograde", "prograde"):
        fm = MF.fate_map(d, C, sense, horizon_days=60.0, with_ftle=True, ftle_horizon_days=10.0)
        maps[sense] = {"code": fm.code.tolist(), "life_d": rnd(fm.lifetime_days, 1),
                       "ftle": rnd(np.nan_to_num(fm.ftle / S.days(1.0), nan=-1.0), 3),
                       "fractions": {k: fm.fraction(k) for k in MF.FATES}}
        log(f"  fate map {sense} ({time.time() - t0:.0f}s)")
    out["fatemap"] = {"d_km": d.tolist(), "C": C.tolist(), "fates": MF.FATES, "horizon_d": 60.0,
                      "gate_margin_km": MF.GATE_MARGIN_KM, "maps": maps}
    # --- flyby sweep + return leg --------------------------------------------
    dro, nrho = P.dro(70000.0), P.nrho()
    td, tof = S.nd_time_days(7.33), S.nd_time_days(9.0)
    sweep = []
    for miss in (0.0, 5.0, 25.0, 100.0, 500.0):
        fb = T.flyby(dro, nrho, td, tof, miss_km=miss)
        sweep.append({"miss": miss, "dv1": fb.dv1_mps, "vrel": fb.v_rel_mps, "ca": fb.miss_km,
                      "t_h": rnd((fb.rel_t - fb.t_ca) * S.tstar / 3600.0, 3),
                      "range_km": rnd(fb.rel_range_km, 2)})
    fb = T.flyby(dro, nrho, td, tof, miss_km=25.0)
    x_after, t_ret = fb.transfer.x_arrive, td + tof

    def coast(t):
        return cr3bp.propagate_state(x_after, t - t_ret, S)

    best = None
    for wait_d in (0.0, 1.0, 2.0):
        for tof_r in np.linspace(3.0, 12.0, 10):
            try:
                tr = T.intercept(coast, dro, t_ret + S.nd_time_days(wait_d), S.nd_time_days(tof_r), S)
            except RuntimeError:
                continue
            if best is None or tr.total_mps < best.total_mps:
                best = tr
    out["flyby"] = {"sweep": sweep, "depart_d": 7.33, "tof_d": 9.0,
                    "return": {"dv1": best.dv1_mps, "dv2": best.dv2_mps, "tof_d": best.tof_days,
                               "wait_d": float(S.days(best.t_depart - t_ret))}}
    log(f"  flyby ({time.time() - t0:.0f}s)")
    # --- TCM curves ----------------------------------------------------------
    leo = json.loads(CACHE.read_text(encoding="utf-8"))["leo"] if CACHE.exists() else None
    tcm = {}
    if leo:
        x_post = np.array(leo["x_chaser"])
        x_post[3:] += np.array(leo["dv1"])
        xf = cr3bp.propagate_state(x_post, leo["tof"], S)
        dva = np.array(leo["target_arrive"])[3:] - xf[3:]
        times_d = np.round(np.geomspace(0.05, 4.2, 32), 4)
        lin = NV.linear_tcm(x_post, leo["tof"], NV.injection_covariance(1.0), S.nd_time_days(times_d),
                            S, dv_arrival=dva)
        mc = NV.monte_carlo_tcm(x_post, leo["tof"], NV.injection_covariance(1.0),
                                S.nd_time_days(np.array([0.5, 3.0])), S,
                                target_state=np.array(leo["target_arrive"]), n=400)
        tcm["leo"] = {"t_d": times_d.tolist(), "mean": lin.tcm_mean.tolist(), "p95": lin.tcm_p95.tolist(),
                      "p99": lin.tcm_p99.tolist(), "tot99": lin.total_p99.tolist(),
                      "miss99_km": lin.miss_p99_km, "mc_total": rnd(mc.total_mps, 3),
                      "mc_p99": mc.percentile(99), "mc_tcm1_p99": float(np.percentile(mc.tcm_mps[:, 0], 99)),
                      "mc_miss_med_km": float(np.median(mc.miss_km)),
                      "lin_single_05_tot99": float(np.interp(0.5, times_d, lin.total_p99))}
    for name, (tr, dva) in (("outbound", (fb.transfer, None)), ("return", (best, best.dv2))):
        x_post = tr.x_chaser.copy()
        x_post[3:] += tr.dv1
        times_d = np.round(np.geomspace(0.05, min(4.2, 0.9 * tr.tof_days), 24), 4)
        lin = NV.linear_tcm(x_post, tr.tof, NV.burn_execution_covariance(tr.dv1, 0.01, 0.5, S),
                            S.nd_time_days(times_d), S, dv_arrival=dva)
        tcm[name] = {"t_d": times_d.tolist(), "p99": lin.tcm_p99.tolist(), "tot99": lin.total_p99.tolist(),
                     "mean": lin.tcm_mean.tolist(), "miss99_km": lin.miss_p99_km}
    out["tcm"] = tcm
    log(f"  TCM ({time.time() - t0:.0f}s)")
    # --- station-keeping sweep -------------------------------------------------
    sk = []
    configs = (("9:2 NRHO", P.nrho(), (1.64, 3.28, 6.56, 9.84, 13.12)),
               ("L2 halo 20,000 km", P.halo(2, 20000.0, "south"), (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0)),
               ("L1 halo 15,000 km", P.halo(1, 15000.0, "north"), (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)),
               ("DRO 70,000 km", P.dro(70000.0), (7.0,)))
    for name, orb, ivs in configs:
        orb.family = name
        for iv in ivs:
            r = NV.stationkeeping_mc(orb, iv, duration_days=180.0, n=12)
            ok = ~r.lost
            rec = {"orbit": name, "interval_d": iv, "efold_d": r.efold_days if np.isfinite(r.efold_days) else None,
                   "success": r.success_rate,
                   "mean": float(np.mean(r.per_year_mps[ok])) if ok.any() else None,
                   "p95": float(np.percentile(r.per_year_mps[ok], 95)) if ok.any() else None,
                   "burns_per_yr": float(np.mean(r.n_burns[ok]) / r.duration_days * 365.25) if ok.any() else None,
                   "maxdev_p95": float(np.percentile(r.max_dev_km[ok], 95)) if ok.any() else None}
            if (name, iv) in (("9:2 NRHO", 6.56), ("L2 halo 20,000 km", 2.0), ("L1 halo 15,000 km", 1.0)):
                rec["hist"] = {"t_d": rnd(r.history_t_days, 2), "dev_km": rnd(r.history_dev_km, 2),
                               "burn_d": rnd(r.history_burn_days, 2)}
            sk.append(rec)
        log(f"  station-keeping {name} ({time.time() - t0:.0f}s)")
    out["sk"] = sk
    return out


def compute_all():
    data = {}
    t0 = time.time()
    lp = cr3bp.lagrange_points(S.mu)
    data["system"] = {
        "mu": S.mu, "lstar": S.lstar, "tstar": S.tstar, "vstar": S.vstar,
        "r1": S.r1, "r2": S.r2, "gm1": S.gm1, "gm2": S.gm2,
        "lpoints": {k: v.tolist() for k, v in lp.items()},
        "lpoint_C": {k: float(cr3bp.jacobi(np.r_[v, 0, 0, 0], S.mu)) for k, v in lp.items()},
        "synodic_d": SYNODIC_MONTH_DAYS,
    }
    log("families ...")
    data["families"] = compute_families()
    log("sandbox / frames ...")
    data["starts"] = compute_sandbox_starts()
    data["frames"] = compute_frames_demo()
    log("LEO -> NRHO transfer (~2-3 min) ...")
    data["leo"] = compute_leo_transfer()
    log("DRO -> NRHO porkchop (~1 min) ...")
    data["porkchop"] = compute_porkchop()
    log("RPOD plan ...")
    data["rpod"] = compute_rpod()
    log("rosettes ...")
    data["rosettes"] = compute_rosettes()
    log("transient orbits & budgets (~15 min) ...")
    data["transient"] = compute_transient()
    log(f"computations done in {time.time() - t0:.0f}s")
    return data


# ---------------------------------------------------------------------------
# API reference
# ---------------------------------------------------------------------------

API_MODULES = [
    ("cislunar.constants", "Systems, constants and unit conversion"),
    ("cislunar.cr3bp", "Three-body dynamics and propagation"),
    ("cislunar.periodic", "Periodic orbits and families"),
    ("cislunar.twobody", "Two-body (Kepler) utilities"),
    ("cislunar.lambert", "Lambert solver"),
    ("cislunar.frames", "Reference frames and LVLH"),
    ("cislunar.targeting", "Intercepts, rendezvous, porkchops"),
    ("cislunar.relative", "RPOD: relative motion and approach planning"),
    ("cislunar.viz.plot2d", "2D plotting"),
    ("cislunar.viz.plot3d", "3D plotting"),
]


def api_html():
    import importlib

    parts, toc = [], []
    for modname, title in API_MODULES:
        mod = importlib.import_module(modname)
        anchor = "api-" + modname.replace(".", "-")
        toc.append(f'<li><a href="#{anchor}"><code>{modname}</code></a> &mdash; {title}</li>')
        parts.append(f'<section class="api-mod" id="{anchor}"><h3><code>{modname}</code> '
                     f'<span class="muted">&mdash; {title}</span></h3>')
        doc = inspect.getdoc(mod)
        if doc:
            parts.append(f'<p class="api-moddoc">{html.escape(doc)}</p>')
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_") or getattr(obj, "__module__", None) != modname:
                continue
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            kind = "class" if inspect.isclass(obj) else "def"
            try:
                sig = str(inspect.signature(obj))
            except (TypeError, ValueError):
                sig = "(...)"
            sig = sig.replace("System(name='Earth-Moon', gm1=398600.435436, gm2=4902.800066, "
                              "lstar=384400.0, r1=6378.137, r2=1737.4, name1='Earth', "
                              "name2='Moon')", "EARTH_MOON")
            d = inspect.getdoc(obj) or ""
            members = ""
            if inspect.isclass(obj):
                props = [n for n, v in vars(obj).items() if not n.startswith("_")
                         and (isinstance(v, property) or inspect.isfunction(v))]
                if props:
                    members = ('<p class="api-members">Members: ' + ", ".join(
                        f"<code>{html.escape(p)}</code>" for p in props) + "</p>")
            parts.append(
                f'<div class="api-item"><div class="api-sig"><span class="kw">{kind}</span> '
                f'<b>{html.escape(name)}</b><span class="sig">{html.escape(sig)}</span></div>'
                + (f"<pre class='api-doc'>{html.escape(d)}</pre>" if d else "")
                + members + "</div>")
        parts.append("</section>")
    return "<ul class='api-toc'>" + "".join(toc) + "</ul>" + "".join(parts)


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

def img_tag(name, alt, caption):
    p = FIGS / name
    if not p.exists():
        return f'<p class="muted">[figure {name} not found: run the examples first]</p>'
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return (f'<figure class="static"><img alt="{html.escape(alt)}" '
            f'src="data:image/png;base64,{b64}"><figcaption>{caption}</figcaption></figure>')


def main():
    rebuild = "--rebuild" in sys.argv
    if CACHE.exists() and not rebuild:
        log(f"using cached data {CACHE}")
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        if "--porkchop" in sys.argv:
            log("recomputing porkchop ...")
            data["porkchop"] = compute_porkchop()
            CACHE.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        if "--rosettes" in sys.argv or "rosettes" not in data:
            log("computing rosettes ...")
            data["rosettes"] = compute_rosettes()
            CACHE.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        if "--transient" in sys.argv or "transient" not in data:
            log("computing transient orbits & budgets (~15 min) ...")
            data["transient"] = compute_transient()
            CACHE.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        if "--families" in sys.argv:
            log("recomputing families ...")
            data["families"] = compute_families()
            CACHE.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    else:
        data = compute_all()
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    from plotly.offline import get_plotlyjs

    tpl = TEMPLATE.read_text(encoding="utf-8")
    images = {
        "IMG_FAMILIES_2D": img_tag("families_2d.png", "Orbit families in three projections",
                                   "<code>plot_trajectories_2d</code> output from "
                                   "<code>examples/01_orbit_families.py</code>."),
        "IMG_SPEED": img_tag("families_speed.png", "Orbits coloured by speed",
                             "<code>plot_trajectories_2d(..., color_by=\"speed\", speed_scale=\"log\")</code>: "
                             "the same orbits as a heat map of speed relative to the Moon."),
        "IMG_HALO_3D": img_tag("halo_family_3d.png", "L2 halo family in 3D",
                               "<code>plot_trajectories_3d</code> output: the L2 southern "
                               "halo family continued into the NRHOs."),
        "IMG_ZVC": img_tag("lyapunov_zvc.png", "Zero-velocity curve",
                           "<code>plot_zero_velocity</code> output: the grey region is "
                           "forbidden at this orbit's energy."),
        "IMG_PORKCHOP": img_tag("dro_nrho_porkchop.png", "Porkchop plot",
                                "<code>plot_porkchop</code> output from "
                                "<code>examples/02_dro_to_nrho_porkchop.py</code>."),
        "IMG_LEO": img_tag("leo_nrho_2d.png", "LEO to NRHO transfer",
                           "<code>examples/03_leo_to_nrho.py</code>: rotating (left) and "
                           "inertial (right) views of the same transfer."),
        "IMG_ROS_RES": img_tag("rosettes_resonant.png", "Resonant rosettes",
                               "<code>examples/05_rosettes.py</code>: resonant orbits in the rotating "
                               "frame (top) and the inertial frame (bottom)."),
        "IMG_ROS_LUN": img_tag("rosettes_lunar.png", "Lunar rosettes",
                               "<code>examples/05_rosettes.py</code>: the NRHO and a DRO seen from a "
                               "Moon-centred inertial frame."),
        "IMG_MANIFOLDS": img_tag("manifolds.png", "Manifold tubes",
                                 "<code>plot_manifold</code> output from <code>examples/06_manifolds_and_fates.py</code>: "
                                 "unstable manifolds of the L1 and L2 Lyapunov orbits."),
        "IMG_FATEMAP": img_tag("fate_map.png", "Fate map",
                               "<code>plot_fate_map</code> output: fate, lifetime and FTLE for retrograde starts near the Moon."),
        "IMG_RPOD": img_tag("rpod_vbar_rbar.png", "RPOD approach",
                            "<code>plot_relative_2d</code> and <code>plot_range_history</code> "
                            "output from <code>examples/04_rpod_nrho.py</code>."),
    }
    out = tpl.replace("__VERSION__", cl.__version__)
    out = out.replace("<!--__API__-->", api_html())
    for k, v in images.items():
        out = out.replace(f"<!--__{k}__-->", v)
    data_js = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out = out.replace("/*__DATA__*/null", data_js)
    out = out.replace("/*__PLOTLY__*/", get_plotlyjs())  # last: never scanned for placeholders
    OUT.write_text(out, encoding="utf-8")
    log(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
