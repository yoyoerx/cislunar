"""Build docs/case_study_rock_or_rocket.html: tell a rock from a rocket stage with orbit determination.

2020 SO (the Surveyor 2 Centaur stage, catalogued at first as an asteroid) and 2024 PT5 (a
natural mini-moon) were both temporarily captured by the Earth. The same pipeline fits each
and measures how hard sunlight pushes on it: hollow hardware is pushed ~100x harder than rock.

    python docs/build_case_study_rock.py            # uses docs/_cache/case_study_rock.json if present
    python docs/build_case_study_rock.py --rebuild  # recompute (~20 min on 7 cores, offline)
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import cislunar as cl
from cislunar import astrometry as A
from cislunar import cr3bp
from cislunar import ephemeris as E
from cislunar import od

HERE = Path(__file__).parent
DATA = HERE / "data" / "rock_or_rocket"
CACHE = HERE / "_cache" / "case_study_rock.json"
TEMPLATE = HERE / "case_study_rock_template.html"
OUT = HERE / "case_study_rock_or_rocket.html"
S = cl.EARTH_MOON
WORKERS = 7
GM_E = 398600.435436
AU = E.AU_KM
SUN_EARTH_MU = (S.gm1 + S.gm2) / (E.GM_SUN + S.gm1 + S.gm2)
R_HILL = AU * (SUN_EARTH_MU / 3) ** (1 / 3)

OBJECTS = {
    "2020so": {"label": "2020 SO", "eph": "ephemeris_2020_2021.json", "iod": "gauss", "iod_days": 3.0,
               "grid": [0.0, 0.004, 0.008, 0.010, 0.011, 0.012, 0.0126, 0.013, 0.014, 0.016]},
    "2024pt5": {"label": "2024 PT5", "eph": "ephemeris_2024_2025.json", "iod": "scan", "iod_days": 10.0,
                "grid": [-0.002, -0.001, -0.0005, -0.0002, 0.0, 0.0002, 0.0005, 0.001, 0.002, 0.005]},
}


def log(m):
    print(m, flush=True)


def rnd(a, nd=0):
    return np.round(np.asarray(a, float), nd).tolist()


def utc_str(jd_utc, fmt="%Y-%m-%d"):
    from datetime import datetime, timedelta
    return (datetime(2000, 1, 1, 12) + timedelta(days=jd_utc - E.JD_J2000)).strftime(fmt)


def setup(key, amr=0.0):
    cfg = OBJECTS[key]
    st = A.load_stations(DATA / "stations.txt")
    recs = sorted(A.parse(DATA / f"{key}_astrometry.txt", strict=False), key=lambda r: r.jd_utc)
    eph = E.Ephemeris.load(DATA / cfg["eph"])
    dyn = E.NBodyDynamics(eph, E.utc_to_tdb(recs[0].jd_utc), amr=amr)
    return recs, st, eph, dyn


def _profile_task(args):
    key, amr, x0, mask = args
    recs, st, _, dyn = setup(key, amr)
    obs = [o for o, m in zip(A.to_observations(recs, st, dyn), mask) if m]
    f = od.fit(obs, np.asarray(x0), 0.0, S, dynamics=dyn, max_iter=80)
    res = A.residuals_arcsec(f, obs)
    return {"amr": amr, "chi2": f.chi2, "rms": float(np.sqrt(np.mean(res**2))), "converged": f.converged, "x": f.x.tolist()}


def reject(f, obs, t_d, span, mask):
    res = A.residuals_arcsec(f, obs)
    tot = np.hypot(res[:, 0], res[:, 1])
    inarc = t_d <= span
    cut = max(3.0, 5 * 1.4826 * np.median(tot[inarc & mask]))
    return mask & ~(inarc & (tot > cut)), tot


def progressive(key, x, amr, t_d, start_days, mask=None):
    recs, st, _, dyn = setup(key, amr)
    obs = A.to_observations(recs, st, dyn)
    mask = np.ones(len(obs), bool) if mask is None else mask.copy()
    span, steps = start_days, []
    while True:
        span = min(span * 1.6, t_d.max() + 1)
        sel = [o for o, t, m in zip(obs, t_d, mask) if t <= span and m]
        f = od.fit(sel, x, 0.0, S, dynamics=dyn, max_iter=60)
        x = f.x
        mask, tot = reject(f, obs, t_d, span, mask)
        steps.append({"span_d": float(span), "n": int(np.sum((t_d <= span) & mask)),
                      "rms": float(np.sqrt(np.mean(tot[(t_d <= span) & mask] ** 2)))})
        if span > t_d.max():
            return x, mask, steps


def sun_earth_view(dyn, eph, X, ts):
    """Geocentric distance, two-body energy w.r.t. Earth, Sun-Earth rotating-frame position
    (Earth-centred, km, x away from the Sun) and Sun-Earth Jacobi value along a trajectory."""
    mu = SUN_EARTH_MU
    out = {"r_km": [], "energy": [], "x": [], "y": [], "C": []}
    for t, x in zip(ts, X):
        r, v = S.km(x[:3]), S.kms(x[3:])
        out["r_km"].append(float(np.linalg.norm(r)))
        out["energy"].append(float(0.5 * v @ v - GM_E / np.linalg.norm(r)))
        rs, vs = eph.state("sun", float(dyn.jd(t)))  # Sun relative to Earth
        d = np.linalg.norm(rs)
        ex = -rs / d  # Sun -> Earth direction
        h = np.cross(-rs, -vs)
        ez = h / np.linalg.norm(h)
        ey = np.cross(ez, ex)
        R = np.vstack((ex, ey, ez))
        w = np.linalg.norm(h) / d**2
        out["x"].append(float(R[0] @ r))
        out["y"].append(float(R[1] @ r))
        # Jacobi in the pulsating Sun-Earth frame (barycentre ~ Sun at mu ~ 3e-6)
        # position relative to the Sun-Earth barycentre: Earth sits at +(1-mu) d along ex
        rb = r + (1 - mu) * (-rs)
        vb = v + (1 - mu) * (-vs)
        rr = R @ rb / d
        vrot = R @ (vb - np.cross(w * ez, rb))
        ddot = (-rs) @ (-vs) / d
        vv = (vrot - ddot * (R @ rb) / d) / (d * np.sqrt((E.GM_SUN + S.gm1 + S.gm2) / d**3))
        r1 = np.linalg.norm(rr - [-mu, 0, 0])
        r2 = np.linalg.norm(rr - [1 - mu, 0, 0])
        out["C"].append(float(2 * (0.5 * (rr[0] ** 2 + rr[1] ** 2) + (1 - mu) / r1 + mu / r2) - vv @ vv))
    return out


def analyse(key):
    cfg = OBJECTS[key]
    log(f"== {cfg['label']} ==")
    recs, st, eph, dyn = setup(key)
    t_d = np.array([r.jd_utc - recs[0].jd_utc for r in recs])
    first = [r for r, t in zip(recs, t_d) if t < cfg["iod_days"]]
    out = {"label": cfg["label"], "n_obs": len(recs), "stations": sorted({r.station for r in recs}),
           "first_utc": utc_str(recs[0].jd_utc), "last_utc": utc_str(recs[-1].jd_utc), "arc_days": float(t_d.max())}
    t0 = time.time()
    if cfg["iod"] == "gauss":
        picks = [first[0], first[len(first) // 2], first[-1]]
        obs = A.to_observations(first, st, dyn)
        best = None
        for jdm, r, v in A.gauss_iod(picks, st):
            x = dyn.propagate_many(S.state_to_nd(np.concatenate((r, v))), float(dyn.t_of_jd(E.utc_to_tdb(jdm))), [0.0],
                                   with_stm=False)[0][0]
            f = od.fit(obs, x, 0.0, S, dynamics=dyn)
            if best is None or f.chi2 < best.chi2:
                best, info = f, {"method": "Gauss", "r_km": float(np.linalg.norm(r))}
        x0 = best.x
    else:
        gs = A.gauss_iod([first[0], first[len(first) // 2], first[-1]], st)
        f, p1, p2 = A.range_scan_iod(first, st, dyn, workers=WORKERS)[0]
        x0 = f.x
        info = {"method": "range scan", "gauss_r_km": [float(np.linalg.norm(r)) for _, r, _ in gs], "rho1_km": p1, "rho2_km": p2,
                "r_km": float(S.km(np.linalg.norm(x0[:3])))}
    info.update(n=len(first), days=cfg["iod_days"], seconds=time.time() - t0)
    out["iod"] = info
    log(f"  first orbit: {info}")

    log("  progressive fit, gravity only ...")
    x_g, mask, steps = progressive(key, x0, 0.0, t_d, cfg["iod_days"])
    out["steps_gravity"] = steps
    log("  sunlight-pressure profile ...")
    with ProcessPoolExecutor(WORKERS) as ex:
        prof = list(ex.map(_profile_task, [(key, a, x_g.tolist(), mask) for a in cfg["grid"]]))
    k = int(np.argmin([p["chi2"] for p in prof]))
    x_b = np.array(prof[k]["x"])
    # re-do outlier rejection at the best sunlight pressure, then the final profile
    recs_, st_, _, dyn_b = setup(key, cfg["grid"][k])
    obs_b = A.to_observations(recs_, st_, dyn_b)
    fb = od.fit([o for o, m in zip(obs_b, mask) if m], x_b, 0.0, S, dynamics=dyn_b, max_iter=60)
    mask, _ = reject(fb, obs_b, t_d, t_d.max() + 1, np.ones(len(obs_b), bool))
    with ProcessPoolExecutor(WORKERS) as ex:
        prof = list(ex.map(_profile_task, [(key, a, fb.x.tolist(), mask) for a in cfg["grid"]]))
    chi = np.array([p["chi2"] for p in prof])
    k = int(np.argmin(chi))
    sel = slice(max(k - 1, 0), min(k + 2, len(chi)))
    g = np.array(cfg["grid"])[sel]
    c2 = np.polyfit(g, chi[sel], 2)
    amr_hat, sig = float(-c2[1] / (2 * c2[0])), float(1 / np.sqrt(c2[0]))
    out["profile"] = [{q: p[q] for q in ("amr", "chi2", "rms", "converged")} for p in prof]
    out["amr"], out["amr_sigma"], out["n_used"] = amr_hat, sig, int(mask.sum())
    log(f"  C_r A/m = {amr_hat:.5f} +/- {sig:.5f} ({mask.sum()} of {len(mask)} observations kept)")

    final = _profile_task((key, amr_hat, prof[k]["x"], mask))
    recs_, st_, _, dyn_f = setup(key, amr_hat)
    obs_f = A.to_observations(recs_, st_, dyn_f)
    ff = od.fit([o for o, m in zip(obs_f, mask) if m], np.array(final["x"]), 0.0, S, dynamics=dyn_f, max_iter=60)
    res = A.residuals_arcsec(ff, obs_f)
    out["final"] = {"rms": final["rms"], "chi2": ff.chi2, "x0_km": S.state_to_dim(ff.x).tolist()}
    out["resid"] = {"t_d": rnd(t_d, 3), "dra": rnd(res[:, 0], 2), "ddec": rnd(res[:, 1], 2), "used": mask.tolist(),
                    "st": [r.station for r in recs]}
    log("  progressive fit with the measured sunlight pressure (for comparison) ...")
    _, _, steps_b = progressive(key, x0, amr_hat, t_d, cfg["iod_days"], mask=mask)
    out["steps_srp"] = steps_b

    log("  trajectory through the capture ...")
    j0, j1 = eph.span()
    ts = np.arange(float(dyn_f.t_of_jd(j0 + 1)), float(dyn_f.t_of_jd(j1 - 1)), S.nd_time_days(0.25))
    X = [x for x, _ in dyn_f.propagate_many(ff.x, 0.0, ts, with_stm=False)]
    view = sun_earth_view(dyn_f, eph, X, ts)
    out["traj"] = {"date": [utc_str(E.tdb_to_utc(float(dyn_f.jd(t))), "%Y-%m-%d %H:%M") for t in ts],
                   "r_km": rnd(view["r_km"]), "energy": rnd(view["energy"], 5), "x": rnd(view["x"]), "y": rnd(view["y"]),
                   "C": rnd(view["C"], 7)}
    cap = [d for d, e, r in zip(out["traj"]["date"], view["energy"], view["r_km"]) if e < 0 and r < 3 * R_HILL]
    out["capture"] = {"start": cap[0][:10] if cap else None, "end": cap[-1][:10] if cap else None,
                      "min_r_km": float(min(view["r_km"])), "min_r_date": out["traj"]["date"][int(np.argmin(view["r_km"]))][:10]}
    log(f"  captured {out['capture']}")
    return out, dyn_f, ff


def compute():
    t0 = time.time()
    out = {"r_hill_km": R_HILL, "sun_earth_mu": SUN_EARTH_MU}
    lp = cr3bp.lagrange_points(SUN_EARTH_MU)
    out["C_L"] = {k: float(cr3bp.jacobi(np.concatenate((lp[k], np.zeros(3))), SUN_EARTH_MU)) for k in ("L1", "L2")}
    out["L_km"] = {k: float((lp[k][0] - (1 - SUN_EARTH_MU)) * AU) for k in ("L1", "L2")}
    so, _, _ = analyse("2020so")
    pt5, dyn, ff = analyse("2024pt5")
    # 2024 PT5 against JPL's orbit (a fit to the same data by JPL, not independent truth)
    jpl = E.Ephemeris.load(DATA / "jpl_2024pt5.json")
    jds = np.arange(2460523.5, 2460712.5, 1.0)
    X = dyn.propagate_many(ff.x, 0.0, [float(dyn.t_of_jd(j)) for j in jds], with_stm=False)
    pt5["vs_jpl"] = {"date": [utc_str(E.tdb_to_utc(j)) for j in jds],
                     "km": rnd([np.linalg.norm(S.km(x[:3]) - jpl.state("pt5", j)[0]) for (x, _), j in zip(X, jds)], 1)}
    pt5["jpl"] = json.loads((DATA / "jpl_2024pt5_sbdb.json").read_text(encoding="utf-8"))
    # Find_Orb's radial non-gravitational term for 2020 SO, converted to C_r A/m
    a1 = 2.87e-9
    a1_sig = 9.29e-12
    k = AU * 1000 / 86400.0**2 / E.SOLAR_PRESSURE_1AU
    so["find_orb"] = {"A1": a1, "amr": a1 * k, "amr_sigma": a1_sig * k, "rms": 1.519, "n": 596,
                      "source": "https://projectpluto.com/pluto/mpecs/66084b.htm"}
    out["objects"] = {"2020so": so, "2024pt5": pt5}
    out["elapsed_s"] = time.time() - t0
    log(f"done in {out['elapsed_s']:.0f}s")
    return out


def main():
    if CACHE.exists() and "--rebuild" not in sys.argv:
        log(f"using cached data {CACHE}")
        data = json.loads(CACHE.read_text(encoding="utf-8"))
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
    tpl = tpl.replace("/*__CSS__*/", manual[manual.index("<style>") + len("<style>"):manual.index("</style>")])
    tpl = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    OUT.write_text(tpl.replace("/*__PLOTLY__*/", get_plotlyjs()), encoding="utf-8")
    log(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
