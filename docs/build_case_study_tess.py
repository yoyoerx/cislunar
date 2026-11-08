"""Build docs/case_study_tess.html: orbit determination of TESS from survey-telescope astrometry.

TESS flies a stable 2:1 lunar-resonant orbit (two orbits per lunar month). Sky surveys
catch it as an unknown moving object; 134 such sightings (2018-2020) are published by
Project Pluto. NASA's own trajectory (JPL Horizons, -95) is the truth for every check.

    python docs/build_case_study_tess.py            # uses docs/_cache/case_study_tess.json if present
    python docs/build_case_study_tess.py --rebuild  # recompute (~10 min, parallel, offline)
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
from cislunar import ephemeris as E
from cislunar import od, twobody
from cislunar import periodic as P

HERE = Path(__file__).parent
DATA = HERE / "data" / "tess"
CACHE = HERE / "_cache" / "case_study_tess.json"
TEMPLATE = HERE / "case_study_tess_template.html"
OUT = HERE / "case_study_tess.html"
S = cl.EARTH_MOON
MU_E = 398600.435436
ARC_START = 2459110.0  # 2020 Sep 17: the late-2020 arc is Sep 24 - Nov 20
WORKERS = 7
YEARS = 10


def log(m):
    print(m, flush=True)


def rnd(a, nd=0):
    return np.round(np.asarray(a, float), nd).tolist()


def load():
    recs = A.parse(DATA / "astrometry.txt")
    st = A.load_stations(DATA / "stations.txt")
    eph = E.Ephemeris.load(DATA / "ephemeris_2018_2021.json")
    truth = E.Ephemeris.load(DATA / "truth_tess_horizons.json")
    return recs, st, eph, truth


def utc_str(jd_utc, fmt="%Y-%m-%d %H:%M"):
    from datetime import datetime, timedelta
    return (datetime(2000, 1, 1, 12) + timedelta(days=jd_utc - E.JD_J2000)).strftime(fmt)


def truth_x(truth, jd_tdb):
    r, v = truth.state("tess", jd_tdb)
    return S.state_to_nd(np.concatenate((r, v)))


def coe(xnd):
    d = S.state_to_dim(xnd)
    c = twobody.rv2coe(d[:3], d[3:], MU_E)
    return {"a_km": c["a"], "e": c["e"], "i_deg": float(np.degrees(c["i"])), "raan_deg": float(np.degrees(c["raan"])),
            "argp_deg": float(np.degrees(c["argp"])), "period_d": float(2 * np.pi * np.sqrt(c["a"] ** 3 / MU_E) / 86400)}


# ------------------------------------------------------------------ worker tasks

def _arc_fit(args):
    """Fit the late-2020 telescope arc with a fixed sunlight-pressure value."""
    amr, x0, jd0 = args
    recs, st, eph, _ = load()
    arc = [r for r in recs if r.jd_utc > ARC_START]
    dyn = E.NBodyDynamics(eph, jd0, amr=amr)
    obs = A.to_observations(arc, st, dyn)
    f = od.fit(obs, np.asarray(x0), 0.0, S, dynamics=dyn, max_iter=80)
    return {"amr": amr, "chi2": f.chi2, "x": f.x.tolist(), "P": f.P.tolist(), "converged": f.converged}


def _window_rms(args):
    """RMS misfit of one unpowered arc fitted to 10 days of NASA's trajectory."""
    jd_a, amr = args
    _, _, eph, truth = load()
    origin = od.Observer(lambda t: np.zeros(6), "geocentre")
    dyn = E.NBodyDynamics(eph, jd_a, amr=amr)
    jds = np.arange(jd_a, jd_a + 10.0, 3.0 / 24)
    obs = [od.Observation(float(dyn.t_of_jd(j)), "position", S.nd_length(truth.state("tess", j)[0]), S.nd_length(0.05),
                          origin, "native") for j in jds]
    f = od.fit(obs, truth_x(truth, jd_a), 0.0, S, dynamics=dyn)
    res = od._prop(dyn, f.x, 0.0, [o.t for o in obs], S, with_stm=False)
    err = np.array([S.km(np.linalg.norm(x[:3] - o.z)) for (x, _), o in zip(res, obs)])
    return float(np.sqrt(np.mean(err**2)))


def _longterm(args):
    """Propagate YEARS years; return per-orbit perigee/apogee and the Moon angle at apogee."""
    x0, jd0, amr = args
    lt = E.Ephemeris.load(DATA / "ephemeris_2020_2031_12h.json")
    dyn = E.NBodyDynamics(lt, jd0, amr=amr)
    tt, X, imp = dyn.propagate(np.asarray(x0), 0.0, float(dyn.t_of_jd(jd0 + 365.25 * YEARS)), n_points=YEARS * 2600)
    r = S.km(np.linalg.norm(X[:, :3], axis=1))
    k = np.arange(1, len(r) - 1)
    peri = k[(r[k] < r[k - 1]) & (r[k] < r[k + 1])]
    apo = k[(r[k] > r[k - 1]) & (r[k] > r[k + 1])]
    days = S.days(tt)
    ang = []
    for i in apo:
        m = S.nd_length(lt.state("moon", float(dyn.jd(tt[i])))[0])
        c = X[i, :3] @ m / (np.linalg.norm(X[i, :3]) * np.linalg.norm(m))
        ang.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    return {"peri_t": rnd(days[peri], 2), "peri_km": rnd(r[peri]), "apo_t": rnd(days[apo], 2), "apo_km": rnd(r[apo]),
            "moon_angle": rnd(ang, 1), "impact": None if imp is None else imp["body"]}


# ------------------------------------------------------------------ analysis

def compute():
    t_start = time.time()
    recs, st, eph, truth = load()
    ref = json.loads((DATA / "reference.json").read_text(encoding="utf-8"))
    out = {"reference": ref, "n_obs": len(recs)}
    out["stations"] = {k: {"name": v[3], "n": sum(r.station == k for r in recs),
                           "first": utc_str(min(r.jd_utc for r in recs if r.station == k), "%Y-%m-%d"),
                           "last": utc_str(max(r.jd_utc for r in recs if r.station == k), "%Y-%m-%d")}
                       for k, v in st.items()}

    log("observations vs NASA's trajectory ...")
    rows = []
    for r in recs:
        jd = E.utc_to_tdb(r.jd_utc)
        rs, _ = E.station_j2000(*st[r.station][:3], r.jd_utc)
        rho = truth.state("tess", jd)[0] - rs
        for _ in range(3):
            tau = np.linalg.norm(rho) / E.C_KMS
            rho = truth.state("tess", jd - tau / 86400)[0] - rs + eph.state("sun", jd)[1] * tau
        ra, de = np.arctan2(rho[1], rho[0]), np.arcsin(rho[2] / np.linalg.norm(rho))
        rows.append((r.jd_utc, r.station, ((r.ra - ra + np.pi) % (2 * np.pi) - np.pi) * np.cos(de) * 206264.8,
                     (r.dec - de) * 206264.8, float(np.linalg.norm(rho)), r.mag))
    out["obs"] = {"date": [utc_str(x[0], "%Y-%m-%d") for x in rows], "jd": rnd([x[0] for x in rows], 4),
                  "st": [x[1] for x in rows], "dra": rnd([x[2] for x in rows], 2), "ddec": rnd([x[3] for x in rows], 2),
                  "range_km": rnd([x[4] for x in rows]), "mag": [x[5] for x in rows]}

    arc = [r for r in recs if r.jd_utc > ARC_START]
    nov = [r for r in arc if r.jd_utc > 2459173.0]
    jd0 = E.utc_to_tdb(nov[0].jd_utc)  # epoch: first sighting on 2020 Nov 20
    out["epoch_utc"] = utc_str(nov[0].jd_utc)

    log("one night: Gauss IOD + fit on 2020 Nov 20 ...")
    dyn0 = E.NBodyDynamics(eph, jd0, amr=0.0)
    on = A.to_observations(nov, st, dyn0)
    best1 = None
    for jdm, r, v in A.gauss_iod([nov[0], nov[5], nov[-1]], st):
        x = dyn0.propagate_many(S.state_to_nd(np.concatenate((r, v))), float(dyn0.t_of_jd(E.utc_to_tdb(jdm))), [0.0],
                                with_stm=False)[0][0]
        f = od.fit(on, x, 0.0, S, dynamics=dyn0)
        if best1 is None or f.chi2 < best1.chi2:
            best1, gauss = f, (float(np.linalg.norm(r)), float(np.linalg.norm(v)))
    d1 = best1.x - truth_x(truth, jd0)
    out["night"] = {"n": len(nov), "stations": sorted({r.station for r in nov}), "gauss_r_km": gauss[0], "gauss_v_kms": gauss[1],
                    "sigma_km": best1.sigma_pos_km(), "sigma_mps": best1.sigma_vel_mps(),
                    "err_km": float(S.km(np.linalg.norm(d1[:3]))), "err_mps": float(S.mps(np.linalg.norm(d1[3:]))),
                    "elements": coe(best1.x), "truth_elements": coe(truth_x(truth, jd0)),
                    "span_h": (nov[-1].jd_utc - nov[0].jd_utc) * 24}

    log("three nights, sunlight-pressure profile ...")
    grid = [0.018, 0.021, 0.024, 0.026, 0.027, 0.028, 0.030, 0.033, 0.036]
    with ProcessPoolExecutor(WORKERS) as ex:
        prof = list(ex.map(_arc_fit, [(a, best1.x.tolist(), jd0) for a in grid]))
    chi = np.array([p["chi2"] for p in prof])
    kmin = int(np.argmin(chi))
    sel = slice(max(kmin - 2, 0), min(kmin + 3, len(grid)))
    c2 = np.polyfit(np.array(grid[sel]), chi[sel], 2)
    amr_hat = float(-c2[1] / (2 * c2[0]))
    sig_amr = float(1 / np.sqrt(c2[0]))
    best = _arc_fit((amr_hat, prof[kmin]["x"], jd0))
    dyn = E.NBodyDynamics(eph, jd0, amr=amr_hat)
    oa = A.to_observations(arc, st, dyn)
    fit = od.ODResult(np.array(best["x"]), 0.0, np.array(best["P"]), best["chi2"], 0.0, 2 * len(arc), True, 0, False, 6,
                      system=S, dynamics=dyn)
    ra = A.residuals_arcsec(fit, oa)
    d3 = fit.x - truth_x(truth, jd0)
    # elements at Find_Orb's epoch, with covariance mapped through the STM
    t23 = float(dyn.t_of_jd(2459145.5))
    (x23, phi), = dyn.propagate_many(fit.x, 0.0, [t23], with_stm=True)
    P23 = phi @ fit.P @ phi.T
    keys = ["a_km", "e", "i_deg", "raan_deg", "argp_deg"]
    J = np.zeros((5, 6))
    for k in range(6):
        e = np.zeros(6)
        e[k] = 1e-8
        J[:, k] = (np.array([coe(x23 + e)[q] for q in keys]) - np.array([coe(x23 - e)[q] for q in keys])) / 2e-8
    out["arc"] = {"n": len(arc), "dates": sorted({utc_str(r.jd_utc, "%Y-%m-%d") for r in arc}),
                  "profile": [{"amr": p["amr"], "chi2": p["chi2"]} for p in prof], "amr": amr_hat, "amr_sigma": sig_amr,
                  "chi2": best["chi2"], "rms_arcsec": float(np.sqrt(np.mean(ra**2))),
                  "resid": {"st": [r.station for r in arc], "date": [utc_str(r.jd_utc, "%m-%d") for r in arc],
                            "dra": rnd(ra[:, 0], 2), "ddec": rnd(ra[:, 1], 2)},
                  "sigma_km": fit.sigma_pos_km(), "sigma_mps": fit.sigma_vel_mps(),
                  "err_km": float(S.km(np.linalg.norm(d3[:3]))), "err_mps": float(S.mps(np.linalg.norm(d3[3:]))),
                  "elements": coe(x23), "elements_sigma": dict(zip(keys, np.sqrt(np.diag(J @ P23 @ J.T)).tolist())),
                  "truth_elements": coe(truth_x(truth, 2459145.5))}
    log(f"  C_r A/m = {amr_hat:.5f} +/- {sig_amr:.5f}; rms {out['arc']['rms_arcsec']:.2f}\"; error vs truth {out['arc']['err_km']:.1f} km")

    log("prediction error vs truth, Apr 2020 - Jan 2021 ...")
    jds = np.arange(2458941.0, 2459245.0, 1.0)
    ts = [float(dyn.t_of_jd(j)) for j in jds]
    curves = {}
    for name, f, d in (("three", fit, dyn), ("one", best1, dyn0)):
        res = d.propagate_many(f.x, 0.0, ts, with_stm=False)
        curves[name] = rnd([S.km(np.linalg.norm(x[:3] - truth_x(truth, j)[:3])) for (x, _), j in zip(res, jds)], 1)
    out["pred"] = {"date": [utc_str(E.tdb_to_utc(j), "%Y-%m-%d") for j in jds], **curves}
    # predicted sky positions for the earlier 2020 sightings (Jan-Apr), vs what the telescopes saw
    early = [r for r in recs if 2458850 < r.jd_utc < ARC_START]
    oe = A.to_observations(early, st, dyn)
    re = A.residuals_arcsec(fit, oe)
    out["back"] = {"date": [utc_str(r.jd_utc, "%Y-%m-%d") for r in early], "sep_arcmin": rnd(np.hypot(re[:, 0], re[:, 1]) / 60, 2)}

    log("is NASA's trajectory one unpowered arc? (10-day windows) ...")
    starts = np.arange(2459116.5, 2459166.6, 2.0)
    with ProcessPoolExecutor(WORKERS) as ex:
        win = list(ex.map(_window_rms, [(a, amr_hat) for a in starts]))
    out["windows"] = {"start": [utc_str(E.tdb_to_utc(a), "%Y-%m-%d") for a in starts], "rms_km": rnd(win, 2)}

    log(f"{YEARS}-year stability runs ...")
    rng = np.random.default_rng(3)
    w, V = np.linalg.eigh(0.5 * (fit.P + fit.P.T))
    L = V * np.sqrt(np.clip(w, 0, None))
    xt = truth_x(truth, jd0)
    runs = [(xt.tolist(), jd0, amr_hat), (fit.x.tolist(), jd0, amr_hat)]
    runs += [((fit.x + L @ rng.standard_normal(6)).tolist(), jd0, float(amr_hat + sig_amr * rng.standard_normal()))
             for _ in range(12)]
    with ProcessPoolExecutor(WORKERS) as ex:
        lt = list(ex.map(_longterm, runs))
    out["longterm"] = {"truth_start": lt[0], "fit": lt[1], "samples": lt[2:], "years": YEARS}

    log("CR3BP view: 2:1 resonant periodic orbit vs TESS in the rotating frame ...")
    o21 = P.resonant(2, 1, out["arc"]["elements"]["e"])
    tr21 = o21.trajectory(1500)
    rot = []
    for j in np.arange(2459116.5, 2459171.1, 0.1):
        rm, vm = eph.state("moon", j)
        ex_ = rm / np.linalg.norm(rm)
        ez = np.cross(rm, vm)
        ez /= np.linalg.norm(ez)
        R = np.vstack((ex_, np.cross(ez, ex_), ez))
        rot.append(R @ truth.state("tess", j)[0] * (S.lstar / np.linalg.norm(rm)))
    rot = np.array(rot)
    out["cr3bp"] = {"orbit": {"x": rnd(S.km(tr21.r[:, 0] - S.pos1[0])), "y": rnd(S.km(tr21.r[:, 1]))},
                    "nu": float(o21.stability_index()), "period_d": o21.period_days, "e": out["arc"]["elements"]["e"],
                    "tess": {"x": rnd(rot[:, 0]), "y": rnd(rot[:, 1]), "z": rnd(rot[:, 2])}}
    out["elapsed_s"] = time.time() - t_start
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
