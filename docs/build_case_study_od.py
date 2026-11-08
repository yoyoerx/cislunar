"""Build docs/case_study_change5t1.html: orbit determination from real amateur astrometry.

The Long March 3C third stage from Chang'e 5-T1 (2014-065B, first misidentified as a
SpaceX DSCOVR stage) hit the Moon on 2022-03-04. This script fits the 111 published
observations from 2022 Jan 15-20, compares force models and Find_Orb, forecasts the
impact, and checks it against the crater LRO found.

    python docs/build_case_study_od.py            # uses docs/_cache/case_study_od.json if present
    python docs/build_case_study_od.py --rebuild  # recompute (~5-10 min, parallel; offline except
                                                  # the optional Horizons check of the station model)
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

import cislunar as cl
from cislunar import astrometry as A
from cislunar import ephemeris as E
from cislunar import od, twobody

HERE = Path(__file__).parent
DATA = HERE / "data" / "change5t1"
CACHE = HERE / "_cache" / "case_study_od.json"
TEMPLATE = HERE / "case_study_od_template.html"
OUT = HERE / "case_study_change5t1.html"
S = cl.EARTH_MOON
MU_E = 398600.435436
R_IMPACT = 1737.4 + 1.863  # km: mean radius + LROC site elevation
JD_END = 2459644.0  # 2022-03-05.5 TDB: propagate to here (impact is on 03-04)
WORKERS = 7


def log(m):
    print(m, flush=True)


def rnd(a, nd=0):
    return np.round(np.asarray(a, float), nd).tolist()


def setup(amr=0.0, kind="nbody", **kw):
    recs = A.parse(DATA / "astrometry.txt")
    st = A.load_stations(DATA / "stations.txt")
    eph = E.Ephemeris.load(DATA / "ephemeris_2022.json")
    jd0 = E.utc_to_tdb(recs[0].jd_utc)
    dyn = E.CR3BPInFrame(eph, jd0) if kind == "cr3bp" else E.NBodyDynamics(eph, jd0, amr=amr, **kw)
    return recs, st, eph, dyn


def utc_str(jd_utc):
    from datetime import datetime, timedelta
    return (datetime(2000, 1, 1, 12) + timedelta(days=jd_utc - E.JD_J2000)).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- worker tasks (top level for Windows)

def _fit_task(args):
    amr, x0, extra = args
    recs, st, eph, dyn = setup(amr)
    obs = A.to_observations(recs, st, dyn)
    if extra:
        obs = obs + _feb_obs(dyn, st, extra)
    r = od.fit(obs, np.asarray(x0), 0.0, S, dynamics=dyn)
    _, _, imp = dyn.propagate(r.x, 0.0, float(dyn.t_of_jd(JD_END)), n_points=2, moon_radius_km=R_IMPACT)
    return {"amr": amr, "chi2": r.chi2, "n": r.n_meas, "converged": r.converged, "x": r.x.tolist(),
            "impact": None if not imp or imp["body"] != "moon" else
            {"lat": imp["lat"], "lon": imp["lon"], "jd_utc": imp["jd_utc"]}}


def _impact_task(args):
    x, amr = args
    _, _, _, dyn = setup(amr)
    _, _, imp = dyn.propagate(np.asarray(x), 0.0, float(dyn.t_of_jd(JD_END)), n_points=2, moon_radius_km=R_IMPACT)
    return None if not imp or imp["body"] != "moon" else (imp["lat"], imp["lon"], imp["jd_utc"])


# ---------------------------------------------------------------- simulated February follow-up

def _feb_plan(dyn, st, x_truth, codes=("C95", "K19", "718"), days=(2459611.5, 2459621.5)):
    """Times a station could see the object in Feb 1-10: object >20 deg up, Sun < -12 deg.
    Up to 3 exposures per station-night, 20 min apart."""
    jds = np.arange(days[0], days[1], 1.0 / 48.0)
    ts = dyn.t_of_jd(jds)
    states = dyn.propagate_many(x_truth, 0.0, ts, with_stm=False)
    plan = []
    for code in codes:
        last_night = None
        for jd, (x, _) in zip(jds, states):
            jd_utc = E.tdb_to_utc(jd)
            rs, _ = E.station_j2000(*st[code][:3], jd_utc)
            up = rs / np.linalg.norm(rs)
            rho = S.km(x[:3]) - rs
            sun = dyn.eph.state("sun", jd)[0] - rs
            el = np.degrees(np.arcsin(up @ rho / np.linalg.norm(rho)))
            sel = np.degrees(np.arcsin(up @ sun / np.linalg.norm(sun)))
            night = int(jd_utc + 0.5 + st[code][0] / 360.0)
            if el > 20 and sel < -12 and night != last_night:
                plan += [(jd_utc + k * 20 / 1440, code) for k in range(3)]
                last_night = night
    return plan


def _feb_obs(dyn, st, extra):
    """Observations for a plan [(jd_utc, code, ra, dec)]."""
    recs = [A.Record(jd, ra, dec, None, code, "simulated") for jd, code, ra, dec in extra]
    return A.to_observations(recs, st, dyn)


def simulate_feb(x_truth, amr_truth, rng=7):
    recs, st, eph, dyn = setup(amr_truth)
    plan = _feb_plan(dyn, st, x_truth)
    tmpl = A.to_observations([A.Record(jd, 0.0, 0.0, None, c, "plan") for jd, c in plan], st, dyn)
    for o in tmpl:
        o.z = None
    sim = od.simulate(tmpl, x_truth, 0.0, S, rng=rng, dynamics=dyn)
    return [(jd, c, float(o.z[0]), float(o.z[1])) for (jd, c), o in zip(plan, sim)]


# ---------------------------------------------------------------- analysis

def horizons_check(eph):
    """Moon astrometric RA/Dec from station C95: our model vs JPL Horizons (needs network)."""
    from datetime import datetime
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
    q = dict(format="json", COMMAND="'301'", EPHEM_TYPE="OBSERVER", CENTER="'C95@399'",
             START_TIME="'2022-01-17 18:00'", STOP_TIME="'2022-01-20 18:00'", STEP_SIZE="'6h'",
             QUANTITIES="'1'", ANG_FORMAT="'DEG'", CSV_FORMAT="'YES'", EXTRA_PREC="'YES'", OBJ_DATA="'NO'")
    res = json.loads(urlopen(Request(f"{E.HORIZONS_URL}?{urlencode(q)}", headers={"User-Agent": "cislunar"}),
                             timeout=60).read())["result"]
    c95 = A.load_stations(DATA / "stations.txt")["C95"][:3]
    out = []
    for ln in res.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines():
        f = [x.strip() for x in ln.split(",")]
        dt = datetime.strptime(f[0], "%Y-%b-%d %H:%M")
        jd_utc = E.jd_from_calendar(dt.year, dt.month, dt.day + (dt.hour + dt.minute / 60) / 24)
        rs, _ = E.station_j2000(*c95, jd_utc)
        jd = E.utc_to_tdb(jd_utc)
        rho = eph.state("moon", jd)[0] - rs
        for _ in range(3):
            tau = np.linalg.norm(rho) / E.C_KMS
            rho = eph.state("moon", jd - tau / 86400)[0] - rs + eph.state("sun", jd)[1] * tau
        ra = np.degrees(np.arctan2(rho[1], rho[0])) % 360
        de = np.degrees(np.arcsin(rho[2] / np.linalg.norm(rho)))
        out.append({"utc": f[0], "dra": (ra - float(f[3])) * np.cos(np.radians(de)) * 3600,
                    "ddec": (de - float(f[4])) * 3600})
    return out


def elements_with_sigma(res, dyn, jd_tdb):
    t = float(dyn.t_of_jd(jd_tdb))
    (x, phi), = dyn.propagate_many(res.x, 0.0, [t], with_stm=True)
    P = phi @ res.P @ phi.T

    def coe(xx):
        d = S.state_to_dim(xx)
        c = twobody.rv2coe(d[:3], d[3:], MU_E)
        return np.array([c["a"], c["e"], np.degrees(c["i"]), np.degrees(c["raan"]), np.degrees(c["argp"])])

    c0 = coe(x)
    J = np.zeros((5, 6))
    for k in range(6):
        h = 1e-8
        e = np.zeros(6)
        e[k] = h
        J[:, k] = (coe(x + e) - coe(x - e)) / (2 * h)
    sig = np.sqrt(np.diag(J @ P @ J.T))
    return dict(zip(["a_km", "e", "i_deg", "raan_deg", "argp_deg"], c0.tolist())), \
        dict(zip(["a_km", "e", "i_deg", "raan_deg", "argp_deg"], sig.tolist()))


def jacobi_history(t, X, dyn):
    """Jacobi 'constant' in the pulsating Earth-Moon rotating frame built from the real
    Moon at each instant (it would be constant in the CR3BP)."""
    mu = S.mu
    out = []
    for tk, xk in zip(t, X):
        rm, vm = dyn.eph.state("moon", float(dyn.jd(tk)))
        d = np.linalg.norm(rm)
        ex = rm / d
        h = np.cross(rm, vm)
        ez = h / np.linalg.norm(h)
        ey = np.cross(ez, ex)
        w = np.linalg.norm(h) / d**2
        R = np.vstack((ex, ey, ez))
        r = S.km(xk[:3]) - mu * rm  # barycentric
        v = S.kms(xk[3:]) - mu * vm
        rr = R @ r / d
        vrot = R @ (v - np.cross(w * ez, r))
        ddot = rm @ vm / d
        vv = (vrot - ddot * (R @ r) / d) / (d * np.sqrt((S.gm1 + S.gm2) / d**3))
        r1 = np.linalg.norm(rr - [-mu, 0, 0])
        r2 = np.linalg.norm(rr - [1 - mu, 0, 0])
        U = 0.5 * (rr[0] ** 2 + rr[1] ** 2) + (1 - mu) / r1 + mu / r2
        out.append(2 * U - vv @ vv)
    return out


def compute():
    t_start = time.time()
    ref = json.loads((DATA / "reference.json").read_text(encoding="utf-8"))
    crater = ref["crater"]
    recs, st, eph, dyn = setup()
    obs = A.to_observations(recs, st, dyn)
    jd0 = dyn.jd0
    out = {"reference": ref, "n_obs": len(recs), "jd0_tdb": jd0, "utc0": utc_str(recs[0].jd_utc)}
    out["stations"] = {k: {"lon": v[0], "name": v[3], "n": sum(r.station == k for r in recs)} for k, v in st.items()}

    log("station model vs Horizons ...")
    try:
        out["horizons_check"] = horizons_check(eph)
    except Exception as exc:  # offline build
        log(f"  skipped ({exc})")
        out["horizons_check"] = []

    log("Gauss IOD + fit ...")
    picks = [min(recs, key=lambda r: abs(r.jd_utc - E.jd_from_calendar(2022, 1, d))) for d in (17.73, 18.75, 19.79)]
    iod = []
    best = None
    for jdm, r, v in A.gauss_iod(picks, st):
        x = S.state_to_nd(np.concatenate((r, v)))
        x0 = dyn.propagate_many(x, float(dyn.t_of_jd(E.utc_to_tdb(jdm))), [0.0], with_stm=False)[0][0]
        res = od.fit(obs, x0, 0.0, S, dynamics=dyn, source="Gauss IOD")
        iod.append({"r_km": float(np.linalg.norm(r)), "v_kms": float(np.linalg.norm(v)), "rms": res.rms,
                    "iterations": res.iterations, "converged": res.converged})
        if best is None or res.chi2 < best.chi2:
            best = res
    out["iod"] = {"picks": [{"utc": utc_str(p.jd_utc), "station": p.station} for p in picks], "roots": iod}
    ra = A.residuals_arcsec(best, obs)
    out["fit"] = {"chi2": best.chi2, "n": best.n_meas, "rms_arcsec": float(np.sqrt(np.mean(ra**2))),
                  "sigma_pos_km": best.sigma_pos_km(), "sigma_vel_mps": best.sigma_vel_mps(),
                  "x0_km": S.state_to_dim(best.x).tolist()}
    out["residuals"] = {"t_d": rnd([r.jd_utc - recs[0].jd_utc for r in recs], 4), "st": [r.station for r in recs],
                        "dra": rnd(ra[:, 0], 3), "ddec": rnd(ra[:, 1], 3),
                        "ra_deg": rnd(np.degrees([r.ra for r in recs]), 4),
                        "dec_deg": rnd(np.degrees([r.dec for r in recs]), 4), "mag": [r.mag for r in recs]}
    el, sg = elements_with_sigma(best, dyn, 2459600.5)
    out["elements"] = {"ours": el, "ours_sigma": sg}

    log("force-model comparison ...")
    models = []
    for name, kw in (("Two-body Earth", dict(sun=False, moon=False, j2=False)),
                     ("Earth + Moon", dict(sun=False)),
                     ("CR3BP (circular Moon, no Sun)", None),
                     ("Earth + Moon + Sun (+J2)", {})):
        _, _, _, d = setup(kind="cr3bp") if kw is None else setup(**kw)
        o = A.to_observations(recs, st, d)
        r = od.fit(o, best.x, 0.0, S, dynamics=d)
        rr = A.residuals_arcsec(r, o)
        models.append({"name": name, "rms": float(np.sqrt(np.mean(rr**2))), "max": float(np.abs(rr).max()),
                       "dpos_km": float(S.km(np.linalg.norm(r.x[:3] - best.x[:3])))})
        log(f"  {name}: rms {models[-1]['rms']:.2f}\"")
    out["models"] = models

    log("nominal forecast ...")
    t1 = float(dyn.t_of_jd(JD_END))
    tt, X, imp = dyn.propagate(best.x, 0.0, t1, n_points=3000, moon_radius_km=R_IMPACT)
    moon = np.array([S.km(dyn.body_state("moon", t)[:3]) for t in tt])
    out["nominal"] = {"lat": imp["lat"], "lon": imp["lon"], "utc": utc_str(imp["jd_utc"]), "jd_utc": imp["jd_utc"],
                      "speed_kms": imp["speed_kms"], "angle_deg": imp["angle_from_vertical_deg"],
                      "miss_km": E.surface_distance_km(imp["lat"], imp["lon"], crater["lat_deg"], crater["lon_east_deg"])}
    k = slice(None, None, 3)
    out["traj"] = {"t_d": rnd(S.days(tt[k]), 3), "obj": rnd(S.km(X[k, :3]), 0), "moon": rnd(moon[k], 0),
                   "jacobi": rnd(jacobi_history(tt[k], X[k], dyn), 5)}
    # final day, Moon-centred, in the Moon's body-fixed frame (shows the approach to the far side)
    last = tt > tt[-1] - S.nd_time_days(1.0)
    bf = [E.moon_orientation(float(dyn.jd(t))) @ (S.km(x[:3]) - m) for t, x, m in zip(tt[last], X[last], moon[last])]
    out["approach_bf"] = rnd(bf, 0)
    lp = cl.lagrange_points(S.mu)
    out["jacobi_L"] = {k2: float(cl.jacobi(np.concatenate((lp[k2], np.zeros(3))), S.mu)) for k2 in ("L1", "L2", "L3")}

    log("Monte Carlo of the January covariance ...")
    rng = np.random.default_rng(1)
    Xs = best.samples(300, rng)
    with ProcessPoolExecutor(WORKERS) as ex:
        hits = [h for h in ex.map(_impact_task, [(x, 0.0) for x in Xs]) if h]
    hits = np.array(hits)
    out["mc"] = {"lat": rnd(hits[:, 0], 4), "lon": rnd(hits[:, 1], 4),
                 "dt_s": rnd((hits[:, 2] - out["nominal"]["jd_utc"]) * 86400, 1), "n": len(Xs)}

    log("solar radiation pressure scan ...")
    grid = [0.0, 0.002, 0.004, 0.006, 0.008, 0.012, 0.016, 0.02]
    with ProcessPoolExecutor(WORKERS) as ex:
        scan = list(ex.map(_fit_task, [(a, best.x.tolist(), None) for a in grid]))
    out["srp_scan"] = [{k2: v for k2, v in s.items() if k2 != "x"} for s in scan]

    def miss(a):
        r = _fit_task((a, best.x.tolist(), None))
        i = r["impact"]
        return E.surface_distance_km(i["lat"], i["lon"], crater["lat_deg"], crater["lon_east_deg"]), r

    m = minimize_scalar(lambda a: miss(a)[0], bounds=(0.0, 0.008), method="bounded", options={"xatol": 2e-5})
    dmin, rbest = miss(m.x)
    out["srp_match"] = {"amr": float(m.x), "miss_km": dmin, "impact": rbest["impact"], "chi2": rbest["chi2"],
                        "utc": utc_str(rbest["impact"]["jd_utc"])}
    log(f"  crater matched at C_r A/m = {m.x:.5f} m^2/kg (miss {dmin:.1f} km)")

    log("simulated February follow-up ...")
    x_truth = np.array(rbest["x"])
    feb = simulate_feb(x_truth, float(m.x))
    grid2 = [float(m.x) + d for d in (-0.0012, -0.0008, -0.0004, 0.0, 0.0004, 0.0008, 0.0012)]
    with ProcessPoolExecutor(WORKERS) as ex:
        prof = list(ex.map(_fit_task, [(a, x_truth.tolist(), feb) for a in grid2]))
    chi = np.array([p["chi2"] for p in prof])
    c2 = np.polyfit(np.array(grid2) - m.x, chi, 2)
    sig_amr = float(1.0 / np.sqrt(c2[0])) if c2[0] > 0 else float("nan")
    out["feb"] = {"n_obs": len(feb), "nights": sorted({(round(j - 0.5) + 0.5) for j, *_ in feb}).__len__(),
                  "stations": sorted({c for _, c, *_ in feb}), "profile": [{k2: v for k2, v in p.items() if k2 != "x"}
                                                                          for p in prof],
                  "sigma_amr": sig_amr, "first_utc": utc_str(min(j for j, *_ in feb)),
                  "last_utc": utc_str(max(j for j, *_ in feb))}
    jan_chi = np.array([s["chi2"] for s in scan])
    out["jan_profile_dchi2_0_to_0.02"] = float(jan_chi[-1] - jan_chi[0])
    log(f"  {len(feb)} simulated Feb observations -> sigma(C_r A/m) = {sig_amr:.5f}")
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
    css = manual[manual.index("<style>") + len("<style>"):manual.index("</style>")]
    tpl = tpl.replace("/*__CSS__*/", css)
    tpl = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    OUT.write_text(tpl.replace("/*__PLOTLY__*/", get_plotlyjs()), encoding="utf-8")
    log(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
