"""Build docs/case_study_2010kq.html: who launched 2010 KQ?

2010 KQ was found in May 2010 looking like a tiny asteroid in an Earth-like orbit around
the Sun, then deleted from the asteroid catalogue as artificial (sunlight pushes on it
too hard, and its spectrum looks like paint, not rock). Nobody has matched it to a
launch. This study fits the 2010 astrometry, measures the push, runs a cloud of
orbits consistent with the data back to the 1950s, and tests the candidate launches.

    python docs/build_case_study_kq.py            # uses docs/_cache/case_study_kq.json if present
    python docs/build_case_study_kq.py --rebuild  # recompute (~3 h on 7 cores, offline; resumable)
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
from cislunar import od

HERE = Path(__file__).parent
DATA = HERE / "data" / "kq2010"
CACHE = HERE / "_cache" / "case_study_kq.json"
FIT_CACHE = HERE / "_cache" / "case_study_kq_fit.json"
TEMPLATE = HERE / "case_study_kq_template.html"
OUT = HERE / "case_study_2010kq.html"
S = cl.EARTH_MOON
WORKERS = 7
GM_E = 398600.435436
AU = E.AU_KM
PLANETS = ("mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune")
GRID = [0.0, 0.004, 0.006, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.010, 0.012]


def log(m):
    print(m, flush=True)


def rnd(a, nd=0):
    return np.round(np.asarray(a, float), nd).tolist()


def utc_str(jd_utc, fmt="%Y-%m-%d"):
    from datetime import datetime, timedelta
    return (datetime(2000, 1, 1, 12) + timedelta(days=jd_utc - E.JD_J2000)).strftime(fmt)


_LONG = {}


def long_eph(forward=False):
    name = "ephemeris_2010_2050.json.gz" if forward else "ephemeris_1957_2010.json.gz"
    if name not in _LONG:
        _LONG[name] = E.Ephemeris.load(DATA / name)
    return _LONG[name]


def setup(amr=0.0, long=False, forward=False):
    """Records, stations, ephemeris and dynamics. long: planets + relativity with the
    1957-2010 tables (or 2010-2050 with forward=True); else the 2010 Moon/Sun table."""
    st = A.load_stations(DATA / "stations.txt")
    recs = sorted(A.parse(DATA / "astrometry.txt", strict=False), key=lambda r: r.jd_utc)
    long = long or forward
    eph = long_eph(forward) if long else E.Ephemeris.load(DATA / "ephemeris_2010.json")
    dyn = E.NBodyDynamics(eph, E.utc_to_tdb(recs[0].jd_utc), amr=amr, planets=PLANETS if long else (), gr=long)
    return recs, st, eph, dyn


def _profile_task(args):
    amr, x0, mask = args
    recs, st, _, dyn = setup(amr)
    obs = [o for o, m in zip(A.to_observations(recs, st, dyn), mask) if m]
    f = od.fit(obs, np.asarray(x0), 0.0, S, dynamics=dyn, max_iter=80)
    res = A.residuals_arcsec(f, obs)
    return {"amr": amr, "chi2": f.chi2, "rms": float(np.sqrt(np.mean(res**2))), "converged": f.converged,
            "x": f.x.tolist(), "cov": f.P.tolist()}


def reject(f, obs, t_d, span, mask):
    res = A.residuals_arcsec(f, obs)
    tot = np.hypot(res[:, 0], res[:, 1])
    inarc = t_d <= span
    cut = max(2.0, 5 * 1.4826 * np.median(tot[inarc & mask]))
    return mask & ~(inarc & (tot > cut)), tot


def progressive(x, amr, t_d, start_days, mask=None):
    recs, st, _, dyn = setup(amr)
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


def fit_2010():
    """First orbit, progressive fit and sunlight-pressure profile on the 2010 arc."""
    recs, st, eph, dyn = setup()
    t_d = np.array([r.jd_utc - recs[0].jd_utc for r in recs])
    out = {"n_obs": len(recs), "stations": sorted({r.station for r in recs}), "first_utc": utc_str(recs[0].jd_utc),
           "last_utc": utc_str(recs[-1].jd_utc), "arc_days": float(t_d.max()), "jd0_tdb": dyn.jd0}
    first = [r for r, t in zip(recs, t_d) if t < 2.2]
    obs = A.to_observations(first, st, dyn)
    best = None
    t0 = time.time()
    for jdm, r, v in A.gauss_iod([first[0], first[len(first) // 2], first[-1]], st):
        x = dyn.propagate_many(S.state_to_nd(np.concatenate((r, v))), float(dyn.t_of_jd(E.utc_to_tdb(jdm))), [0.0],
                               with_stm=False)[0][0]
        try:
            f = od.fit(obs, x, 0.0, S, dynamics=dyn)
        except Exception as e:  # noqa: BLE001 - a bad root can fail to converge
            log(f"  Gauss root r={np.linalg.norm(r):.0f} km failed: {e}")
            continue
        if best is None or f.chi2 < best.chi2:
            best, info = f, {"method": "Gauss", "r_km": float(np.linalg.norm(r))}
    info.update(n=len(first), days=2.2, seconds=time.time() - t0)
    out["iod"] = info
    log(f"  first orbit: {info}")
    x_g, mask, steps = progressive(best.x, 0.0, t_d, 2.2)
    out["steps_gravity"] = steps
    log(f"  gravity only: {steps[-1]}")
    with ProcessPoolExecutor(WORKERS) as ex:
        prof = list(ex.map(_profile_task, [(a, x_g.tolist(), mask) for a in GRID]))
    k = int(np.argmin([p["chi2"] for p in prof]))
    recs_, st_, _, dyn_b = setup(GRID[k])
    obs_b = A.to_observations(recs_, st_, dyn_b)
    fb = od.fit([o for o, m in zip(obs_b, mask) if m], np.array(prof[k]["x"]), 0.0, S, dynamics=dyn_b, max_iter=60)
    mask, _ = reject(fb, obs_b, t_d, t_d.max() + 1, np.ones(len(obs_b), bool))
    with ProcessPoolExecutor(WORKERS) as ex:
        prof = list(ex.map(_profile_task, [(a, fb.x.tolist(), mask) for a in GRID]))
    chi = np.array([p["chi2"] for p in prof])
    k = int(np.argmin(chi))
    sel = slice(max(k - 2, 0), min(k + 3, len(chi)))
    c2 = np.polyfit(np.array(GRID)[sel], chi[sel], 2)
    amr_hat, sig = float(-c2[1] / (2 * c2[0])), float(1 / np.sqrt(c2[0]))
    log(f"  C_r A/m = {amr_hat:.5f} +/- {sig:.5f} ({mask.sum()} of {len(mask)} kept)")
    final = _profile_task((amr_hat, prof[k]["x"], mask))
    recs_, st_, _, dyn_f = setup(amr_hat)
    obs_f = A.to_observations(recs_, st_, dyn_f)
    ff = od.fit([o for o, m in zip(obs_f, mask) if m], np.array(final["x"]), 0.0, S, dynamics=dyn_f, max_iter=60)
    res = A.residuals_arcsec(ff, obs_f)
    out.update(profile=prof, amr=amr_hat, amr_sigma=sig, n_used=int(mask.sum()), mask=mask.tolist(),
               final={"rms": final["rms"], "chi2": ff.chi2, "x": ff.x.tolist(), "cov": ff.P.tolist(),
                      "x0_km": S.state_to_dim(ff.x).tolist()},
               resid={"t_d": rnd(t_d, 3), "dra": rnd(res[:, 0], 2), "ddec": rnd(res[:, 1], 2), "used": mask.tolist(),
                      "st": [r.station for r in recs]})
    return out


SUN_EARTH_MU = (S.gm1 + S.gm2) / (E.GM_SUN + S.gm1 + S.gm2)
R_HILL = AU * (SUN_EARTH_MU / 3) ** (1 / 3)
SAMPLE_DAYS = 5.0
START_JD = E.jd_from_calendar(1957, 10, 4)  # Sputnik 1: nothing artificial before this
# candidate launches (UTC dates) and what each implies
CANDIDATES = {
    "luna1": {"label": "Luna 1 Blok-E stage", "launch": "1959-01-02", "jd": E.jd_from_calendar(1959, 1, 2.67)},
    "luna4": {"label": "Luna 4 Blok-L stage", "launch": "1963-04-02", "jd": E.jd_from_calendar(1963, 4, 2.36)},
    "luna22": {"label": "Luna 22 Blok-D stage", "launch": "1974-05-29", "jd": E.jd_from_calendar(1974, 5, 29.37)},
    "luna23": {"label": "Luna 23 Blok-D stage", "launch": "1974-10-28", "jd": E.jd_from_calendar(1974, 10, 28.6)},
    "helios1": {"label": "Helios 1 Centaur", "launch": "1974-12-10", "jd": E.jd_from_calendar(1974, 12, 10.3)},
}


def clone_states(fit, n, seed=2010):
    """Draw n (amr, x0) clones consistent with the 2010 data: C_r A/m from its profile
    (normal, mean and sigma from the parabola), the state from the fit at that A/m
    (interpolated between profile fits) plus the fit covariance."""
    rng = np.random.default_rng(seed)
    grid = np.array([p["amr"] for p in fit["profile"]])
    xs = np.array([p["x"] for p in fit["profile"]])
    P = np.array(fit["final"]["cov"])
    out = []
    for _ in range(n):
        a = float(rng.normal(fit["amr"], fit["amr_sigma"]))
        xm = np.array([np.interp(a, grid, xs[:, k]) for k in range(6)])
        out.append((a, (xm + rng.multivariate_normal(np.zeros(6), P)).tolist()))
    return out


def _clone_task(args):
    """Run one clone back to Sputnik: Earth (< 0.1 AU) and Moon (< 50,000 km) close
    approaches, and the geocentric distance and two-body energy every 5 days."""
    i, amr, x0 = args
    t0 = time.time()
    recs, st, eph, dyn = setup(amr, long=True)
    t1 = float(dyn.t_of_jd(START_JD))
    times = np.arange(0.0, t1, -S.nd_time_days(SAMPLE_DAYS))[1:]
    ca, X, imp = dyn.close_approaches(np.array(x0), 0.0, t1, r_max_km=(0.1 * AU, 5.0e4), body=("earth", "moon"),
                                      times=times)
    r = np.linalg.norm(X[:, :3], axis=1) * S.lstar
    v = np.linalg.norm(X[:, 3:], axis=1) * S.vstar
    energy = 0.5 * v**2 - GM_E / r
    app = []
    for c in ca:
        rr, vv = S.km(c["x"][:3]), S.kms(c["x"][3:])
        app.append({"body": c["body"], "jd": c["jd_tdb"], "r_km": c["r_km"],
                    "energy": float(0.5 * vv @ vv - GM_E / np.linalg.norm(rr)) if c["body"] == "earth" else None})
    k30 = np.arange(5, len(times), int(30 / SAMPLE_DAYS))  # full states every 30 days (for elements)
    states = [[float(dyn.jd(times[k]))] + np.round(S.km(X[k, :3]), 1).tolist() + np.round(S.kms(X[k, 3:]), 7).tolist()
              for k in k30 if np.isfinite(X[k, 0])]
    return {"i": i, "amr": amr, "jd": dyn.jd(times).tolist(), "r_km": np.round(r).tolist(),
            "energy": np.round(energy, 4).tolist(), "approaches": app, "impact": imp, "states": states,
            "seconds": time.time() - t0}


RAW_CLONES = HERE / "_cache" / "raw" / "case_study_kq_clones.jsonl"
RAW_FORWARD = HERE / "_cache" / "raw" / "case_study_kq_forward.jsonl"
N_CLONES = 280
N_FORWARD = 140
END_JD = E.jd_from_calendar(2049, 12, 1)


def _forward_task(args):
    """Run one clone forward to 2049: Earth (< 0.1 AU) and Moon (< 50,000 km) approaches."""
    i, amr, x0 = args
    t0 = time.time()
    recs, st, eph, dyn = setup(amr, forward=True)
    ca, x1, imp = dyn.close_approaches(np.array(x0), 0.0, float(dyn.t_of_jd(END_JD)), r_max_km=(0.1 * AU, 5.0e4),
                                      body=("earth", "moon"))
    app = []
    for c in ca:
        rr, vv = S.km(c["x"][:3]), S.kms(c["x"][3:])
        app.append({"body": c["body"], "jd": c["jd_tdb"], "r_km": c["r_km"], "v_kms": float(np.linalg.norm(vv)),
                    "energy": float(0.5 * vv @ vv - GM_E / np.linalg.norm(rr)) if c["body"] == "earth" else None})
    return {"i": i, "amr": amr, "approaches": app, "impact": imp, "seconds": time.time() - t0}


def run_clones(fit, n=N_CLONES, raw=RAW_CLONES, task=None):
    """Run (or resume) a clone cloud; one JSON line per clone in ``raw``. The same
    seed gives the same clones, so the forward cloud reuses the first clones."""
    task = task or _clone_task
    raw.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if raw.exists():
        for ln in raw.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                c = json.loads(ln)
                done[c["i"]] = c
    todo = [(i, a, x) for i, (a, x) in enumerate(clone_states(fit, n)) if i not in done]
    log(f"clones: {len(done)} done, {len(todo)} to run")
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex, raw.open("a", encoding="utf-8") as f:
        for k, c in enumerate(ex.map(task, todo)):
            f.write(json.dumps(c) + "\n")
            f.flush()
            done[c["i"]] = c
            if k % 7 == 6:
                log(f"  {len(done)}/{n} clones, {time.time() - t0:.0f}s")
    return [done[i] for i in sorted(done) if i < n]


N_SUSPECT = {"luna4": 140, "luna22": 200, "luna23": 200}


def run_suspects(n=N_SUSPECT):
    """Fly n lunar-miss variants of each suspect stage forward (resumable, one JSON line each)."""
    import kq_suspects as K
    K.references(long_eph, K.REFS_PATH)  # build the reference transfers once, before the workers
    out = {}
    for key in K.SUSPECTS:
        raw = HERE / "_cache" / "raw" / f"case_study_kq_suspect_{key}.jsonl"
        done = {}
        if raw.exists():
            for ln in raw.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    c = json.loads(ln)
                    done[c["i"]] = c
        nk = n[key] if isinstance(n, dict) else n
        todo = [(i, v) for i, v in enumerate(K.variants(key, nk)) if i not in done]
        log(f"  {key}: {len(done)} done, {len(todo)} to run")
        with ProcessPoolExecutor(WORKERS) as ex, raw.open("a", encoding="utf-8") as f:
            for (i, _), c in zip(todo, ex.map(K._variant, [v for _, v in todo])):
                c["i"] = i
                f.write(json.dumps(c) + "\n")
                f.flush()
                done[i] = c
        out[key] = [done[i] for i in sorted(done) if i < nk]
    return out


def helio_elements(state, eph):
    """Heliocentric a [AU], e, inclination to the Earth's orbit [deg] and Tisserand T
    w.r.t. the Earth, from a stored [jd, x, y, z, vx, vy, vz] geocentric row."""
    jd, r, v = state[0], np.array(state[1:4]), np.array(state[4:7])
    sr, sv = eph.state("sun", jd)
    T, a, e = _tisserand(r - sr, v - sv, -sr, -sv)
    h, hE = np.cross(r - sr, v - sv), np.cross(-sr, -sv)
    inc = float(np.degrees(np.arccos(min(1.0, h @ hE / np.linalg.norm(h) / np.linalg.norm(hE)))))
    return {"a": a, "e": e, "i": inc, "T": T}


def _row_at(states, jd):
    k = int(np.argmin([abs(s_[0] - jd) for s_ in states]))
    return states[k] if abs(states[k][0] - jd) < 20 else None


def escape_date(jd, r, en, after_jd, hold_days=365.0):
    """First sample after ``after_jd`` from which the object stays outside the Hill
    sphere for at least ``hold_days`` (it may come back years later)."""
    jd, r = np.asarray(jd), np.asarray(r, float)
    order = np.argsort(jd)
    jd, r = jd[order], r[order]
    out = np.where(np.isfinite(r), r > R_HILL, False)
    for k in range(len(jd)):
        if jd[k] > after_jd and not out[k - 1 if k else 0] and out[k]:
            w = (jd >= jd[k]) & (jd <= jd[k] + hold_days)
            if out[w].all() and jd[-1] >= jd[k] + min(hold_days, 30):
                return float(jd[k])
    return None


def summarise_suspects(sus, kq76):
    """Fate of every simulated stage, and whether it ends up where 2010 KQ was in 1976."""
    eph = long_eph()
    j76 = E.jd_from_calendar(1976, 1, 1)
    res = {}
    for key, runs in sus.items():
        rows = []
        for c in runs:
            if c.get("failed"):
                continue
            row = {"b_km": c["b_km"], "moon_pass": c["moon_pass"], "impact": c["impact"] and c["impact"]["body"]}
            # first Earth perigee after the lunar pass, and whether still bound on 1975 April 1
            jd_m = E.jd_from_calendar(*[int(x) for x in c["moon_pass"]["date"][:10].split("-")]) if c["moon_pass"] else c["jd"][0]
            per = [x for x in c["earth"] if x[0] > jd_m + 1]
            row["first_perigee"] = [_utc(per[0][0]), per[0][1]] if per else None
            j75 = E.jd_from_calendar(1975, 4, 1)
            k75 = int(np.argmin(np.abs(np.array(c["jd"]) - j75)))
            row["bound_apr75"] = bool(abs(c["jd"][k75] - j75) < 6 and np.isfinite(c["r_km"][k75]) and c["r_km"][k75] < R_HILL
                                      and c["energy"][k75] < 0)
            esc = escape_date(c["jd"], c["r_km"], c["energy"], c["jd"][0] + 1)
            row["escape"] = _utc(esc) if esc else None
            st76 = _row_at(c["states"], j76) if not c["impact"] else None
            row["el76"] = helio_elements(st76, eph) if st76 and esc and esc < j76 else None
            # did it make a 2010 KQ-like pass in April 1975 (< 1 million km, Apr 1 - May 5)?
            pa, pb = E.jd_from_calendar(1975, 4, 1), E.jd_from_calendar(1975, 5, 5)
            p75 = [x for x in c["earth"] if pa <= x[0] <= pb and x[1] < 1.0e6]
            row["pass75"] = [_utc(p75[0][0]), p75[0][1]] if p75 else None
            ea, eb = E.jd_from_calendar(1975, 4, 10), E.jd_from_calendar(1975, 6, 30)
            row["escape75"] = bool(esc and ea <= esc <= eb)
            d = row["el76"]
            row["orbit_like"] = bool(d and abs(d["a"] - kq76["a"][0]) < 0.005 and abs(d["e"] - kq76["e"][0]) < 0.005
                                     and abs(d["i"] - kq76["i"][0]) < 0.1)
            row["match"] = bool(row["pass75"] and row["escape75"] and row["orbit_like"])
            rows.append(row)
        res[key] = rows
    return res


def _validation_task(args):
    """Back-propagate a JPL start state of a natural asteroid and compare with JPL's positions."""
    label, gr = args
    v = json.loads((DATA / "validation_horizons.json").read_text(encoding="utf-8"))["objects"][label]
    st = v["start"]
    dyn = E.NBodyDynamics(long_eph(), st["jd_tdb"], planets=PLANETS, gr=gr)
    x0 = S.state_to_nd(np.concatenate((st["r_km"], st["v_kms"])))
    jds = v["jd_tdb"][::-1]
    X = dyn.propagate_many(x0, 0.0, [float(dyn.t_of_jd(j)) for j in jds], with_stm=False)
    err = [float(np.linalg.norm(S.km(x[:3]) - np.array(r))) for (x, _), r in zip(X, v["r_km"][::-1])]
    return {"label": label, "gr": gr, "date": [utc_str(j) for j in jds], "err_km": rnd(err)}


def tisserand_series(fit):
    """Tisserand parameter w.r.t. the Earth (and heliocentric a, e) of the nominal orbit, yearly."""
    recs, st, eph, dyn = setup(fit["amr"], long=True)
    jds = [E.jd_from_calendar(y, 1, 1) for y in range(2010, 1957, -1)]
    X = dyn.propagate_many(np.array(fit["final"]["x"]), 0.0, [float(dyn.t_of_jd(j)) for j in jds], with_stm=False)
    out = {"date": [], "T": [], "a_au": [], "e": [], "r_geo_km": []}
    for (x, _), j in zip(X, jds):
        sr, sv = eph.state("sun", j)
        r, v = S.km(x[:3]) - sr, S.kms(x[3:]) - sv
        T, a, e = _tisserand(r, v, -sr, -sv)
        out["date"].append(utc_str(j)); out["T"].append(T); out["a_au"].append(a); out["e"].append(e)
        out["r_geo_km"].append(float(np.linalg.norm(S.km(x[:3]))))
    return out


def _tisserand(r, v, rE, vE):
    def el(r, v):
        en = 0.5 * v @ v - E.GM_SUN / np.linalg.norm(r)
        a = -E.GM_SUN / (2 * en)
        h = np.cross(r, v)
        e = np.linalg.norm(np.cross(v, h) / E.GM_SUN - r / np.linalg.norm(r))
        return a, e, h
    a, e, h = el(r, v)
    aE, _, hE = el(rE, vE)
    cosi = min(1.0, h @ hE / np.linalg.norm(h) / np.linalg.norm(hE))
    return float(aE / a + 2 * np.sqrt(a / aE * (1 - e**2)) * cosi), float(a / AU), float(e)


def _utc(jd_tdb, fmt="%Y-%m-%d"):
    return utc_str(E.tdb_to_utc(jd_tdb), fmt)


def summarise_clones(clones, ref):
    """Statistics of the backward cloud."""
    J75a, J75b = E.jd_from_calendar(1975, 3, 1), E.jd_from_calendar(1975, 6, 30)
    g2 = ref["gray_2010_07_06"]["encounter_1975"]
    jg2 = E.jd_from_calendar(1975, 4, 27.0)
    rows = []
    for c in clones:
        ea = [x for x in c["approaches"] if x["body"] == "earth"]
        e75 = [x for x in ea if J75a < x["jd"] < J75b]
        enc = min(e75, key=lambda x: x["r_km"]) if e75 else None
        r = np.array(c["r_km"], float)
        en = np.array(c["energy"], float)
        bound = (r < R_HILL) & (en < 0)
        imp = c["impact"]
        if imp and imp["body"] == "earth":
            cat = "launch"
        elif imp:
            cat = "moon"
        elif bool(bound[-1]):
            cat = "bound1957"
        else:
            cat = "solar"
        pre = [x for x in ea if x["jd"] < E.jd_from_calendar(1974, 1, 1) and x["r_km"] < 0.02 * AU]
        # months spent bound to the Earth before the 1975 encounter (5-day samples, 1972-1975)
        jd = np.array(c["jd"])
        win = (jd > E.jd_from_calendar(1972, 1, 1)) & (jd < (enc["jd"] if enc else J75a))
        w = 1.0
        if enc:
            w = float(np.exp(-0.5 * (((enc["jd"] - jg2) / g2["date_sigma_days"]) ** 2
                                     + ((enc["r_km"] - g2["r_km"]) / g2["r_sigma_km"]) ** 2)))
        # date the clone reaches the Earth's surface or passes within 10,000 km of its centre
        # (a launch read forwards: a real stage's burn starts ~190 km up)
        low = [x for x in ea if x["r_km"] < 1.0e4 and x["jd"] < J75a]
        touch = imp["jd_tdb"] if cat == "launch" else (min(low, key=lambda x: -x["jd"])["jd"] if low else None)
        rows.append({"i": c["i"], "amr": c["amr"], "cat": cat, "touch_date": _utc(touch) if touch else None,
                     "touch_kind": "ground" if cat == "launch" else ("perigee" if touch else None),
                     "enc_date": _utc(enc["jd"]) if enc else None, "enc_r_km": enc["r_km"] if enc else None,
                     "enc_bound": bool(enc and enc["energy"] < 0),
                     "launch_date": _utc(imp["jd_tdb"]) if cat == "launch" else None,
                     "bound_days_before_1975": float(np.sum(bound[win]) * SAMPLE_DAYS),
                     "pre_1974_close": [[_utc(x["jd"]), x["r_km"]] for x in pre],
                     "weight_gray2": w})
    # status over time: bound to Earth / on the ground (before a backward 'launch') / solar orbit
    jd0 = np.array(clones[0]["jd"])
    step = int(30 / SAMPLE_DAYS)
    idx = np.arange(0, len(jd0), step)
    frac = {"date": [_utc(jd0[k]) for k in idx], "bound": [], "ground": [], "solar": []}
    stat = []
    for c in clones:
        r = np.array(c["r_km"], float)
        en = np.array(c["energy"], float)
        s_ = np.where(np.isnan(r), 2, np.where((r < R_HILL) & (en < 0), 0, 1))  # 0 bound, 1 solar, 2 ground
        stat.append(s_[idx])
    stat = np.array(stat)
    for k, name in ((0, "bound"), (2, "ground"), (1, "solar")):
        frac[name] = rnd((stat == k).mean(axis=0), 4)
    # examples: distance histories 1973-07 to 1975-09 for a few clones of each kind
    ex = []
    lo, hi = E.jd_from_calendar(1973, 7, 1), E.jd_from_calendar(1975, 9, 1)
    picks = {}
    for row, c in zip(rows, clones):
        key = row["cat"] if row["cat"] != "launch" else "launch" + row["launch_date"][:4]
        picks.setdefault(key, [])
        if len(picks[key]) < 2:
            picks[key].append(c)
    for key, cs in picks.items():
        for c in cs:
            jd = np.array(c["jd"])
            m = (jd > lo) & (jd < hi)
            ex.append({"i": c["i"], "kind": key, "amr": c["amr"], "date": [_utc(j) for j in jd[m]],
                       "r_km": [None if not np.isfinite(v) else v for v in np.array(c["r_km"], float)[m]]})
    # how many clones touch the Earth in each candidate's launch window (launch to +60 days)
    support = {}
    for key, cnd in CANDIDATES.items():
        a, b = cnd["jd"], cnd["jd"] + 60
        hit = [r for r in rows if r["touch_date"] and a <= E.jd_from_calendar(*map(int, r["touch_date"].split("-"))) <= b]
        support[key] = {"n": len(hit), "weighted": float(sum(r["weight_gray2"] for r in hit) / sum(r["weight_gray2"] for r in rows)),
                        "dates": sorted(r["touch_date"] for r in hit)}
    touched = [r for r in rows if r["touch_date"]]
    support["any"] = {"n": len(touched), "weighted": float(sum(r["weight_gray2"] for r in touched) / sum(r["weight_gray2"] for r in rows))}
    return {"rows": rows, "frac": frac, "examples": ex, "support": support}


def kq_elements(clones):
    """2010 KQ's heliocentric orbit on 1976-01-01 (after the 1975 escape) and on
    1974-01-01 (for clones still in solar orbit then), and its 1975 escape date."""
    eph = long_eph()
    j76, j74 = E.jd_from_calendar(1976, 1, 1), E.jd_from_calendar(1974, 1, 1)
    el76, el74, esc = [], [], []
    for c in clones:
        s76 = _row_at(c["states"], j76)
        if s76:
            el76.append(helio_elements(s76, eph))
        s74 = _row_at(c["states"], j74)
        if s74 and np.linalg.norm(s74[1:4]) > 2 * R_HILL:
            el74.append(helio_elements(s74, eph))
        e = escape_date(c["jd"], c["r_km"], c["energy"], E.jd_from_calendar(1975, 3, 1))
        esc.append(_utc(e) if e else None)
    stats = {k: [float(np.mean([d[k] for d in el76])), float(np.std([d[k] for d in el76]))] for k in ("a", "e", "i", "T")}
    return {"el76": el76, "el74": el74, "escape": esc, "el76_stats": stats}


def luna4_check(runs):
    """Validation: Luna 4 itself missed the Moon by ~8,400 km and was left in an
    ~89,000 x 694,000 km Earth orbit. Report perigee/apogee after the flyby for the
    variants whose lunar pass was within 7,000-10,000 km."""
    out = []
    for c in runs:
        mp = c["moon_pass"]
        if not mp or not 7000 < mp["r_km"] < 10000:
            continue
        r = np.array(c["r_km"], float)
        jd = np.array(c["jd"])
        m = (jd > jd[0] + 8) & (jd < jd[0] + 120) & np.isfinite(r)
        per = [x[1] for x in c["earth"] if jd[0] + 8 < x[0] < jd[0] + 120]
        if m.sum() < 3 or not per or not np.isfinite(r[m]).any():
            continue
        out.append({"moon_r_km": mp["r_km"], "r_min_km": float(min(per)), "r_max_km": float(np.nanmax(r[m]))})
    return out


def summarise_forward(fw):
    rows = []
    a = E.jd_from_calendar(2011, 1, 1)
    for c in fw:
        ea = [x for x in c["approaches"] if x["body"] == "earth" and x["jd"] > a]
        near = [x for x in ea if x["r_km"] < 0.01 * AU]
        first = min(near, key=lambda x: x["jd"]) if near else None
        closest = min(ea, key=lambda x: x["r_km"]) if ea else None
        rows.append({"i": c["i"], "amr": c["amr"],
                     "first_date": _utc(first["jd"]) if first else None, "first_r_km": first["r_km"] if first else None,
                     "closest_date": _utc(closest["jd"]) if closest else None, "closest_r_km": closest["r_km"] if closest else None,
                     "n_perigees_bound": sum(1 for x in ea if x["energy"] < 0 and x["r_km"] < R_HILL),
                     "moon": [[_utc(x["jd"]), x["r_km"]] for x in c["approaches"] if x["body"] == "moon"],
                     "impact": ({"body": c["impact"]["body"], "date": _utc(c["impact"]["jd_tdb"])} if c["impact"] else None)})
    return rows


def _read_raw(path, key="i"):
    out = {}
    if path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                c = json.loads(ln)
                out[c[key]] = c
    return [out[k] for k in sorted(out)]


def compute():
    draft = "--draft" in sys.argv  # use whatever runs are on disk; start none
    t0 = time.time()
    ref = json.loads((DATA / "reference.json").read_text(encoding="utf-8"))
    if FIT_CACHE.exists() and "--refit" not in sys.argv:
        fit = json.loads(FIT_CACHE.read_text(encoding="utf-8"))
    else:
        log("== fitting the 2010 arc ==")
        fit = fit_2010()
        FIT_CACHE.write_text(json.dumps(fit), encoding="utf-8")
    log(f"fit: C_r A/m {fit['amr']:.5f} +/- {fit['amr_sigma']:.5f}, rms {fit['final']['rms']:.3f}")
    log("validation against JPL (4 runs) ...")
    with ProcessPoolExecutor(4) as ex:
        val = list(ex.map(_validation_task, [(l, g) for l in ("1991 VG", "3753 Cruithne") for g in (False, True)]))
    log("Tisserand series ...")
    tis = tisserand_series(fit)
    log("backward clone cloud ...")
    clones = _read_raw(RAW_CLONES) if draft else run_clones(fit)
    log("forward clone cloud ...")
    fw = _read_raw(RAW_FORWARD) if draft else run_clones(fit, N_FORWARD, RAW_FORWARD, _forward_task)
    log("2010 KQ in 1974 and 1976, from the clones ...")
    kq = kq_elements(clones)
    log("suspect stages flown forward ...")
    import kq_suspects as K
    sus = ({k: _read_raw(HERE / "_cache" / "raw" / f"case_study_kq_suspect_{k}.jsonl") for k in K.SUSPECTS} if draft
           else run_suspects())
    refs = K.references(long_eph, K.REFS_PATH)
    q, Q = 0.978, 1.318
    al, el = (q + Q) / 2, (Q - q) / (Q + q)
    out = {"fit": {k: fit[k] for k in ("n_obs", "stations", "first_utc", "last_utc", "arc_days", "iod", "steps_gravity",
                                         "amr", "amr_sigma", "n_used", "resid")},
           "profile": [{k: p[k] for k in ("amr", "chi2", "rms", "converged")} for p in fit["profile"]],
           "rms": fit["final"]["rms"], "ref": ref, "validation": val, "tisserand": tis,
           "luna1_T": float(1 / al + 2 * np.sqrt(al * (1 - el**2))),
           "r_hill_km": R_HILL, "n_clones": len(clones), "n_forward": len(fw),
           "back": summarise_clones(clones, ref), "forward": summarise_forward(fw),
           "candidates": {k: v["launch"] for k, v in CANDIDATES.items()},
           "kq": kq, "suspects": summarise_suspects(sus, kq["el76_stats"]),
           "suspect_refs": {k: {q: v[q] for q in ("transfer_deg", "coast_min", "v_kms", "fpa_deg", "out_of_plane_deg",
                                                  "moon_plane_offset_deg")} | {"arrive": _utc(v["jd_arr"], "%Y-%m-%d %H:%M"),
                                                                              "tli": _utc(v["jd_tli"], "%Y-%m-%d %H:%M")}
                            for k, v in refs.items()},
           "suspect_meta": {k: {"label": v["label"], "launch": "%04d-%02d-%02d" % v["launch_utc"][:3],
                                "known_miss_km": v["known_miss_km"]} for k, v in K.SUSPECTS.items()},
           "luna4_check": luna4_check(sus.get("luna4", []))}
    # distance histories (1973-07 to 1975-09) of the simulated stages that reproduce 2010 KQ
    out["suspect_matches"] = []
    lo, hi = E.jd_from_calendar(1973, 7, 1), E.jd_from_calendar(1975, 9, 1)
    for key, rows in out["suspects"].items():
        live = [c for c in sus[key] if not c.get("failed")]
        for k, (row, c) in enumerate(zip(rows, live)):
            if row["match"]:
                jd = np.array(c["jd"])
                m = (jd > lo) & (jd < hi)
                out["suspect_matches"].append({"key": key, "n": k, "row": row, "date": [_utc(j) for j in jd[m]],
                                               "r_km": [None if not np.isfinite(v) else v for v in np.array(c["r_km"], float)[m]]})
    out["elapsed_s"] = time.time() - t0
    return out


def main():
    if "--clones" in sys.argv:
        run_clones(json.loads(FIT_CACHE.read_text(encoding="utf-8")))
        return
    if "--forward" in sys.argv:
        run_clones(json.loads(FIT_CACHE.read_text(encoding="utf-8")), N_FORWARD, RAW_FORWARD, _forward_task)
        return
    if "--suspects" in sys.argv:
        run_suspects()
        return
    if CACHE.exists() and "--rebuild" not in sys.argv:
        log(f"using cached data {CACHE}")
        data = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        data = compute()
        if "--draft" not in sys.argv:
            CACHE.parent.mkdir(exist_ok=True)
            CACHE.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    if not TEMPLATE.exists():
        log("no template yet; data cached")
        return
    from plotly.offline import get_plotlyjs
    tpl = TEMPLATE.read_text(encoding="utf-8").replace("__VERSION__", cl.__version__)
    manual = (HERE / "manual_template.html").read_text(encoding="utf-8")
    tpl = tpl.replace("/*__CSS__*/", manual[manual.index("<style>") + len("<style>"):manual.index("</style>")])
    texts = (HERE / "_kq_texts.js").read_text(encoding="utf-8")
    verdict = (HERE / "_kq_verdict.js").read_text(encoding="utf-8") if (HERE / "_kq_verdict.js").exists() else ""
    tpl = tpl.replace("/*__TEXTS__*/", texts.replace("/*__VERDICT__*/", verdict))
    tpl = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    OUT.write_text(tpl.replace("/*__PLOTLY__*/", get_plotlyjs()), encoding="utf-8")
    log(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
