"""Forward models of the candidate rocket stages for the 2010 KQ case study.

For each suspect launch the stage is put on a lunar transfer that starts in the real
parking orbit (plane fixed by the launch time, the launch-site latitude and the
51.6 deg inclination these vehicles flew), aimed at the Moon on the known arrival
date, and then given a spread of lunar miss vectors (the stage separates before any
mid-course correction, and makes none). Each variant is flown forward in the full
model (Sun, Moon, planets, relativity, sunlight pressure) and its fate recorded.
Used by docs/build_case_study_kq.py.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from cislunar import ephemeris as E
from cislunar.constants import EARTH_MOON as S

R_PARK_KM = 6378.137 + 190.0
INC_DEG = 51.6
GM_E = 398600.435436
# launch sites (geodetic lat, east lon); Baikonur 81/24 (Proton) and 1/5 (Molniya/R-7)
SITES = {"81": (46.07, 62.93), "1/5": (45.92, 63.34)}
SUSPECTS = {
    "luna4": {"label": "Luna 4 (Blok-L)", "launch_utc": (1963, 4, 2, 8, 16, 38), "site": "1/5", "inc": 65.0,
              "arrive_utc": (1963, 4, 5, 13, 25, 0), "known_miss_km": 8400.0, "eph": "past",
              "end": (1975, 12, 31)},
    "luna22": {"label": "Luna 22 (Blok-D)", "launch_utc": (1974, 5, 29, 8, 56, 51), "site": "81",
               "arrive_utc": (1974, 6, 2, 12, 0, 0), "known_miss_km": None, "eph": "past", "end": (1975, 12, 31)},
    "luna23": {"label": "Luna 23 (Blok-D)", "launch_utc": (1974, 10, 28, 14, 30, 32), "site": "81",
               "arrive_utc": (1974, 11, 2, 12, 0, 0), "known_miss_km": None, "eph": "past", "end": (1975, 12, 31)},
}


def jd_utc(t):
    y, mo, d, h, mi, s = t
    return E.jd_from_calendar(y, mo, d + (h + (mi + s / 60) / 60) / 24)


def parking_plane(key):
    """Unit normal of the parking orbit (J2000) and the site direction at launch."""
    c = SUSPECTS[key]
    lat, lon = SITES[c["site"]]
    phi = np.radians(lat)
    jd = jd_utc(c["launch_utc"])
    site = np.asarray(E.station_j2000(lon, np.cos(phi), np.sin(phi) * 0.99331, jd)[0], float)
    site = site / np.linalg.norm(site)
    inc = np.radians(c.get("inc", INC_DEG))
    # normal n with n.site = 0 and n.z = cos(inc); north-east launch (ascending pass)
    z = np.array([0.0, 0.0, 1.0])
    east = np.cross(z, site); east /= np.linalg.norm(east)
    north = np.cross(site, east)
    # velocity direction at the site for azimuth A: cos(inc) = cos(lat) sin(A)
    az = np.arcsin(np.cos(inc) / np.cos(phi))
    vdir = np.sin(az) * east + np.cos(az) * north
    n = np.cross(site, vdir)
    return n / np.linalg.norm(n), site, vdir


def reference_transfer(key, eph, dyn_nomoon):
    """Coplanar TLI from the parking orbit that reaches the Moon's centre at arrival.

    The TLI point is the place on the parking orbit, reached after the shortest
    coast, from which a near-horizontal burn reaches the Moon at the arrival time.
    Returns dict with t_tli, r0, v0 (nondim), moon offset of the plane, etc."""
    c = SUSPECTS[key]
    n, site, vdir = parking_plane(key)
    jd_l = jd_utc(c["launch_utc"])
    best = None
    for dt_h in (-24.0, -18.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0, 24.0):
        b = _transfer_at(key, c, n, site, jd_l, E.utc_to_tdb(jd_utc(c["arrive_utc"])) + dt_h / 24, dyn_nomoon)
        if b and (best is None or b["score"] < best["score"]):
            best = b
    return best


def _transfer_at(key, c, n, site, jd_l, jd_a, dyn_nomoon):
    t_a = float(dyn_nomoon.t_of_jd(jd_a))
    moon = dyn_nomoon.body_state("moon", t_a)[:3]
    m_hat = moon / np.linalg.norm(moon)
    plane_err = float(np.degrees(np.arcsin(abs(m_hat @ n))))
    v_park = np.sqrt(GM_E / R_PARK_KM)
    period = 2 * np.pi * R_PARK_KM / v_park
    # Moon direction projected into the parking plane; the burn point sits a
    # transfer angle (150-180 deg) behind it along the direction of motion
    mp = m_hat - (m_hat @ n) * n
    mp /= np.linalg.norm(mp)
    best = None
    for tr_deg in (175.0, 170.0, 165.0, 160.0, 155.0, 150.0, 145.0):
        ang = -np.radians(tr_deg)
        p = np.cos(ang) * mp + np.sin(ang) * np.cross(n, mp)
        tang = np.cross(n, p)
        # coast from the launch site to p along the orbit (at least 15 min after launch)
        u = np.arctan2(np.cross(site, p) @ n, site @ p) % (2 * np.pi)
        coast_s = u / (2 * np.pi) * period
        while coast_s < 15 * 60:
            coast_s += period
        jd_t = E.utc_to_tdb(jd_l) + coast_s / 86400
        t_t = float(dyn_nomoon.t_of_jd(jd_t))
        r0 = S.state_to_nd(np.concatenate((p * R_PARK_KM, np.zeros(3))))[:3]
        v_g = S.state_to_nd(np.concatenate((np.zeros(3), tang * 10.92)))[3:]
        try:
            v0, miss = E.shoot_velocity(dyn_nomoon, r0, v_g, t_t, t_a, moon, tol_km=1.0)
        except Exception:  # noqa: BLE001 - poor guesses can fail; skip them
            continue
        if miss > 10:
            continue
        vk = S.kms(v0)
        fpa = float(np.degrees(np.arcsin(p @ vk / np.linalg.norm(vk))))
        oop = float(np.degrees(np.arcsin(n @ vk / np.linalg.norm(vk))))
        score = abs(fpa) + abs(oop)
        if best is None or score < best["score"]:
            best = {"score": score, "transfer_deg": tr_deg, "coast_min": coast_s / 60, "jd_tli": jd_t, "t_tli": t_t,
                    "r0": r0, "v0": v0, "v_kms": float(np.linalg.norm(vk)), "fpa_deg": fpa, "out_of_plane_deg": oop,
                    "t_arr": t_a, "jd_arr": jd_a}
    if best:
        best["moon_plane_offset_deg"] = plane_err
    return best


def _variant(args):
    """One miss vector: aim, then fly the full model forward; summarise the fate."""
    key, b_km, theta, amr, dt_arr_h = args
    from build_case_study_kq import long_eph, PLANETS, utc_str  # noqa: PLC0415 - worker import
    c = SUSPECTS[key]
    eph = long_eph()
    jd0 = E.utc_to_tdb(jd_utc(c["launch_utc"]))
    nomoon = E.NBodyDynamics(eph, jd0, moon=False)  # aiming only: 4 days, planets negligible
    full = E.NBodyDynamics(eph, jd0, amr=amr, planets=PLANETS, gr=True)
    ref = references(long_eph, REFS_PATH)[key]
    t_a = ref["t_arr"] + S.nd_time_days(dt_arr_h / 24)
    (xa, _), = nomoon.propagate_many(np.concatenate((ref["r0"], ref["v0"])), ref["t_tli"], [t_a], with_stm=False)
    mst = nomoon.body_state("moon", t_a)
    vrel = xa[3:] - mst[3:]
    k = vrel / np.linalg.norm(vrel)
    e1 = np.cross(k, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(k, e1)
    aim = mst[:3] + (b_km / S.lstar) * (np.cos(theta) * e1 + np.sin(theta) * e2)
    try:
        v0, miss = E.shoot_velocity(nomoon, ref["r0"], ref["v0"], ref["t_tli"], t_a, aim, tol_km=1.0)
    except Exception as e:  # noqa: BLE001 - an aim that cannot be reached is recorded, not fatal
        return {"key": key, "b_km": b_km, "theta": theta, "amr": amr, "dt_arr_h": dt_arr_h, "failed": str(e)}
    if miss > 50:
        return {"key": key, "b_km": b_km, "theta": theta, "amr": amr, "dt_arr_h": dt_arr_h, "failed": f"aim missed by {miss:.0f} km"}
    t_end = float(full.t_of_jd(E.utc_to_tdb(E.jd_from_calendar(*c["end"]))))
    t1 = time.time()
    lo = ref["t_tli"]
    times = np.arange(lo + S.nd_time_days(5.0), t_end, S.nd_time_days(5.0))
    ca, X, imp = full.close_approaches(np.concatenate((ref["r0"], v0)), lo, t_end, r_max_km=(0.1 * 1.496e8, 6.0e4),
                                       body=("earth", "moon"), times=times)
    r = np.linalg.norm(X[:, :3], axis=1) * S.lstar
    v = np.linalg.norm(X[:, 3:], axis=1) * S.vstar
    en = 0.5 * v**2 - GM_E / r
    moon_pass = next((a for a in ca if a["body"] == "moon"), None)
    k30 = np.arange(0, len(times), 6)
    states = [[float(full.jd(times[j]))] + np.round(S.km(X[j, :3]), 1).tolist() + np.round(S.kms(X[j, 3:]), 7).tolist()
              for j in k30 if np.isfinite(X[j, 0])]
    return {"key": key, "b_km": b_km, "theta": theta, "amr": amr, "dt_arr_h": dt_arr_h, "aim_miss_km": miss,
            "moon_pass": ({"date": utc_str(E.tdb_to_utc(moon_pass["jd_tdb"]), "%Y-%m-%d %H:%M"), "r_km": moon_pass["r_km"]}
                          if moon_pass else None),
            "earth": [[a["jd_tdb"], a["r_km"]] for a in ca if a["body"] == "earth"],
            "impact": imp, "jd": full.jd(times).tolist(), "r_km": np.round(r).tolist(), "energy": np.round(en, 4).tolist(),
            "states": states, "seconds": time.time() - t1}


REFS_PATH = Path(__file__).parent / "_cache" / "case_study_kq_refs.json"


def references(eph_loader, path):
    """Reference transfers for every suspect, cached as JSON at ``path``."""
    import json
    from pathlib import Path
    path = Path(path)
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        for v in d.values():
            v["r0"], v["v0"] = np.array(v["r0"]), np.array(v["v0"])
        return d
    out = {}
    for key, c in SUSPECTS.items():
        nomoon = E.NBodyDynamics(eph_loader(), E.utc_to_tdb(jd_utc(c["launch_utc"])), moon=False)
        out[key] = reference_transfer(key, eph_loader(), nomoon)
    path.write_text(json.dumps({k: {q: (w.tolist() if hasattr(w, "tolist") else w) for q, w in v.items()}
                                for k, v in out.items()}), encoding="utf-8")
    return out


def variants(key, n, b_max_km=20000.0, seed=1974, amr=0.008):
    """Miss vectors uniform over a disk of radius b_max_km in the lunar B-plane."""
    rng = np.random.default_rng(seed + sum(map(ord, key)))
    out = []
    for i in range(n):
        b = b_max_km * np.sqrt(rng.uniform(0.0, 1.0))
        out.append((key, float(b), float(rng.uniform(0, 2 * np.pi)), amr, 0.0))
    return out
