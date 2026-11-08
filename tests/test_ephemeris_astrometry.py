import json
from pathlib import Path

import numpy as np
import pytest

from cislunar import astrometry as A
from cislunar import ephemeris as E
from cislunar import od
from cislunar.constants import EARTH_MOON as S

DATA = Path(__file__).parents[1] / "docs" / "data" / "change5t1"


@pytest.fixture(scope="module")
def real():
    recs = A.parse(DATA / "astrometry.txt")
    st = A.load_stations(DATA / "stations.txt")
    eph = E.Ephemeris.load(DATA / "ephemeris_2022.json")
    return recs, st, eph


def test_time_scales():
    jd = E.jd_from_calendar(2022, 1, 18.0)
    assert jd == 2459597.5
    assert (E.utc_to_tdb(jd) - jd) * 86400 == pytest.approx(69.184, abs=0.002)
    assert E.tdb_to_utc(E.utc_to_tdb(jd)) == pytest.approx(jd, abs=1e-10)


def test_parse_both_formats(real):
    recs, st, _ = real
    assert len(recs) == 111 and set(st) == {r.station for r in recs}
    r = recs[0]  # "C2022 01 15.75603 23 58 03.57 -09 30 54.7 ... J57"
    assert r.station == "J57"
    assert r.jd_utc == pytest.approx(E.jd_from_calendar(2022, 1, 15.75603))
    assert np.degrees(r.ra) * 240 == pytest.approx(23 * 3600 + 58 * 60 + 3.57, abs=1e-6)
    assert np.degrees(r.dec) == pytest.approx(-(9 + 30 / 60 + 54.7 / 3600))
    merged = next(x for x in recs if "16.73075800" in x.line)  # 6-decimal date runs into RA
    assert merged.jd_utc == pytest.approx(E.jd_from_calendar(2022, 1, 16.730758))
    assert np.degrees(merged.ra) * 240 == pytest.approx(11 * 60 + 11.73, abs=1e-6)
    hp = next(x for x in recs if "CK220120:18340858" in x.line)  # high-precision form
    assert hp.jd_utc == pytest.approx(E.jd_from_calendar(2022, 1, 20 + (18 + 34 / 60 + 8.58 / 3600) / 24))
    assert np.degrees(hp.ra) == pytest.approx(72.84231) and np.degrees(hp.dec) == pytest.approx(18.88656)


def test_station_position_matches_horizons(real):
    _, st, _ = real
    # JPL Horizons: Earth centre relative to MPC site C95 at 2022-01-18 00:00 TDB (fetched 2026-11-07)
    horizons = -np.array([-2464.70891649, 3879.56171464, 4408.30743985])
    jd_utc = E.tdb_to_utc(2459597.5)
    rs, vs = E.station_j2000(*st["C95"][:3], jd_utc)
    assert np.linalg.norm(rs + horizons) < 0.1  # km
    assert np.linalg.norm(vs) == pytest.approx(E.OMEGA_EARTH * 6378.137 * st["C95"][1], rel=1e-3)


def test_ephemeris_interpolation_and_moon_frame(real):
    _, _, eph = real
    j, r, v = eph.tables["moon"]
    k = 200
    p, vel = eph.state("moon", j[k])
    assert np.allclose(p, r[k]) and np.allclose(vel, v[k])
    # halfway between nodes: velocity is consistent with the position derivative
    jm = 0.5 * (j[k] + j[k + 1])
    pa, _ = eph.state("moon", jm - 1e-4)
    pb, _ = eph.state("moon", jm + 1e-4)
    assert np.allclose((pb - pa) / (2e-4 * 86400), eph.state("moon", jm)[1], rtol=1e-5, atol=1e-6)
    M = E.moon_orientation(j[k])
    assert np.allclose(M @ M.T, np.eye(3), atol=1e-12)
    # the Moon keeps roughly the same face to the Earth: sub-Earth point within ~8 deg of (0, 0)
    lat, lon, _ = E.selenographic(-p, j[k])
    assert abs(lat) < 8 and min(lon, 360 - lon) < 9


def test_srp_points_away_from_sun(real):
    _, _, eph = real
    d0 = E.NBodyDynamics(eph, 2459600.0)
    d1 = E.NBodyDynamics(eph, 2459600.0, amr=0.01)
    r = S.nd_length(np.array([300000.0, 100000.0, 50000.0]))
    da = d1.accel(0.0, r) - d0.accel(0.0, r)
    away = r - d0.body_state("sun", 0.0)[:3]
    assert da @ away > 0.999 * np.linalg.norm(da) * np.linalg.norm(away)
    acc = np.linalg.norm(da) * S.lstar / S.tstar**2 * 1000  # m/s^2
    assert acc == pytest.approx(4.56e-6 * 0.01, rel=0.05)


def test_gauss_iod_two_body():
    st = {"XXX": (10.0, 0.75, 0.66, "test")}
    from cislunar import twobody
    r0, v0 = twobody.coe2rv(300000.0, 0.85, np.radians(27), np.radians(18), np.radians(149), np.radians(170), 398600.4418)
    recs = []
    for dt in (-0.4, 0.0, 0.4):
        jd = 2459597.5 + dt
        r, v = twobody.kepler_propagate(r0, v0, dt * 86400, 398600.4418)
        rs, _ = E.station_j2000(*st["XXX"][:3], jd)
        rho = r - rs
        recs.append(A.Record(jd, float(np.arctan2(rho[1], rho[0])), float(np.arcsin(rho[2] / np.linalg.norm(rho))),
                             None, "XXX", ""))
    sols = A.gauss_iod(recs, st, mu_kms=398600.4418)
    err = min(np.linalg.norm(r - r0) / np.linalg.norm(r0) for _, r, _ in sols)
    assert err < 0.02


def test_real_fit_matches_find_orb(real):
    recs, st, eph = real
    dyn = E.NBodyDynamics(eph, E.utc_to_tdb(recs[0].jd_utc))
    obs = A.to_observations(recs, st, dyn)
    picks = [min(recs, key=lambda r: abs(r.jd_utc - E.jd_from_calendar(2022, 1, d))) for d in (17.73, 18.75, 19.79)]
    jdm, r, v = A.gauss_iod(picks, st)[0]
    x = dyn.propagate_many(S.state_to_nd(np.concatenate((r, v))), float(dyn.t_of_jd(E.utc_to_tdb(jdm))), [0.0],
                           with_stm=False)[0][0]
    res = od.fit(obs, x, 0.0, S, dynamics=dyn)
    rms = np.sqrt(np.mean(A.residuals_arcsec(res, obs) ** 2))
    assert res.converged and rms < 0.8  # Find_Orb: 0.731" mean residual
    from cislunar import twobody
    (x21, _), = dyn.propagate_many(res.x, 0.0, [float(dyn.t_of_jd(2459600.5))], with_stm=False)
    d = S.state_to_dim(x21)
    c = twobody.rv2coe(d[:3], d[3:], 398600.435436)
    ref = json.loads((DATA / "reference.json").read_text(encoding="utf-8"))["find_orb_elements_jan_arc"]
    assert c["a"] == pytest.approx(ref["a_km"], abs=3 * ref["a_sigma_km"])
    assert c["e"] == pytest.approx(ref["e"], abs=3 * ref["e_sigma"])
    assert np.degrees(c["i"]) == pytest.approx(ref["i_deg"], abs=3 * ref["i_sigma_deg"])
    assert np.degrees(c["raan"]) == pytest.approx(ref["raan_deg"], abs=3 * ref["raan_sigma_deg"])


def test_parse_cmos_and_packed_dec_lines():
    lines = ["     K24P05T KB2024 08 12.89051518 09 43.90 -02 45 25.0          17.2 GX~8AlgI93",
             "K20S00O  C2020 11 04.35919802 01 55.527+02 26 33.69   .50   19.77iU~4L64F52",
             "K20S00O  v2020 12 01.0346751 016.33606  +48.20451    301           ~4ORV247"]
    skipped = []
    recs = A.parse(lines, strict=False, skipped=skipped)
    assert len(recs) == 2 and len(skipped) == 1 and [r.station for r in recs] == ["I93", "F52"]
    assert np.degrees(recs[1].ra) * 240 == pytest.approx(2 * 3600 + 1 * 60 + 55.527, abs=1e-6)
    assert np.degrees(recs[1].dec) == pytest.approx(2 + 26 / 60 + 33.69 / 3600)
    with pytest.raises(ValueError):
        A.parse(lines)


def test_range_scan_iod_recovers_distant_object(real):
    """An object ~1 million km away on a Sun-dominated path: the range scan finds it."""
    _, st, eph = real
    dyn = E.NBodyDynamics(eph, 2459600.0)
    r0 = np.array([700000.0, 600000.0, 250000.0])
    v0 = np.array([-0.35, 0.55, 0.10])
    x_true = S.state_to_nd(np.concatenate((r0, v0)))
    jds = [2459600.0 + d for d in (0.0, 0.02, 0.9, 0.92, 1.9, 1.92)]
    recs = []
    for jd in jds:
        (x, _), = dyn.propagate_many(x_true, 0.0, [float(dyn.t_of_jd(jd))], with_stm=False)
        jd_utc = E.tdb_to_utc(jd)
        rs, _ = E.station_j2000(*st["K19"][:3], jd_utc)
        rho = S.km(x[:3]) - rs
        recs.append(A.Record(jd_utc, float(np.arctan2(rho[1], rho[0])), float(np.arcsin(rho[2] / np.linalg.norm(rho))),
                             None, "K19", ""))
    sols = A.range_scan_iod(recs, st, dyn, ranges_km=np.geomspace(3e5, 3e6, 6), top=1)
    f, _, _ = sols[0]
    assert S.km(np.linalg.norm(f.x[:3] - x_true[:3])) < 5000.0  # angles over 2 days; light-time ignored in truth


KQ = Path(__file__).parents[1] / "docs" / "data" / "kq2010"


def test_leap_seconds_back_to_1972():
    assert E.tai_minus_utc(E.jd_from_calendar(2010, 5, 16.0)) == 34.0
    assert E.tai_minus_utc(E.jd_from_calendar(1975, 4, 20.0)) == 14.0
    assert E.tai_minus_utc(E.jd_from_calendar(1963, 4, 2.0)) == 10.0  # pre-1972 approximation


def test_gzip_ephemeris_round_trip(tmp_path, real):
    eph = real[2]
    p = tmp_path / "e.json.gz"
    eph.save(p)
    back = E.Ephemeris.load(p)
    jd = eph.tables["moon"][0][10] + 0.37
    assert np.allclose(back.state("moon", jd)[0], eph.state("moon", jd)[0])


def test_relativity_term_is_relative_to_the_earth(real):
    """The Earth-centred frame needs the Sun's GR pull on the object minus that on the Earth:
    zero for an object riding with the Earth, ~1.8e-13 km/s^2 one AU-scale away."""
    eph = real[2]
    dyn = E.NBodyDynamics(eph, eph.tables["sun"][0][5], gr=True)
    assert np.linalg.norm(dyn._gr_accel(0.0, np.zeros(3), np.zeros(3))) < 1e-25
    sun = dyn.body_state("sun", 0.0)
    far = dyn._gr_accel(0.0, 2 * sun[:3], 2 * sun[3:]) * S.lstar / S.tstar**2  # object across the Sun
    assert 3.0e-13 < np.linalg.norm(far) < 4.5e-13


def test_close_approach_finds_2010kq_perigee():
    """Fit the 2010 KQ arc (sunlight pressure fixed at the published 0.0077 m^2/kg) and recover
    the perigee that Find_Orb reports: 2010 May 21.30 TT at 492,283 km."""
    recs = sorted(A.parse(KQ / "astrometry.txt"), key=lambda r: r.jd_utc)
    st = A.load_stations(KQ / "stations.txt")
    eph = E.Ephemeris.load(KQ / "ephemeris_2010.json")
    dyn = E.NBodyDynamics(eph, E.utc_to_tdb(recs[0].jd_utc), amr=0.0077)
    obs = A.to_observations(recs, st, dyn)
    first = [r for r in recs if r.jd_utc - recs[0].jd_utc < 2.2]
    jdm, r, v = min(A.gauss_iod([first[0], first[len(first) // 2], first[-1]], st),
                    key=lambda g: abs(np.linalg.norm(g[1]) - 5.5e5))
    x = dyn.propagate_many(S.state_to_nd(np.concatenate((r, v))), float(dyn.t_of_jd(E.utc_to_tdb(jdm))), [0.0],
                           with_stm=False)[0][0]
    f = od.fit([o for o, rr in zip(obs, recs) if rr.jd_utc - recs[0].jd_utc < 2.2], x, 0.0, S, dynamics=dyn)
    f = od.fit(obs, f.x, 0.0, S, dynamics=dyn, max_iter=60)
    t_end = float(dyn.t_of_jd(E.utc_to_tdb(recs[-1].jd_utc)))
    ca, _, imp = dyn.close_approaches(f.x, 0.0, t_end, r_max_km=1e6)
    assert imp is None and len(ca) == 1
    assert ca[0]["jd_tdb"] - 2400000.5 == pytest.approx(55337.301886, abs=0.05)
    assert ca[0]["r_km"] == pytest.approx(492283, abs=1500)
