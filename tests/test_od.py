import numpy as np
import pytest

from cislunar import cr3bp, od
from cislunar import periodic as P
from cislunar.constants import EARTH_MOON as S


@pytest.fixture(scope="module")
def catalog():
    return od.OrbitCatalog.build(families=("L2 Lyapunov", "DRO"), n_members=15, n_phase=256)


@pytest.fixture(scope="module")
def dro_truth():
    d = P.dro(70000.0)
    return d.state_at(0.2 * d.period)


def test_measurement_models_consistent():
    """Analytic range-rate and angle rates equal time derivatives of range and angles."""
    x0 = P.nrho().state_at(0.3)
    site = od.Observer.ground_site(32.0, -110.0, 2.0)
    dt = 1e-5
    xs = [cr3bp.propagate_state(x0, d) for d in (-dt, 0.0, dt)]
    t = lambda d: S.days(d)  # noqa: E731
    ang = [od.obs_angles(t(d), observer=site).predict(x)[0] for d, x in zip((-dt, 0, dt), xs)]
    rng = [od.obs_range(t(d), observer=site).predict(x)[0] for d, x in zip((-dt, 0, dt), xs)]
    rate = od.obs_angle_rates(0.0, observer=site).predict(xs[1])[0]
    rdot = od.obs_range_rate(0.0, observer=site).predict(xs[1])[0]
    assert rate == pytest.approx((ang[2] - ang[0]) / (2 * dt), rel=1e-6)
    assert rdot[0] == pytest.approx((rng[2][0] - rng[0][0]) / (2 * dt), rel=1e-6)
    # the site sits on the Earth's surface and moves at omega * R * cos(lat) inertially
    from cislunar import frames
    xin = frames.rotating_to_inertial(0.4, site.state(0.4), S, "earth")
    assert S.km(np.linalg.norm(xin[:3])) == pytest.approx(6378.137 + 2.0, rel=1e-9)
    v = S.kms(np.linalg.norm(xin[3:])) * 1000
    assert v == pytest.approx(od.OMEGA_EARTH * (6380.137e3) * np.cos(np.radians(32.0)), rel=1e-6)
    # Jacobian matches a direct finite difference
    o = od.obs_angles(0.0, observer=site)
    J = o.jacobian(x0)
    e = np.zeros(6)
    e[4] = 1e-6
    assert J[:, 4] == pytest.approx((o.predict(x0 + e)[0] - o.predict(x0 - e)[0]) / 2e-6, rel=1e-5)


def test_fit_recovers_nrho_from_telescope():
    truth = P.nrho().state_at(1.0)
    site = od.Observer.ground_site(32.0, -110.0)
    tmpl = []
    for td in np.arange(0.0, 2.0, 0.25):
        tmpl += [od.obs_angles(td, sigma_arcsec=1.0, observer=site),
                 od.obs_range_rate(td, sigma_mps=0.5, observer=site)]
    obs = od.simulate(tmpl, truth, rng=1)
    guess = truth + np.concatenate((S.nd_length([150.0, -80.0, 60.0]), S.nd_velocity([0.004, 0.0, -0.003])))
    r = od.fit(obs, guess)
    assert r.converged and not r.prior_used and r.info_rank == 6
    d = r.x - truth
    assert d @ np.linalg.solve(r.P, d) < 22.5  # 99.9% for 6 dof
    assert 0.2 * r.dof < r.chi2 < 3.0 * r.dof
    X, sig = r.predict([0.0, 5.0, 10.0])
    assert sig[0] == pytest.approx(r.sigma_pos_km(), rel=1e-6) and sig[2] > sig[0]


def test_iod_from_two_fixes(dro_truth):
    tm = [od.obs_position(0.0, observer="moon"), od.obs_position(1.5, observer="moon")]
    obs = od.simulate(tm, dro_truth, noise=False)
    g = od.iod_guesses(obs)
    assert any(np.linalg.norm(x - dro_truth) < 1e-7 for _, x in g)
    # position + velocity in inertial axes from the Earth's centre inverts exactly too
    tm = [od.obs_position(0.0, observer="earth", frame="inertial"),
          od.obs_velocity(0.0, observer="earth", frame="inertial")]
    g = od.iod_guesses(od.simulate(tm, dro_truth, noise=False))
    assert np.linalg.norm(g[0][1] - dro_truth) < 1e-12


def test_catalog_interpolation_and_roundtrip(catalog, tmp_path):
    assert set(catalog.families) == {"L2 Lyapunov", "DRO"} and len(catalog) >= 20
    m = len(catalog) // 2
    o = catalog.members[m].orbit()
    taus = np.linspace(0.0, o.period, 301)
    err = max(np.linalg.norm(catalog.states(m, t)[:3] - o.state_at(t)[:3]) for t in taus)
    assert S.km(err) < 2.0
    f = tmp_path / "cat.npz"
    catalog.save(f)
    c2 = od.OrbitCatalog.load(f)
    assert len(c2) == len(catalog) and np.allclose(c2.X, catalog.X)
    assert c2.members[m].describe() == catalog.members[m].describe()


def test_single_fix_is_ambiguous_until_direction_added(catalog, dro_truth):
    fix = od.simulate([od.obs_position(0.0, sigma_km=20.0, observer="moon")], dro_truth, rng=3)
    rep = od.determine_orbit(fix, catalog, n_hypotheses=6)
    assert not rep.determined
    fams = {h.source.split(":")[1].split(",")[0].strip() for h in rep.plausible(0.05)}
    assert fams == {"DRO", "L2 Lyapunov"}
    more = fix + od.simulate([od.obs_velocity_direction(0.0, sigma_deg=3.0, observer="moon")],
                             dro_truth, rng=4)
    rep2 = od.determine_orbit(more, catalog, n_hypotheses=6)
    assert [h.match.family for h in rep2.plausible(0.05)] == ["DRO"]
    assert rep2.best.match.stable and rep2.best.match.dr_km < 500


def test_determined_fit_match_and_fate(catalog, dro_truth):
    tm = [od.obs_position(t, sigma_km=2.0, observer="moon") for t in (0.0, 1.0, 2.0, 3.0)]
    rep = od.determine_orbit(od.simulate(tm, dro_truth, rng=7), catalog)
    b = rep.best
    assert rep.determined and not b.prior_used
    assert S.km(np.linalg.norm(b.x[:3] - dro_truth[:3])) < 10.0
    assert b.match.family == "DRO" and b.match.stable and b.match.dr_km < 50.0
    fa = od.assess_fate(b, n_samples=8, horizon_days=30.0)
    assert fa["p"]["captured"] == 1.0
