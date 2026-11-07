import numpy as np
import pytest

from cislunar import cr3bp, frames, periodic as P, twobody
from cislunar.constants import EARTH_MOON as S
from cislunar.lambert import lambert


@pytest.fixture(scope="module")
def nrho():
    return P.nrho((9, 2), "south")


def test_system_constants():
    assert S.mu == pytest.approx(0.0121506, abs=1e-6)
    assert S.days(2 * np.pi) == pytest.approx(27.28, abs=0.02)  # sidereal month


def test_lagrange_points_are_equilibria():
    for name, p in cr3bp.lagrange_points(S.mu).items():
        x = np.concatenate((p, np.zeros(3)))
        assert np.linalg.norm(cr3bp.eom(0, x, S.mu)) < 1e-12, name
    lp = cr3bp.lagrange_points(S.mu)
    assert lp["L1"][0] == pytest.approx(0.836915, abs=1e-5)
    assert lp["L2"][0] == pytest.approx(1.155682, abs=1e-5)


def test_jacobi_conserved():
    x0 = np.array([0.9, 0.05, 0.02, 0.01, 0.3, -0.05])
    tr = cr3bp.propagate(x0, 3.0, S, n_points=50)
    C = tr.jacobi()
    assert np.ptp(C) < 1e-10


def test_stm_matches_finite_difference():
    x0 = np.array([1.02, 0.0, -0.18, 0.0, -0.1, 0.0])
    _, phi = cr3bp.propagate_state(x0, 0.5, S, with_stm=True)
    fd = np.zeros((6, 6))
    h = 1e-7
    for k in range(6):
        e = np.zeros(6)
        e[k] = h
        fd[:, k] = (cr3bp.propagate_state(x0 + e, 0.5, S)
                    - cr3bp.propagate_state(x0 - e, 0.5, S)) / (2 * h)
    assert np.abs(phi - fd).max() < 1e-5 * max(1.0, np.abs(phi).max())


def test_nrho_9_2(nrho):
    assert nrho.period_days == pytest.approx(29.530589 * 2 / 9, abs=1e-6)
    end = cr3bp.propagate_state(nrho.state0, nrho.period, S)
    assert np.linalg.norm(end - nrho.state0) < 1e-8
    rp, ra = nrho.perilune_apolune_km()
    assert 3000 < rp < 3600 and 69000 < ra < 73000
    assert nrho.state0[2] < 0  # southern: apolune below the plane
    assert nrho.stability_index() < 2.0


@pytest.mark.parametrize("make", [
    lambda: P.lyapunov(1, 5000.0),
    lambda: P.halo(1, 10000.0, "north"),
    lambda: P.halo(2, 10000.0, "south"),
    lambda: P.dro(40000.0),
])
def test_periodicity(make):
    o = make()
    end = cr3bp.propagate_state(o.state0, o.period, S)
    assert np.linalg.norm(end - o.state0) < 1e-8


def test_halo_branch_convention():
    # southern: z < 0 at the crossing farther from the Moon
    for point in (1, 2):
        o = P.halo(point, 10000.0, "south")
        tr = o.trajectory(2000)
        far = np.argmax(np.linalg.norm(tr.r - S.pos2, axis=1))
        assert tr.r[far, 2] < 0


def test_dro_stable():
    assert P.dro(70000.0).stability_index() == pytest.approx(1.0, abs=1e-3)


def test_arclength_passes_fold():
    start = P.halo(2, 5000.0, "south")
    fam = P.continue_arclength(start, 0.01, stop=lambda o: o.period_days < 7.0,
                               direction=(0.0, 1.0, 0.0))
    assert fam[-1].period_days < 7.0


def test_kepler_and_elements_roundtrip():
    mu = 398600.4418
    r0, v0 = twobody.coe2rv(9000, 0.2, 0.7, 1.1, 0.4, 2.0, mu)
    coe = twobody.rv2coe(r0, v0, mu)
    for k, v in dict(a=9000, e=0.2, i=0.7, raan=1.1, argp=0.4, nu=2.0).items():
        assert coe[k] == pytest.approx(v, rel=1e-9)
    T = 2 * np.pi * np.sqrt(9000**3 / mu)
    r, v = twobody.kepler_propagate(r0, v0, T, mu)
    assert np.linalg.norm(r - r0) < 1e-6


def test_lambert_vallado_7_5():
    v1, v2 = lambert([15945.34, 0, 0], [12214.83899, 10249.46731, 0], 76 * 60, 398600.4418)
    assert np.allclose(v1, [2.058913, 2.915965, 0], atol=1e-5)
    assert np.allclose(v2, [-3.451565, 0.910315, 0], atol=1e-5)


@pytest.mark.parametrize("prograde", [True, False])
def test_lambert_3d_consistency(prograde):
    mu = 398600.4418
    r1, r2 = np.array([7000.0, 100, 500]), np.array([-3000.0, 9000, 2000])
    v1, v2 = lambert(r1, r2, 5000.0, mu, prograde=prograde)
    r, v = twobody.kepler_propagate(r1, v1, 5000.0, mu)
    assert np.linalg.norm(r - r2) < 1e-6 and np.linalg.norm(v - v2) < 1e-9


def test_inertial_roundtrip():
    x = np.array([0.9, 0.1, 0.05, 0.02, -0.3, 0.01])
    for c in ("earth", "moon", "barycenter"):
        xi = frames.rotating_to_inertial(1.3, x, S, c)
        assert np.allclose(frames.inertial_to_rotating(1.3, xi, S, c), x, atol=1e-14)


def test_lvlh_roundtrip_and_rate(nrho):
    xt = nrho.state_at(0.3)
    xc = xt + np.array([1e-5, -2e-6, 3e-6, 1e-6, 2e-6, -1e-6])
    rel = frames.to_lvlh(xt, xc, S)
    assert np.allclose(frames.from_lvlh(xt, rel, S), xc, atol=1e-15)
    h = 1e-6
    _, Cd = frames.lvlh_matrix(xt, S, with_rate=True)
    fd = (frames.lvlh_matrix(cr3bp.propagate_state(xt, h, S), S)
          - frames.lvlh_matrix(cr3bp.propagate_state(xt, -h, S), S)) / (2 * h)
    assert np.abs(fd - Cd).max() < 1e-8 * max(1.0, np.abs(Cd).max())
    C = frames.lvlh_matrix(xt, S)
    assert np.allclose(C @ C.T, np.eye(3), atol=1e-14)
    # z axis (R-bar) points to the Moon
    assert np.dot(C[2], S.pos2 - xt[:3]) > 0


def test_continue_to_return_path():
    o = P.lyapunov(1, 2000.0)
    path = P.continue_to(o, P.IX, o.state0[0] + S.nd_length(6000.0), S.nd_length(1000.0),
                         free=(P.IVY,), targets=(P.IVX,), return_path=True)
    xs = [p.state0[0] for p in path]
    assert len(path) > 2 and np.all(np.diff(xs) > 0)
    # stays on the (strongly unstable) Lyapunov family
    assert path[-1].stability_index() > 100


@pytest.mark.parametrize("p,q,e", [(3, 1, None), (2, 1, 0.55), (1, 2, None)])
def test_resonant_orbits(p, q, e):
    o = P.resonant(p, q, e)
    end = cr3bp.propagate_state(o.state0, o.period, S)
    assert np.linalg.norm(end - o.state0) < 1e-8
    assert o.period / (2 * np.pi * q) == pytest.approx(1.0, abs=0.05)
    tr = o.trajectory(3000)
    if (p, q) == (2, 1):  # TESS-like: apogee near lunar distance, but phased away from the Moon
        assert S.km(np.linalg.norm(tr.r - S.pos1, axis=1).max()) > 360000
        assert S.km(np.linalg.norm(tr.r - S.pos2, axis=1).min()) > 250000
        assert o.stability_index() == pytest.approx(1.0, abs=1e-3)


def test_resonant_homotopy_path():
    # direct correction of this guess fails; the mass homotopy recovers it
    st, T = P.resonant_guess(5, 2, 0.5, S)
    with pytest.raises(RuntimeError):
        P.correct_symmetric_period(st, T, S, free=(P.IVY,))
    o = P.resonant(5, 2, 0.5)
    assert np.linalg.norm(cr3bp.propagate_state(o.state0, o.period, S) - o.state0) < 1e-7


def test_speed_kms_references():
    from cislunar import targeting as T
    llo = T.keplerian_state_fn(S.r2 + 100.0, 0.0, np.radians(90), 0.0, 0.0, 0.0, S, center="moon")
    x = llo(0.37)
    assert frames.speed_kms(x, S, "moon") == pytest.approx(np.sqrt(S.gm2 / (S.r2 + 100.0)), rel=1e-9)
    # in an equatorial (prograde) orbit the frame's rotation subtracts directly from the
    # inertial speed: |v_rot| = |v_in| - omega * r  (omega * r ~ 4.9 m/s at 100 km altitude)
    eq = T.keplerian_state_fn(S.r2 + 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, S, center="moon")(0.37)
    omega_r = (S.r2 + 100.0) / S.tstar
    assert frames.speed_kms(eq, S, "moon") - frames.speed_kms(eq, S, "rotating") == pytest.approx(omega_r, rel=1e-6)
    # inertial speed about a body is the same as |v| of the inertial trajectory
    tr = P.nrho().trajectory(200)
    ti = frames.trajectory_to_inertial(tr, "moon")
    assert np.allclose(frames.speed_kms(tr.x, S, "moon"), S.kms(np.linalg.norm(ti.v, axis=1)))


def test_speed_heatmap_plots(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from cislunar.viz import interactive_3d, plot_trajectories_2d, plot_trajectories_3d
    trs = [P.nrho().trajectory(300), P.dro(70000.0).trajectory(300)]
    ax = plot_trajectories_2d(trs, "xy", S, origin="moon", color_by="speed", speed_scale="log")
    assert len(ax.collections) >= 2
    ax3 = plot_trajectories_3d(trs[0], S, origin="moon", color_by="speed", speed_ref="rotating")
    assert len(ax3.collections) >= 1
    fig = interactive_3d(trs, str(tmp_path / "s.html"), S, origin="moon", color_by="speed")
    assert np.ndim(fig.data[0].line.color) == 1
    with pytest.raises(ValueError):
        plot_trajectories_2d(trs, color_by="altitude")
