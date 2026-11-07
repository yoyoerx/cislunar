import numpy as np
import pytest

import cislunar as cl
from cislunar import cr3bp, manifolds as M, navigation as N, periodic as P, targeting as T
from cislunar.budget import DeltaVBudget

S = cl.EARTH_MOON


@pytest.fixture(scope="module")
def lyap1():
    return P.lyapunov(1, 20000.0)


@pytest.fixture(scope="module")
def nrho():
    return P.nrho()


@pytest.fixture(scope="module")
def dro():
    return P.dro(70000.0)


# ----------------------------------------------------------------- manifolds

def test_eigenvalues_reciprocal(lyap1):
    (lu, vu), (ls, vs) = M.eigen_directions(lyap1)
    assert lu > 100 and lu * ls == pytest.approx(1.0, rel=1e-4)  # symplectic pair
    assert np.linalg.norm(vu[:3]) == pytest.approx(1.0)
    assert S.days(M.efolding_time(lyap1)) == pytest.approx(1.85, abs=0.05)


def test_l1_interior_unstable_manifold_goes_to_earth(lyap1):
    tube = M.manifold(lyap1, "unstable", "interior", n=12, eps_km=50, tf_days=35, gates=True)
    assert tube.fate_counts()["exit_L1"] == 12
    assert all(f.t > 0 for f in tube.fates)


def test_stable_unstable_mirror_symmetry(lyap1):
    u = M.manifold(lyap1, "unstable", "exterior", n=10, tf_days=30, gates=True)
    s = M.manifold(lyap1, "stable", "exterior", n=10, tf_days=30, gates=True)
    assert u.fate_counts() == s.fate_counts()
    assert all(f.t < 0 for f in s.fates)


def test_stable_orbit_has_no_manifold(dro):
    assert M.efolding_time(dro) == np.inf
    with pytest.raises(ValueError):
        M.manifold(dro)


def test_fate_and_forbidden_states():
    x = M.planar_state(3000.0, 3.0, "prograde")  # 3,000 km from the Moon's centre
    f, tr = M.fate(x, S.nd_time_days(60), S, n_points=200)
    assert f.kind in M.FATES and tr.t[-1] == pytest.approx(f.t)
    assert M.planar_state(80000.0, 3.25) is None  # beyond L2: forbidden at this energy


def test_fate_map_serial_small():
    fm = M.fate_map([-30000.0, 20000.0], [3.0, 3.15], "retrograde", 20.0, with_ftle=True,
                    ftle_horizon_days=5.0, workers=1)
    assert fm.code.shape == (2, 2)
    assert np.all((fm.code >= M.FORBIDDEN) & (fm.code < len(M.FATES)))
    ok = fm.code >= 0
    assert np.all(fm.ftle[ok] > 0) and np.all(fm.lifetime_days[ok] <= 20.0 + 1e-9)


# ----------------------------------------------------------------- flyby

def test_flyby_miss_distance(dro, nrho):
    td, tof = S.nd_time_days(7.33), S.nd_time_days(9.0)
    fb = T.flyby(dro, nrho, td, tof, miss_km=25.0)
    assert fb.miss_km == pytest.approx(25.0, abs=0.1)
    assert 500 < fb.v_rel_mps < 2000
    tr = T.intercept(dro, nrho, td, tof)
    assert tr.v_rel_mps == pytest.approx(tr.dv2_mps, rel=1e-9)  # rendezvous burn = relative speed


# ----------------------------------------------------------------- navigation

@pytest.fixture(scope="module")
def transfer(dro, nrho):
    tr = T.intercept(dro, nrho, S.nd_time_days(5.33), S.nd_time_days(8.36))
    x_post = tr.x_chaser.copy()
    x_post[3:] += tr.dv1
    return tr, x_post


def test_linear_tcm_scales_linearly_and_grows_with_delay(transfer):
    tr, x_post = transfer
    times = S.nd_time_days(np.array([0.5, 2.0, 5.0]))
    a = N.linear_tcm(x_post, tr.tof, N.injection_covariance(0.5), times, S, dv_arrival=tr.dv2)
    b = N.linear_tcm(x_post, tr.tof, N.injection_covariance(1.0), times, S, dv_arrival=tr.dv2)
    assert np.allclose(b.tcm_p99, 2 * a.tcm_p99, rtol=1e-6)
    assert np.all(np.diff(a.tcm_p99) > 0)
    with pytest.raises(ValueError):
        N.linear_tcm(x_post, tr.tof, N.injection_covariance(1.0), [tr.tof * 2], S)


def test_monte_carlo_agrees_with_linear(transfer):
    tr, x_post = transfer
    P0 = N.injection_covariance(0.5)
    t1 = S.nd_time_days(np.array([1.0]))
    lin = N.linear_tcm(x_post, tr.tof, P0, t1, S)
    mc = N.monte_carlo_tcm(x_post, tr.tof, P0, t1, S, n=150, nav_sigma_km=0.01,
                           nav_sigma_mps=1e-4, exec_frac=0.0, exec_floor_mps=0.0)
    assert np.percentile(mc.tcm_mps[:, 0], 99) == pytest.approx(lin.tcm_p99[0], rel=0.25)
    assert np.median(mc.miss_km) < 50.0


def test_execution_covariance_shape():
    P = N.burn_execution_covariance(np.array([1.0, 0, 0]), 0.01, 0.5)
    assert P[3, 3] == pytest.approx(1e-4) and P[4, 4] == pytest.approx(np.tan(np.radians(0.5)) ** 2)
    assert np.all(P[:3, :3] == 0)


def test_stationkeeping(dro):
    halo = P.halo(2, 20000.0, "south")
    r = N.stationkeeping_mc(halo, 2.0, duration_days=60, n=4)
    assert not r.lost.any() and 1.0 < r.per_year_mps.mean() < 30.0
    r_dro = N.stationkeeping_mc(dro, 7.0, duration_days=60, n=3)
    assert np.all(r_dro.n_burns == 0) and np.all(r_dro.dv_total_mps == 0)
    r_lost = N.stationkeeping_mc(P.halo(1, 15000.0, "north"), 20.0, duration_days=60, n=2)
    assert r_lost.lost.all()


# ----------------------------------------------------------------- budget

def test_budget_rollup(transfer):
    tr, _ = transfer
    b = DeltaVBudget("t", margin_frac=0.1).add_transfer(tr, "leg")
    b.add("TCM", 10.0, "statistical").add("SK", 2.0, "station-keeping")
    det = tr.dv1_mps + tr.dv2_mps
    assert b.subtotal("deterministic") == pytest.approx(det)
    assert b.total == pytest.approx(1.1 * (det + 12.0))
    assert "TOTAL" in b.table()
    with pytest.raises(ValueError):
        b.add("x", 1.0, "margin")
