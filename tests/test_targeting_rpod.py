import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cislunar import cr3bp, frames, periodic as P, relative as R, targeting as T
from cislunar.constants import EARTH_MOON as S


@pytest.fixture(scope="module")
def nrho():
    return P.nrho((9, 2), "south")


@pytest.fixture(scope="module")
def dro():
    return P.dro(70000.0)


def test_raan_for_target():
    u = np.array([0.3, 0.8, -0.2])
    inc = np.radians(28.5)
    for raan in T.raan_for_target(u, inc):
        r, v = T.twobody.coe2rv(7000, 0, inc, raan, 0, 0, 1.0)
        assert abs(np.dot(np.cross(r, v), u)) < 1e-9


def test_shoot_hits_target(nrho):
    x0 = nrho.state_at(0.0)
    goal = nrho.state_at(0.4)[:3] + 1e-4
    v, xf = T.shoot(x0, goal, 0.4, S)
    assert np.linalg.norm(xf[:3] - goal) < 1e-10


def test_dro_to_nrho_rendezvous(dro, nrho):
    tr = T.intercept(dro, nrho, 0.0, S.nd_time_days(6.0), S)
    # arrival state equals the target state after dv2
    xt = nrho.state_at(tr.t_depart + tr.tof)
    assert np.allclose(tr.x_arrive[:3], xt[:3], atol=1e-9)
    assert np.allclose(tr.x_arrive[3:] + tr.dv2, xt[3:], atol=1e-12)
    assert 100 < tr.total_mps < 3000


def test_cw_stm_against_integration():
    n = 1e-3
    x0 = np.array([0.1, 0.2, -0.3, 0.01, -0.02, 0.005])

    def f(t, y):
        x, yy, z, vx, vy, vz = y
        return [vx, vy, vz, 2 * n * vz, -n * n * yy, 3 * n * n * z - 2 * n * vx]

    sol = solve_ivp(f, (0, 3000), x0, rtol=1e-12, atol=1e-14)
    assert np.allclose(sol.y[:, -1], R.cw_stm(n, 3000) @ x0, atol=1e-9)


def test_cw_two_impulse():
    n = 1e-3
    rel0 = np.array([-10.0, 0, 1.0, 0, 0, 0])
    dv1, dv2 = R.cw_two_impulse(rel0, [-1.0, 0, 0], 2000.0, n)
    x = rel0.copy()
    x[3:] += dv1
    xf = R.cw_stm(n, 2000.0) @ x
    assert np.allclose(xf[:3], [-1, 0, 0], atol=1e-10)
    assert np.allclose(xf[3:] + dv2, 0, atol=1e-12)


def test_waypoint_plan(nrho):
    xt0 = nrho.state_at(0.0)
    rel0 = np.concatenate((S.nd_length(np.array([-10.0, 0, 0.5])), np.zeros(3)))
    xc0 = frames.from_lvlh(xt0, rel0, S)
    wps = [R.Waypoint(6, (-2, 0, 0), "a"), R.Waypoint(10, (-0.1, 0, 0), "b"),
           R.Waypoint(11, (0, 0, 0), "dock")]
    plan = R.plan_waypoints(xt0, xc0, wps, S, final_rhodot_mps=(0.02, 0, 0))
    for wp, (_, xt, xc) in zip(wps, plan.leg_states[1:]):
        rho = S.km(frames.to_lvlh(xt, xc, S)[:3])
        assert np.allclose(rho, wp.rho_km, atol=1e-5)  # < 1 cm
    _, xt, xc = plan.leg_states[-1]
    assert np.allclose(S.mps(frames.to_lvlh(xt, xc, S)[3:]), [0.02, 0, 0], atol=1e-6)
    assert len(plan.maneuvers) == 4
    assert 0 < plan.total_dv_mps < 5


def test_linear_matches_nonlinear_for_small_offsets(nrho):
    xt = nrho.state_at(0.0)
    rel0 = np.concatenate((S.nd_length(np.array([-1.0, 0.1, 0.2])), np.zeros(3)))
    tof = S.nd_time_seconds(3 * 3600)
    rho_f = S.nd_length(np.array([-0.1, 0, 0]))
    dv_lin, _ = R.linear_relative_targeting(xt, rel0, rho_f, tof, S)
    xc = frames.from_lvlh(xt, rel0, S)
    goal = frames.from_lvlh(cr3bp.propagate_state(xt, tof, S),
                            np.concatenate((rho_f, np.zeros(3))), S)[:3]
    v, _ = T.shoot(xc, goal, tof, S, tol=1e-13)
    dv_nl = frames.rotating_vector_to_lvlh(xt, v - xc[3:], S)
    assert np.allclose(S.mps(dv_lin), S.mps(dv_nl), atol=1e-4)


def test_safety_checks():
    t = np.linspace(0, 1, 5)
    rho = np.array([[-1, 0, 0], [-0.5, 0, 0.3], [-0.1, 0, 0.05], [-0.05, 0, 0], [0, 0, 0]])
    rel = R.RelativeTrajectory(t, np.hstack((S.nd_length(rho), np.zeros((5, 3)))), S)
    assert list(R.keep_out_violations(rel, 0.2)) == [2, 3, 4]
    assert list(R.corridor_violations(rel, (-1, 0, 0), 15.0, 0.6)) == [1, 2]


def test_porkchop_refine_never_worse(dro, nrho):
    t_dep = np.linspace(0.0, S.nd_time_days(2.0), 3)
    tofs = np.linspace(S.nd_time_days(4.0), S.nd_time_days(6.0), 3)
    base = T.porkchop(dro, nrho, t_dep, tofs, S, keep_transfers=True)
    ref = T.porkchop(dro, nrho, t_dep, tofs, S, keep_transfers=True, refine=1)
    ok = np.isfinite(base.total_mps)
    assert np.all(ref.total_mps[ok] <= base.total_mps[ok] + 1e-6)
    assert ref.best.total_mps <= base.best.total_mps + 1e-6
    with pytest.raises(ValueError):
        T.porkchop(dro, nrho, t_dep[:1], tofs[:1], S, refine=1)
