"""Rendezvous, proximity operations and docking (RPOD) tools.

Relative states are expressed in the target's LVLH frame (see frames.py):
    x = V-bar, y = -H-bar, z = R-bar (toward the reference primary).

Two dynamics levels:
* Nonlinear: both vehicles propagated in the CR3BP, relative state computed
  exactly. Used by the waypoint planner (with nonlinear shooting).
* Linear: the target's CR3BP STM (valid in any cislunar orbit), or the
  Clohessy-Wiltshire STM (circular two-body orbit; LEO/LLO baseline).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import cr3bp, frames
from .constants import EARTH_MOON, System
from .targeting import shoot


# ---------------------------------------------------------------------------
# Clohessy-Wiltshire (circular orbit), LVLH convention above
# ---------------------------------------------------------------------------

# Hill frame (radial-out, along-track, orbit-normal) -> LVLH (V-bar, -H-bar, R-bar)
_P_HILL_TO_LVLH = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])


def cw_stm(n: float, t: float) -> np.ndarray:
    """Clohessy-Wiltshire state transition matrix in LVLH for mean motion n."""
    s, c = np.sin(n * t), np.cos(n * t)
    nt = n * t
    phi = np.array([
        [4 - 3 * c, 0, 0, s / n, 2 * (1 - c) / n, 0],
        [6 * (s - nt), 1, 0, -2 * (1 - c) / n, (4 * s - 3 * nt) / n, 0],
        [0, 0, c, 0, 0, s / n],
        [3 * n * s, 0, 0, c, 2 * s, 0],
        [-6 * n * (1 - c), 0, 0, -2 * s, 4 * c - 3, 0],
        [0, 0, -n * s, 0, 0, c],
    ])
    P = np.zeros((6, 6))
    P[:3, :3] = P[3:, 3:] = _P_HILL_TO_LVLH
    return P @ phi @ P.T


def cw_two_impulse(rel0, rho_f, tof: float, n: float, rhodot_f=(0.0, 0.0, 0.0)):
    """Two-impulse CW transfer from relative state rel0 (6,) to position rho_f
    with final relative velocity rhodot_f. Returns (dv1, dv2) in LVLH."""
    rel0 = np.asarray(rel0, float)
    phi = cw_stm(n, tof)
    Prr, Prv, Pvr, Pvv = phi[:3, :3], phi[:3, 3:], phi[3:, :3], phi[3:, 3:]
    v0_req = np.linalg.solve(Prv, np.asarray(rho_f) - Prr @ rel0[:3])
    vf = Pvr @ rel0[:3] + Pvv @ v0_req
    return v0_req - rel0[3:], np.asarray(rhodot_f) - vf


def cw_propagate(rel0, n: float, times) -> np.ndarray:
    return np.array([cw_stm(n, t) @ rel0 for t in times])


# ---------------------------------------------------------------------------
# Relative trajectories
# ---------------------------------------------------------------------------

@dataclass
class RelativeTrajectory:
    t: np.ndarray  # nondim absolute times (N,)
    rel: np.ndarray  # (N, 6) LVLH relative state, nondim
    system: System = EARTH_MOON
    label: str = ""

    @property
    def rho_km(self):
        return self.system.km(self.rel[:, :3])

    @property
    def rhodot_mps(self):
        return self.system.mps(self.rel[:, 3:])

    @property
    def range_km(self):
        return np.linalg.norm(self.rho_km, axis=1)

    @property
    def t_hours(self):
        return self.system.seconds(self.t) / 3600.0


def relative_trajectory(x_target, x_chaser, t0: float, tf: float,
                        system: System = EARTH_MOON, primary: str = "moon",
                        n_points: int = 400, label: str = ""):
    """Propagate target and chaser (rotating-frame states at t0) to tf.

    Returns (RelativeTrajectory, target Trajectory, chaser Trajectory).
    """
    tt = cr3bp.propagate(x_target, tf, system, t0=t0, n_points=n_points, label="target")
    tc = cr3bp.propagate(x_chaser, tf, system, t0=t0, n_points=n_points, label="chaser")
    rel = np.array([frames.to_lvlh(a, b, system, primary) for a, b in zip(tt.x, tc.x)])
    return RelativeTrajectory(tt.t, rel, system, label), tt, tc


def linear_relative_targeting(x_target, rel0, rho_f, tof: float,
                              system: System = EARTH_MOON, primary: str = "moon",
                              rhodot_f=(0.0, 0.0, 0.0)):
    """Two-impulse targeting using the target's CR3BP STM (linearized relative
    dynamics about the target orbit). All quantities nondim, LVLH in/out.
    Returns (dv1_lvlh, dv2_lvlh)."""
    x_target = np.asarray(x_target, float)
    xc0 = frames.from_lvlh(x_target, rel0, system, primary)
    dx0 = xc0 - x_target
    xtf, phi = cr3bp.propagate_state(x_target, tof, system, with_stm=True)
    Cf = frames.lvlh_matrix(xtf, system, primary)
    C0 = frames.lvlh_matrix(x_target, system, primary)
    dr_f = Cf.T @ np.asarray(rho_f, float)
    dv0_req = np.linalg.solve(phi[:3, 3:], dr_f - phi[:3, :3] @ dx0[:3])
    dv1_rot = dv0_req - dx0[3:]
    dxf = phi @ np.concatenate((dx0[:3], dv0_req))
    xcf = xtf + dxf
    want = frames.from_lvlh(xtf, np.concatenate((rho_f, rhodot_f)), system, primary)
    dv2_rot = want[3:] - xcf[3:]
    return C0 @ dv1_rot, Cf @ dv2_rot


# ---------------------------------------------------------------------------
# Waypoint planner
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    t_hours: float  # arrival time, hours after plan start
    rho_km: tuple  # LVLH position (V-bar, -H-bar, R-bar) [km]
    label: str = ""


@dataclass
class Maneuver:
    t: float  # nondim absolute time
    dv_rot: np.ndarray  # rotating frame, nondim
    dv_lvlh_mps: np.ndarray  # LVLH, m/s
    label: str = ""

    @property
    def mag_mps(self) -> float:
        return float(np.linalg.norm(self.dv_lvlh_mps))


@dataclass
class RpodPlan:
    t0: float
    maneuvers: list[Maneuver]
    rel: RelativeTrajectory
    target: cr3bp.Trajectory
    chaser: cr3bp.Trajectory
    waypoints: list[Waypoint]
    system: System = EARTH_MOON
    primary: str = "moon"
    leg_states: list = field(default_factory=list, repr=False)  # (t, xt, xc_after_burn)

    @property
    def total_dv_mps(self) -> float:
        return sum(m.mag_mps for m in self.maneuvers)

    def summary(self) -> str:
        s = self.system
        lines = [f"RPOD plan: {len(self.maneuvers)} burns, total {self.total_dv_mps:.3f} m/s"]
        for k, m in enumerate(self.maneuvers):
            h = s.seconds(m.t - self.t0) / 3600.0
            d = m.dv_lvlh_mps
            lines.append(f"  burn {k + 1:2d} t+{h:7.2f} h  dv[Vbar,-Hbar,Rbar] = "
                         f"[{d[0]:8.3f} {d[1]:8.3f} {d[2]:8.3f}] m/s  |dv| = {m.mag_mps:.3f}"
                         f"  {m.label}")
        return "\n".join(lines)


def plan_waypoints(x_target0, x_chaser0, waypoints: list[Waypoint],
                   system: System = EARTH_MOON, primary: str = "moon", t0: float = 0.0,
                   final_rhodot_mps=(0.0, 0.0, 0.0), n_points_per_leg: int = 200,
                   tol: float = 1e-12) -> RpodPlan:
    """Plan an impulsive multi-waypoint approach in the target's LVLH frame.

    A burn is placed at the start of each leg (nonlinear CR3BP shooting to the
    next waypoint position, seeded by the linear STM solution), and a final
    burn at the last waypoint sets the relative velocity to final_rhodot_mps.
    x_target0, x_chaser0 : rotating-frame states at t0 (nondim).
    """
    xt = np.asarray(x_target0, float).copy()
    xc = np.asarray(x_chaser0, float).copy()
    t_prev = t0
    mans, legs = [], []
    rel_t, rel_x, tt_t, tt_x, tc_x = [], [], [], [], []
    for wp in waypoints:
        t_arr = t0 + system.nd_time_seconds(wp.t_hours * 3600.0)
        tof = t_arr - t_prev
        if tof <= 0:
            raise ValueError("waypoint times must be increasing")
        rho = system.nd_length(np.asarray(wp.rho_km, float))
        xtf = cr3bp.propagate_state(xt, tof, system)
        r_goal = frames.from_lvlh(xtf, np.concatenate((rho, np.zeros(3))), system, primary)[:3]
        # linear seed from the target STM
        rel0 = frames.to_lvlh(xt, xc, system, primary)
        dv1_l, _ = linear_relative_targeting(xt, rel0, rho, tof, system, primary)
        v_seed = xc[3:] + frames.lvlh_vector_to_rotating(xt, dv1_l, system, primary)
        v_new, _ = shoot(xc, r_goal, tof, system, v_guess=v_seed, tol=tol)
        dv = v_new - xc[3:]
        mans.append(Maneuver(t_prev, dv, system.mps(frames.rotating_vector_to_lvlh(
            xt, dv, system, primary)), label=f"-> {wp.label}" if wp.label else ""))
        xc_burn = xc.copy()
        xc_burn[3:] = v_new
        legs.append((t_prev, xt.copy(), xc_burn.copy()))
        rel, ttr, tcr = relative_trajectory(xt, xc_burn, t_prev, t_arr, system, primary,
                                            n_points_per_leg)
        rel_t.append(rel.t)
        rel_x.append(rel.rel)
        tt_x.append(ttr.x)
        tc_x.append(tcr.x)
        xt, xc = ttr.x[-1].copy(), tcr.x[-1].copy()
        t_prev = t_arr
    # final burn: match the requested relative velocity
    rho_f = frames.to_lvlh(xt, xc, system, primary)[:3]
    want = frames.from_lvlh(xt, np.concatenate((rho_f, system.nd_velocity(
        np.asarray(final_rhodot_mps, float) / 1000.0))), system, primary)
    dv = want[3:] - xc[3:]
    mans.append(Maneuver(t_prev, dv, system.mps(frames.rotating_vector_to_lvlh(
        xt, dv, system, primary)), label="final"))
    legs.append((t_prev, xt.copy(), want.copy()))
    t_all = np.concatenate(rel_t)
    rel = RelativeTrajectory(t_all, np.vstack(rel_x), system, "approach")
    tgt = cr3bp.Trajectory(t_all, np.vstack(tt_x), system, "target")
    chs = cr3bp.Trajectory(t_all, np.vstack(tc_x), system, "chaser")
    return RpodPlan(t0, mans, rel, tgt, chs, list(waypoints), system, primary, legs)


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------

def keep_out_violations(rel: RelativeTrajectory, radius_km: float,
                        exclude_final_km: float | None = None):
    """Indices where the chaser is inside the keep-out sphere. Points with range
    below exclude_final_km (e.g. inside the approach corridor) are ignored."""
    r = rel.range_km
    bad = r < radius_km
    if exclude_final_km is not None:
        bad &= r >= exclude_final_km
    return np.nonzero(bad)[0]


def corridor_violations(rel: RelativeTrajectory, axis=(-1.0, 0.0, 0.0),
                        half_angle_deg: float = 15.0, max_range_km: float = 1.0):
    """Indices within max_range_km that lie outside the approach cone of the
    given half-angle about `axis` (LVLH; default: approach along -V-bar)."""
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    rho = rel.rho_km
    r = np.linalg.norm(rho, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = rho @ axis / r
    inside = r <= max_range_km
    bad = inside & (r > 1e-6) & (cosang < np.cos(np.radians(half_angle_deg)))
    return np.nonzero(bad)[0]


@dataclass
class AbortResult:
    burn_index: int
    min_range_km: float
    t_min_hours: float  # hours after the burn
    rel: RelativeTrajectory


def passive_abort(plan: RpodPlan, duration_hours: float = 24.0,
                  n_points: int = 600) -> list[AbortResult]:
    """For each burn, assume all later burns are missed: free-drift the chaser
    for duration_hours after it and report the closest approach to the target."""
    s = plan.system
    out = []
    dt = s.nd_time_seconds(duration_hours * 3600.0)
    for k, (t, xt, xc) in enumerate(plan.leg_states):
        rel, _, _ = relative_trajectory(xt, xc, t, t + dt, s, plan.primary, n_points)
        i = int(np.argmin(rel.range_km))
        out.append(AbortResult(k, float(rel.range_km[i]),
                               float(s.seconds(rel.t[i] - t) / 3600.0), rel))
    return out
