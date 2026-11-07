"""Transient dynamics: invariant manifolds, fates, lifetimes and FTLE.

An unstable periodic orbit has a stable and an unstable manifold: tubes of
trajectories that approach it (stable) or leave it (unstable) asymptotically.
They are the natural "free" routes in and out of libration orbits and through
the L1/L2 gates.

A *fate* classifies what a trajectory starting near the Moon does first:
    impact_moon / impact_earth, exit_L1 (passes the L1 gate heading Earthward),
    exit_L2 (passes the L2 gate heading outward), or captured (none of these
    within the horizon). The time of that event is the trajectory's lifetime.
    "Passing a gate" means crossing a plane GATE_MARGIN_KM beyond the Lagrange
    point, so orbits that merely oscillate around L1/L2 are not counted as exits.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from . import cr3bp
from .constants import EARTH_MOON, System
from .periodic import PeriodicOrbit

FATES = ["captured", "exit_L1", "exit_L2", "impact_moon", "impact_earth"]
FATE_LABELS = {
    "captured": "still near the Moon at the horizon",
    "exit_L1": "leaves through the L1 gate (toward Earth)",
    "exit_L2": "leaves through the L2 gate (outward)",
    "impact_moon": "hits the Moon",
    "impact_earth": "hits the Earth",
}
STABLE_TOL = 1e-3  # |lambda| - 1 below this counts as stable (numerical noise)
FORBIDDEN = -1  # fate-map code for initial conditions inside the forbidden region
GATE_MARGIN_KM = 40000.0  # exit planes sit this far beyond L1 (Earthward) and L2 (outward)


# ---------------------------------------------------------------------------
# Eigen-directions and manifold tubes
# ---------------------------------------------------------------------------

def eigen_directions(orbit: PeriodicOrbit):
    """Return ((lambda_u, v_u), (lambda_s, v_s)) of the monodromy matrix at state0.

    v_u (v_s) is the real eigenvector of the largest (smallest) |eigenvalue|,
    scaled so its position part has unit length. For a stable orbit
    lambda_u is ~1 and the manifolds do not exist.
    """
    w, V = np.linalg.eig(orbit.monodromy())
    iu, i_s = int(np.argmax(np.abs(w))), int(np.argmin(np.abs(w)))
    out = []
    for i in (iu, i_s):
        v = np.real(V[:, i])
        v = v / np.linalg.norm(v[:3])
        out.append((float(np.abs(w[i])), v))
    return tuple(out)


def efolding_time(orbit: PeriodicOrbit) -> float:
    """Time (nondim) for a small error to grow by a factor e; inf if stable."""
    lam = eigen_directions(orbit)[0][0]
    return np.inf if lam <= 1.0 + STABLE_TOL else orbit.period / np.log(lam)


@dataclass
class Fate:
    kind: str  # one of FATES
    t: float  # nondim time of the event (or the horizon if captured); negative if backward

    @property
    def label(self) -> str:
        return FATE_LABELS[self.kind]


@dataclass
class ManifoldTube:
    orbit: PeriodicOrbit
    kind: str  # "unstable" | "stable"
    branch: str  # "interior" | "exterior"
    trajectories: list = field(default_factory=list)  # list[cr3bp.Trajectory]
    fates: list = field(default_factory=list)  # list[Fate]
    eps_km: float = 0.0

    def fate_counts(self) -> dict:
        out = {k: 0 for k in FATES}
        for f in self.fates:
            out[f.kind] += 1
        return out


def _gate_events(system: System, margin_km: float = GATE_MARGIN_KM):
    mu = system.mu
    lp = cr3bp.lagrange_points(mu)
    m = system.nd_length(margin_km)
    xl1, xl2 = lp["L1"][0] - m, lp["L2"][0] + m
    rr1, rr2 = system.r1 / system.lstar, system.r2 / system.lstar

    def impact_moon(t, y, mu):
        return (y[0] - 1 + mu) ** 2 + y[1] ** 2 + y[2] ** 2 - rr2 ** 2

    def impact_earth(t, y, mu):
        return (y[0] + mu) ** 2 + y[1] ** 2 + y[2] ** 2 - rr1 ** 2

    def exit_l1(t, y, mu):
        return y[0] - xl1

    def exit_l2(t, y, mu):
        return y[0] - xl2

    for ev in (impact_moon, impact_earth, exit_l1, exit_l2):
        ev.terminal = True
    impact_moon.direction = impact_earth.direction = -1
    exit_l1.direction = -1  # moving Earthward past the L1 exit plane
    exit_l2.direction = +1  # moving outward past the L2 exit plane
    return [impact_moon, impact_earth, exit_l1, exit_l2], ["impact_moon", "impact_earth",
                                                           "exit_L1", "exit_L2"]


def fate(x0, horizon: float, system: System = EARTH_MOON, gates: bool = True,
         rtol: float = 1e-10, atol: float = 1e-12, n_points: int = 0,
         gate_margin_km: float = GATE_MARGIN_KM):
    """Classify a trajectory: first impact / gate exit within |horizon| (nondim).

    horizon < 0 integrates backward: then "exit_L1" means the trajectory *arrived*
    through the L1 gate from the Earth side (and "impact" that it came from the
    surface). With gates=False only impacts stop the integration.
    Returns (Fate, Trajectory or None if n_points == 0).
    """
    events, names = _gate_events(system, gate_margin_km)
    if not gates:
        events, names = events[:2], names[:2]
    t_eval = np.linspace(0.0, horizon, n_points) if n_points else None
    sol = solve_ivp(cr3bp.eom, (0.0, horizon), np.asarray(x0, float), method="DOP853",
                    rtol=rtol, atol=atol, args=(system.mu,), events=events,
                    t_eval=t_eval, dense_output=bool(n_points))
    kind, t_end = "captured", horizon
    if sol.status == 1:
        for name, te in zip(names, sol.t_events):
            if len(te):
                kind, t_end = name, float(te[0])
                break
    traj = None
    if n_points:
        t = sol.t
        x = sol.y.T
        if sol.status == 1:
            t = np.append(t, t_end)
            x = np.vstack((x, sol.sol(t_end)))
        traj = cr3bp.Trajectory(t=t, x=x, system=system)
    return Fate(kind, t_end), traj


def classify_path(traj, system: System = EARTH_MOON, gate_margin_km: float = GATE_MARGIN_KM,
                  impact: Fate | None = None) -> Fate:
    """Fate of an already-propagated trajectory, from the first exit-plane crossing
    in its samples (forward or backward in time). `impact` (from the integrator)
    wins if it happened before any crossing."""
    lp = cr3bp.lagrange_points(system.mu)
    m = system.nd_length(gate_margin_km)
    x1, x2 = lp["L1"][0] - m, lp["L2"][0] + m
    x = traj.x[:, 0]
    k1 = np.nonzero((x[:-1] >= x1) & (x[1:] < x1))[0]
    k2 = np.nonzero((x[:-1] <= x2) & (x[1:] > x2))[0]
    cands = []
    if len(k1):
        cands.append((abs(traj.t[k1[0] + 1]), Fate("exit_L1", float(traj.t[k1[0] + 1]))))
    if len(k2):
        cands.append((abs(traj.t[k2[0] + 1]), Fate("exit_L2", float(traj.t[k2[0] + 1]))))
    if impact is not None and impact.kind.startswith("impact"):
        cands.append((abs(impact.t), impact))
    if not cands:
        return Fate("captured", float(traj.t[-1]))
    return min(cands, key=lambda c: c[0])[1]


def manifold(orbit: PeriodicOrbit, kind: str = "unstable", branch: str = "interior",
             n: int = 30, eps_km: float = 50.0, tf_days: float = 40.0, n_points: int = 300,
             gates: bool = False) -> ManifoldTube:
    """Generate a stable or unstable manifold tube of a periodic orbit.

    n points are spaced evenly in time around the orbit; at each the eigenvector
    is transported with the STM and the state is displaced by eps_km along it,
    toward Earth (branch="interior") or away from it ("exterior"). Unstable tubes
    are integrated forward, stable tubes backward, for tf_days, stopping at impact
    (and also at the first L1/L2 exit-plane crossing if gates=True). Fates are
    classified either way (from the recorded path when gates=False).
    """
    if kind not in ("unstable", "stable"):
        raise ValueError("kind must be 'unstable' or 'stable'")
    if branch not in ("interior", "exterior"):
        raise ValueError("branch must be 'interior' or 'exterior'")
    system = orbit.system
    (lam_u, vu), (lam_s, vs) = eigen_directions(orbit)
    if lam_u <= 1.0 + STABLE_TOL:
        raise ValueError(f"{orbit.family or 'orbit'} is stable (lambda_max={lam_u:.6f}); "
                         "it has no stable/unstable manifolds")
    v0 = vu if kind == "unstable" else vs
    eps = system.nd_length(eps_km)
    tf = system.nd_time_days(tf_days) * (1.0 if kind == "unstable" else -1.0)
    tube = ManifoldTube(orbit, kind, branch, eps_km=eps_km)
    for tk in np.linspace(0.0, orbit.period, n, endpoint=False):
        if tk == 0.0:
            xk, phi = orbit.state0.copy(), np.eye(6)
        else:
            xk, phi = cr3bp.propagate_state(orbit.state0, tk, system, with_stm=True)
        v = phi @ v0
        v = v / np.linalg.norm(v[:3])
        sgn = -1.0 if v[0] > 0 else 1.0  # interior: first step toward Earth (-x)
        if branch == "exterior":
            sgn = -sgn
        x0 = xk + sgn * eps * v
        f, traj = fate(x0, tf, system, gates=gates, n_points=n_points)
        if not gates:
            f = classify_path(traj, system, impact=f)
        traj.label = f"{kind} {branch}"
        tube.trajectories.append(traj)
        tube.fates.append(f)
    return tube


# ---------------------------------------------------------------------------
# FTLE and fate maps
# ---------------------------------------------------------------------------

def ftle(x0, horizon: float, system: System = EARTH_MOON, stop_at_events: bool = True) -> float:
    """Finite-time Lyapunov exponent ln(sigma_max(Phi)) / T, in 1/nondim-time.

    If stop_at_events, T is cut at the first impact/gate event (so trajectories
    that leave quickly report their stretching up to leaving).
    """
    t_end = horizon
    if stop_at_events:
        f, _ = fate(x0, horizon, system)
        t_end = f.t if abs(f.t) > 1e-6 else horizon
    _, phi = cr3bp.propagate_state(x0, t_end, system, with_stm=True)
    s = np.linalg.svd(phi, compute_uv=False)[0]
    return float(np.log(s) / abs(t_end))


def planar_state(d_km: float, jacobi_c: float, sense: str = "retrograde",
                 system: System = EARTH_MOON):
    """State on the x-axis at signed distance d_km from the Moon (negative = Earth
    side), moving perpendicular to the axis with the speed set by the Jacobi
    constant, prograde or retrograde about the Moon. None if forbidden."""
    mu = system.mu
    x = 1.0 - mu + system.nd_length(d_km)
    r = np.array([x, 0.0, 0.0])
    v2 = 2.0 * cr3bp.pseudo_potential(r, mu) - jacobi_c
    if v2 < 0:
        return None
    vy = np.sqrt(v2) * np.sign(d_km)
    if sense == "retrograde":
        vy = -vy
    return np.array([x, 0.0, 0.0, 0.0, vy, 0.0])


def _fate_cell(args):
    d_km, c, sense, horizon, with_ftle, ftle_horizon, system = args
    x0 = planar_state(d_km, c, sense, system)
    if x0 is None:
        return FORBIDDEN, np.nan, np.nan
    f, _ = fate(x0, horizon, system, rtol=1e-9, atol=1e-11)
    fl = ftle(x0, ftle_horizon, system) if with_ftle else np.nan
    return FATES.index(f.kind), f.t, fl


@dataclass
class FateMap:
    d_km: np.ndarray  # (M,) signed distance from the Moon along x
    jacobi: np.ndarray  # (N,)
    code: np.ndarray  # (N, M) index into FATES, or FORBIDDEN
    lifetime: np.ndarray  # (N, M) nondim
    ftle: np.ndarray  # (N, M) 1/nondim (NaN if not computed)
    sense: str
    horizon: float
    system: System = EARTH_MOON

    @property
    def lifetime_days(self):
        return self.system.days(self.lifetime)

    def fraction(self, kind: str) -> float:
        ok = self.code >= 0
        return float(np.mean(self.code[ok] == FATES.index(kind))) if ok.any() else np.nan


def fate_map(d_km, jacobi_c, sense: str = "retrograde", horizon_days: float = 60.0,
             system: System = EARTH_MOON, with_ftle: bool = False,
             ftle_horizon_days: float = 10.0, workers: int | None = None) -> FateMap:
    """Fate / lifetime (and optionally FTLE) over a grid of planar initial states.

    Each cell starts on the x-axis at distance d from the Moon, moving
    perpendicular to it (prograde or retrograde) with the speed set by the Jacobi
    constant. Cells are run in parallel processes (workers=1 runs serially).
    On Windows, call from under `if __name__ == "__main__":`.
    """
    d_km = np.asarray(d_km, float)
    jacobi_c = np.asarray(jacobi_c, float)
    horizon = system.nd_time_days(horizon_days)
    fh = system.nd_time_days(ftle_horizon_days)
    tasks = [(d, c, sense, horizon, with_ftle, fh, system) for c in jacobi_c for d in d_km]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1:
        res = [_fate_cell(a) for a in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_fate_cell, tasks, chunksize=max(1, len(tasks) // (8 * workers))))
    shape = (len(jacobi_c), len(d_km))
    code = np.array([r[0] for r in res], int).reshape(shape)
    life = np.array([r[1] for r in res], float).reshape(shape)
    fl = np.array([r[2] for r in res], float).reshape(shape)
    return FateMap(d_km, jacobi_c, code, life, fl, sense, horizon, system)
