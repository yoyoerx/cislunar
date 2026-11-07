"""Intercept and rendezvous targeting in the CR3BP.

The core is a "CR3BP Lambert" problem: find the initial velocity at r0 that
reaches position r_f after time tof. Solved by single shooting (Newton on the
initial velocity using the position/velocity block of the STM), seeded from
Earth- or Moon-centred two-body Lambert solutions or from a warm start.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import cr3bp, frames, twobody
from .constants import EARTH_MOON, System
from .lambert import lambert
from .periodic import PeriodicOrbit

StateFn = Callable[[float], np.ndarray]


# ---------------------------------------------------------------------------
# State providers: anything that gives a rotating-frame state at time t
# ---------------------------------------------------------------------------

def as_state_fn(obj, system: System = EARTH_MOON) -> StateFn:
    """Turn a PeriodicOrbit, a state at t = 0 (6,), or a callable into t -> state."""
    if callable(obj) and not isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, PeriodicOrbit):
        return obj.state_at
    x0 = np.asarray(obj, float)
    cache: dict[float, np.ndarray] = {}

    def fn(t):
        if t not in cache:
            cache[t] = cr3bp.propagate_state(x0, t, system)
        return cache[t]

    return fn


def keplerian_state_fn(a_km, e, inc, raan, argp, nu0, system: System = EARTH_MOON,
                       center: str = "earth", t0: float = 0.0) -> StateFn:
    """State provider for a two-body orbit about Earth or Moon (e.g. a LEO
    parking orbit or LLO), expressed in the CR3BP rotating frame.

    Elements are relative to the inertial frame aligned with the rotating
    frame at t = 0 (z = Earth-Moon orbit normal). nu0 is the true anomaly at
    time t0 (nondim). Moon/Earth third-body perturbations are ignored, which is
    appropriate for short-period parking orbits.
    """
    gm = system.gm1 if center == "earth" else system.gm2
    r0, v0 = twobody.coe2rv(a_km, e, inc, raan, argp, nu0, gm)

    def fn(t):
        dt_s = float(system.seconds(t - t0))
        r, v = twobody.kepler_propagate(r0, v0, dt_s, gm) if dt_s != 0 else (r0, v0)
        x_in = np.concatenate((system.nd_length(r), system.nd_velocity(v)))
        return frames.inertial_to_rotating(t, x_in, system, center)

    return fn


# ---------------------------------------------------------------------------
# Single-shooting intercept
# ---------------------------------------------------------------------------

def shoot(x0, r_target, tof, system: System = EARTH_MOON, v_guess=None,
          tol: float = 1e-10, max_iter: int = 25, max_step: float = 0.5):
    """Newton single shooting: find v0 so that position(tof) == r_target.

    Returns (v0, x_final) or raises RuntimeError. Steps are limited to
    max_step (nondim velocity) and backtracked if the miss distance grows.
    """
    r0 = np.asarray(x0[:3], float)
    v = np.asarray(x0[3:6] if v_guess is None else v_guess, float).copy()

    def miss(vv):
        xf, phi = cr3bp.propagate_state(np.concatenate((r0, vv)), tof, system,
                                        with_stm=True, stop_on_impact=True)
        return xf, phi, xf[:3] - r_target

    try:
        xf, phi, err = miss(v)
    except RuntimeError as exc:
        raise RuntimeError(f"shooting: initial propagation failed ({exc})") from exc
    en = np.linalg.norm(err)
    for it in range(max_iter):
        if en < tol:
            return v, xf
        if it >= 8 and en > 1e-3:
            raise RuntimeError(f"shooting: stagnated (miss {en:.3e})")
        dv = -np.linalg.solve(phi[0:3, 3:6], err)
        n = np.linalg.norm(dv)
        if n > max_step:
            dv *= max_step / n
        lam = 1.0
        for _ in range(6):
            try:
                xf_n, phi_n, err_n = miss(v + lam * dv)
                en_n = np.linalg.norm(err_n)
            except RuntimeError:
                en_n = np.inf
            if en_n < en:
                break
            lam *= 0.5
        else:
            raise RuntimeError(f"shooting: line search failed (miss {en:.3e})")
        v = v + lam * dv
        xf, phi, err, en = xf_n, phi_n, err_n, en_n
    if en < tol * 100:
        return v, xf
    raise RuntimeError(f"shooting did not converge (miss {en:.3e})")


def lambert_candidates(x0, r_target, tof, system: System = EARTH_MOON, t0: float = 0.0):
    """Two-body Lambert solutions about Earth and Moon (prograde & retrograde),
    as rotating-frame (name, v_depart, v_arrive) tuples."""
    out = []
    for center, gm_nd in (("earth", 1.0 - system.mu), ("moon", system.mu)):
        xi0 = frames.rotating_to_inertial(t0, np.concatenate((x0[:3], np.zeros(3))),
                                          system, center)
        xif = frames.rotating_to_inertial(t0 + tof, np.concatenate((r_target, np.zeros(3))),
                                          system, center)
        for prograde in (True, False):
            try:
                v1, v2 = lambert(xi0[:3], xif[:3], tof, gm_nd, prograde=prograde)
            except (ValueError, RuntimeError):
                continue
            x1 = frames.inertial_to_rotating(t0, np.concatenate((xi0[:3], v1)), system, center)
            x2 = frames.inertial_to_rotating(t0 + tof, np.concatenate((xif[:3], v2)),
                                             system, center)
            out.append((f"lambert-{center}-{'pro' if prograde else 'retro'}", x1[3:], x2[3:]))
    return out


def lambert_seeds(x0, r_target, tof, system: System = EARTH_MOON, t0: float = 0.0):
    """Candidate initial velocities (rotating frame) from two-body Lambert."""
    return [(n, v1) for n, v1, _ in lambert_candidates(x0, r_target, tof, system, t0)]


def raan_for_target(r_target_inertial, inc: float) -> list[float]:
    """RAAN values (rad) that place a parking orbit of inclination inc in a
    plane containing the direction r_target_inertial (Earth-centred inertial).
    Empty if the inclination is too low to reach the target's declination."""
    u = np.asarray(r_target_inertial, float)
    u = u / np.linalg.norm(u)
    rho = np.hypot(u[0], u[1])
    phi = np.arctan2(u[1], u[0])
    if abs(np.sin(inc)) < 1e-12:
        return [0.0] if abs(u[2]) < 1e-9 else []
    s = -u[2] / (np.tan(inc) * rho)
    if abs(s) > 1:
        return []
    a = np.arcsin(s)
    return [float((phi + a) % (2 * np.pi)), float((phi + np.pi - a) % (2 * np.pi))]


@dataclass
class Transfer:
    t_depart: float  # nondim
    tof: float  # nondim
    x_chaser: np.ndarray  # chaser state before departure burn
    dv1: np.ndarray  # rotating frame, nondim
    x_arrive: np.ndarray  # chaser state at arrival, before arrival burn
    dv2: np.ndarray  # rotating frame, nondim (zero vector if intercept only)
    system: System = EARTH_MOON
    seed: str = ""
    v_rel: np.ndarray | None = None  # chaser minus target velocity at arrival (nondim)
    _traj: cr3bp.Trajectory | None = field(default=None, repr=False)

    @property
    def v_rel_mps(self) -> float:
        """Relative speed at the encounter (before any arrival burn), m/s."""
        return float(self.system.mps(np.linalg.norm(self.v_rel))) if self.v_rel is not None else np.nan

    @property
    def dv1_mps(self) -> float:
        return float(self.system.mps(np.linalg.norm(self.dv1)))

    @property
    def dv2_mps(self) -> float:
        return float(self.system.mps(np.linalg.norm(self.dv2)))

    @property
    def total_mps(self) -> float:
        return self.dv1_mps + self.dv2_mps

    @property
    def tof_days(self) -> float:
        return float(self.system.days(self.tof))

    def trajectory(self, n_points: int = 1000) -> cr3bp.Trajectory:
        if self._traj is None or len(self._traj.t) != n_points:
            x = self.x_chaser.copy()
            x[3:] += self.dv1
            tr = cr3bp.propagate(x, self.t_depart + self.tof, self.system,
                                 t0=self.t_depart, n_points=n_points, label="transfer")
            self._traj = tr
        return self._traj

    def summary(self) -> str:
        s = self.system
        return (f"depart t = {s.days(self.t_depart):.4f} d, TOF = {self.tof_days:.4f} d "
                f"[{self.seed}]\n  dv1 = {self.dv1_mps:.2f} m/s, dv2 = {self.dv2_mps:.2f} m/s, "
                f"total = {self.total_mps:.2f} m/s, relative speed at encounter "
                f"{self.v_rel_mps:.2f} m/s")


def intercept(chaser, target, t_depart: float, tof: float, system: System = EARTH_MOON,
              rendezvous: bool = True, v_guess=None, seeds: str = "all",
              tol: float = 1e-10, aim_offset=None) -> Transfer:
    """Two-impulse intercept (rendezvous=False) or rendezvous of chaser with target.

    chaser, target : PeriodicOrbit, initial state (6,) at t = 0, or callable t -> state.
    t_depart, tof  : nondimensional.
    seeds          : 'all' tries the warm start (v_guess), the chaser's current
                     velocity and all Lambert seeds, returning the minimum-cost
                     converged solution; 'first' returns the first that converges;
                     'warm' tries only v_guess.
    aim_offset     : optional rotating-frame vector (nondim) added to the target's
                     arrival position, e.g. for a flyby at a miss distance.
    """
    cfn, tfn = as_state_fn(chaser, system), as_state_fn(target, system)
    xc = cfn(t_depart)
    xt = tfn(t_depart + tof)
    aim = xt[:3] + (np.zeros(3) if aim_offset is None else np.asarray(aim_offset, float))
    candidates = []
    if v_guess is not None:
        candidates.append(("warm", np.asarray(v_guess, float)))
    if seeds != "warm":
        candidates.append(("coast", xc[3:].copy()))
        candidates += lambert_seeds(xc, aim, tof, system, t_depart)
    best = None
    for name, vg in candidates:
        try:
            v1, xf = shoot(xc, aim, tof, system, v_guess=vg, tol=tol)
        except (RuntimeError, np.linalg.LinAlgError):
            continue
        dv2 = (xt[3:] - xf[3:]) if rendezvous else np.zeros(3)
        tr = Transfer(t_depart, tof, xc, v1 - xc[3:], xf, dv2, system, seed=name,
                      v_rel=xf[3:] - xt[3:])
        if best is None or tr.total_mps < best.total_mps - 1e-9:
            best = tr
        if seeds == "first" or name == "warm":
            break
    if best is None:
        raise RuntimeError("no seed converged for this departure/TOF")
    return best


@dataclass
class Porkchop:
    t_departs: np.ndarray  # nondim (M,)
    tofs: np.ndarray  # nondim (N,)
    dv1_mps: np.ndarray  # (N, M), NaN where not converged
    dv2_mps: np.ndarray
    system: System = EARTH_MOON
    best: Transfer | None = None
    transfers: list | None = None  # [i_tof][j_dep] -> Transfer or None (keep_transfers)
    vrel_mps: np.ndarray | None = None  # (N, M) relative speed at encounter

    @property
    def total_mps(self):
        return self.dv1_mps + self.dv2_mps


def porkchop(chaser, target, t_departs, tofs, system: System = EARTH_MOON,
             rendezvous: bool = True, verbose: bool = False,
             keep_transfers: bool = False, refine: int = 0) -> Porkchop:
    """Grid of transfers over departure time x time of flight (both nondim).

    Each row warm-starts from its neighbour; failed points are NaN.
    keep_transfers stores every converged Transfer in Porkchop.transfers.
    refine: number of extra passes in which every cell is re-solved from each
    neighbour's solution and the cheapest result kept. This repairs cells that
    locked onto an expensive solution branch during the first sweep.
    """
    t_departs = np.asarray(t_departs, float)
    tofs = np.asarray(tofs, float)
    cfn, tfn = as_state_fn(chaser, system), as_state_fn(target, system)
    dv1 = np.full((len(tofs), len(t_departs)), np.nan)
    dv2 = np.full_like(dv1, np.nan)
    vrel = np.full_like(dv1, np.nan)
    best = None
    kept = [[None] * len(t_departs) for _ in tofs] if keep_transfers else None
    prev_col = [None] * len(tofs)
    for j, td in enumerate(t_departs):
        warm = None
        for i, tof in enumerate(tofs):
            guess = warm if warm is not None else prev_col[i]
            try:
                tr = intercept(cfn, tfn, td, tof, system, rendezvous=rendezvous,
                               v_guess=guess, seeds="all" if guess is None else "first")
            except RuntimeError:
                warm = None
                prev_col[i] = None
                continue
            v1 = tr.x_chaser[3:] + tr.dv1
            warm = v1
            prev_col[i] = v1
            dv1[i, j], dv2[i, j], vrel[i, j] = tr.dv1_mps, tr.dv2_mps, tr.v_rel_mps
            if kept is not None:
                kept[i][j] = tr
            if best is None or tr.total_mps < best.total_mps:
                best = tr
        if verbose:
            print(f"  porkchop column {j + 1}/{len(t_departs)}")
    if refine:
        grid = kept if kept is not None else None
        if grid is None:  # need the transfers to re-seed from
            raise ValueError("refine requires keep_transfers=True")
        for _ in range(refine):
            improved = 0
            for i in range(len(tofs)):
                for j in range(len(t_departs)):
                    cur = grid[i][j]
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ni, nj = i + di, j + dj
                        if not (0 <= ni < len(tofs) and 0 <= nj < len(t_departs)):
                            continue
                        nb = grid[ni][nj]
                        if nb is None:
                            continue
                        try:
                            tr = intercept(cfn, tfn, t_departs[j], tofs[i], system, rendezvous,
                                           v_guess=nb.x_chaser[3:] + nb.dv1, seeds="warm")
                        except RuntimeError:
                            continue
                        if cur is None or tr.total_mps < cur.total_mps - 0.5:
                            cur = tr
                            improved += 1
                    grid[i][j] = cur
                    if cur is not None:
                        dv1[i, j], dv2[i, j], vrel[i, j] = cur.dv1_mps, cur.dv2_mps, cur.v_rel_mps
                        if best is None or cur.total_mps < best.total_mps:
                            best = cur
            if verbose:
                print(f"  refine pass: {improved} cells improved")
            if not improved:
                break
    return Porkchop(t_departs, tofs, dv1, dv2, system, best, kept, vrel)


def optimize_transfer(chaser, target, t_bounds, tof_bounds, system: System = EARTH_MOON,
                      rendezvous: bool = True, n_scan=(48, 10), verbose: bool = False) -> Transfer:
    """Find a low-cost transfer over a departure-time window and TOF range.

    1. Cheap scan: two-body Lambert cost (Earth/Moon centred) on an n_scan grid.
    2. CR3BP intercept from the best few scan points.
    3. Nelder-Mead refinement in (t_depart, tof) with warm-started shooting.
    All times nondimensional.
    """
    from scipy.optimize import minimize

    cfn, tfn = as_state_fn(chaser, system), as_state_fn(target, system)
    tds = np.linspace(*t_bounds, n_scan[0])
    tofs = np.linspace(*tof_bounds, n_scan[1])
    scan = []
    for td in tds:
        xc = cfn(td)
        for tof in tofs:
            xt = tfn(td + tof)
            for name, v1, v2 in lambert_candidates(xc, xt[:3], tof, system, td):
                cost = np.linalg.norm(v1 - xc[3:])
                if rendezvous:
                    cost += np.linalg.norm(xt[3:] - v2)
                scan.append((cost, td, tof, v1))
    scan.sort(key=lambda s: s[0])
    best = None
    for cost, td, tof, v1 in scan[:6]:
        try:
            tr = intercept(cfn, tfn, td, tof, system, rendezvous, v_guess=v1, seeds="first")
        except RuntimeError:
            continue
        if verbose:
            print(f"  scan seed: two-body {system.mps(cost):.1f} m/s -> CR3BP {tr.total_mps:.1f} m/s")
        if best is None or tr.total_mps < best.total_mps:
            best = tr
    if best is None:
        raise RuntimeError("optimize_transfer: no scan point converged in the CR3BP")

    state = {"best": best}
    scale = np.array([t_bounds[1] - t_bounds[0], tof_bounds[1] - tof_bounds[0]]) / 10.0

    def cost(p):
        td, tof = best.t_depart + p[0] * scale[0], best.tof + p[1] * scale[1]
        if not (t_bounds[0] <= td <= t_bounds[1] and tof_bounds[0] <= tof <= tof_bounds[1]):
            return 1e6
        cur = state["best"]
        try:
            tr = intercept(cfn, tfn, td, tof, system, rendezvous,
                           v_guess=cur.x_chaser[3:] + cur.dv1, seeds="warm")
        except RuntimeError:
            return 1e6
        if tr.total_mps < state["best"].total_mps:
            state["best"] = tr
        return tr.total_mps

    minimize(cost, np.zeros(2), method="Nelder-Mead",
             options={"xatol": 1e-4, "fatol": 0.01, "maxiter": 40,
                      "initial_simplex": [[0, 0], [0.5, 0], [0, 0.5]]})
    if verbose:
        print(f"  refined: {state['best'].total_mps:.1f} m/s")
    return state["best"]


# ---------------------------------------------------------------------------
# Flybys
# ---------------------------------------------------------------------------

@dataclass
class FlybyResult:
    transfer: Transfer  # departure burn and arrival state at the aim point
    miss_km_requested: float
    miss_km: float  # true closest approach
    t_ca: float  # nondim time of closest approach
    v_rel_mps: float  # relative speed at closest approach
    rel_t: np.ndarray = field(repr=False, default=None)  # times around the encounter
    rel_range_km: np.ndarray = field(repr=False, default=None)

    @property
    def dv1_mps(self) -> float:
        return self.transfer.dv1_mps

    def summary(self) -> str:
        s = self.transfer.system
        return (f"flyby: depart day {s.days(self.transfer.t_depart):.3f}, TOF "
                f"{self.transfer.tof_days:.3f} d, dv1 {self.dv1_mps:.2f} m/s\n"
                f"  closest approach {self.miss_km:.2f} km (requested {self.miss_km_requested:.2f}) "
                f"at relative speed {self.v_rel_mps:.1f} m/s")


def closest_approach(x_chaser, x_target, t: float, window: float, system: System = EARTH_MOON,
                     n: int = 2001):
    """Closest approach between two coasting spacecraft whose states at time t are
    given, searched over [t - window, t + window] (nondim).
    Returns (t_ca, range_km, rel_speed_mps, times, ranges_km)."""
    from scipy.optimize import minimize_scalar

    x_chaser = np.asarray(x_chaser, float)
    x_target = np.asarray(x_target, float)

    def track(x):
        fwd = cr3bp.propagate(x, t + window, system, t0=t, n_points=n // 2 + 1)
        bwd = cr3bp.propagate(x, t - window, system, t0=t, n_points=n // 2 + 1)
        return (np.concatenate((bwd.t[::-1], fwd.t[1:])),
                np.vstack((bwd.x[::-1], fwd.x[1:])))

    tt, xc = track(x_chaser)
    _, xt = track(x_target)
    rng = system.km(np.linalg.norm(xc[:, :3] - xt[:, :3], axis=1))
    k = int(np.argmin(rng))
    lo, hi = tt[max(k - 1, 0)], tt[min(k + 1, len(tt) - 1)]

    def dist(tq):
        a = cr3bp.propagate_state(x_chaser, tq - t, system)
        b = cr3bp.propagate_state(x_target, tq - t, system)
        return float(np.linalg.norm(a[:3] - b[:3]))

    t_ca = float(tt[k])
    if hi > lo:
        r = minimize_scalar(dist, bounds=(lo, hi), method="bounded", options={"xatol": 1e-11})
        t_ca = float(r.x)
    a = cr3bp.propagate_state(x_chaser, t_ca - t, system)
    b = cr3bp.propagate_state(x_target, t_ca - t, system)
    return (t_ca, float(system.km(np.linalg.norm(a[:3] - b[:3]))),
            float(system.mps(np.linalg.norm(a[3:] - b[3:]))), tt, rng)


def flyby(chaser, target, t_depart: float, tof: float, miss_km: float = 0.0,
          system: System = EARTH_MOON, miss_hint=(0.0, 0.0, 1.0), v_guess=None,
          seeds: str = "all", window: float | None = None, max_iter: int = 6) -> FlybyResult:
    """Single-burn flyby of a target at a requested miss distance.

    The aim point is the target's arrival position plus miss_km along the
    component of miss_hint (rotating frame) perpendicular to the encounter
    relative velocity, so the aim point is (to first order) the closest approach.
    Because the relative velocity depends on the solution, the aim is re-solved
    until it stops changing. The true closest approach is then found by a dense
    search around the encounter (window, nondim; default 5 hours).
    """
    cfn, tfn = as_state_fn(chaser, system), as_state_fn(target, system)
    tr = intercept(cfn, tfn, t_depart, tof, system, rendezvous=False, v_guess=v_guess,
                   seeds=seeds)
    hint = np.asarray(miss_hint, float)
    for _ in range(max_iter if miss_km > 0 else 0):
        u = tr.v_rel / np.linalg.norm(tr.v_rel)
        d = hint - np.dot(hint, u) * u
        if np.linalg.norm(d) < 1e-6:  # hint parallel to v_rel: pick any perpendicular
            d = np.cross(u, [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0])
        offset = system.nd_length(miss_km) * d / np.linalg.norm(d)
        new = intercept(cfn, tfn, t_depart, tof, system, rendezvous=False,
                        v_guess=tr.x_chaser[3:] + tr.dv1, seeds="warm", aim_offset=offset)
        change = np.linalg.norm(new.dv1 - tr.dv1)
        tr = new
        if system.mps(change) < 1e-6:
            break
    x_tgt = tfn(t_depart + tof)
    window = window or system.nd_time_seconds(5 * 3600.0)
    t_ca, rng, vrel, tt, rr = closest_approach(tr.x_arrive, x_tgt, t_depart + tof, window, system)
    return FlybyResult(tr, miss_km, rng, t_ca, vrel, tt, rr)
