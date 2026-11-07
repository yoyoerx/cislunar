"""Circular restricted three-body problem in the barycentric rotating frame.

State vector x = [x, y, z, vx, vy, vz], nondimensional. Larger primary at
(-mu, 0, 0), smaller primary at (1 - mu, 0, 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .constants import EARTH_MOON, System

RTOL = 1e-12
ATOL = 1e-12


def _distances(r, mu):
    x, y, z = r[0], r[1], r[2]
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)
    return r1, r2


def acceleration(x, mu):
    """Acceleration in the rotating frame for state x (6,)."""
    px, py, pz, vx, vy, vz = x
    r1, r2 = _distances(x, mu)
    k1 = (1.0 - mu) / r1**3
    k2 = mu / r2**3
    ax = 2.0 * vy + px - k1 * (px + mu) - k2 * (px - 1.0 + mu)
    ay = -2.0 * vx + py - k1 * py - k2 * py
    az = -k1 * pz - k2 * pz
    return np.array([ax, ay, az])


def eom(t, x, mu):
    """CR3BP equations of motion for solve_ivp."""
    return np.concatenate((x[3:6], acceleration(x, mu)))


def hessian_u(r, mu):
    """Second partials of the pseudo-potential U(x, y, z) as a 3x3 matrix."""
    x, y, z = r[0], r[1], r[2]
    r1, r2 = _distances(r, mu)
    m1, m2 = 1.0 - mu, mu
    d1 = np.array([x + mu, y, z])
    d2 = np.array([x - 1.0 + mu, y, z])
    H = -(m1 / r1**3 + m2 / r2**3) * np.eye(3)
    H += 3.0 * m1 * np.outer(d1, d1) / r1**5 + 3.0 * m2 * np.outer(d2, d2) / r2**5
    H[0, 0] += 1.0
    H[1, 1] += 1.0
    return H


_OMEGA = np.array([[0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def jacobian(x, mu):
    """Linearized dynamics matrix A(x) such that d(dx)/dt = A dx."""
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = hessian_u(x[:3], mu)
    A[3:6, 3:6] = _OMEGA
    return A


def eom_stm(t, y, mu):
    """State + STM (6 + 36) equations of motion (hand-expanded for speed)."""
    px, py, pz, vx, vy, vz = y[0], y[1], y[2], y[3], y[4], y[5]
    m1 = 1.0 - mu
    dx1, dx2 = px + mu, px - 1.0 + mu
    r1s = dx1 * dx1 + py * py + pz * pz
    r2s = dx2 * dx2 + py * py + pz * pz
    r1 = np.sqrt(r1s)
    r2 = np.sqrt(r2s)
    k1 = m1 / (r1s * r1)
    k2 = mu / (r2s * r2)
    q1 = 3.0 * k1 / r1s
    q2 = 3.0 * k2 / r2s
    kk = k1 + k2
    uxx = 1.0 - kk + q1 * dx1 * dx1 + q2 * dx2 * dx2
    uyy = 1.0 - kk + (q1 + q2) * py * py
    uzz = -kk + (q1 + q2) * pz * pz
    uxy = (q1 * dx1 + q2 * dx2) * py
    uxz = (q1 * dx1 + q2 * dx2) * pz
    uyz = (q1 + q2) * py * pz
    out = np.empty(42)
    out[0], out[1], out[2] = vx, vy, vz
    out[3] = 2.0 * vy + px - k1 * dx1 - k2 * dx2
    out[4] = -2.0 * vx + py - kk * py
    out[5] = -kk * pz
    phi = y[6:].reshape(6, 6)
    d = out[6:].reshape(6, 6)
    d[0:3] = phi[3:6]
    p0, p1, p2, p3, p4 = phi[0], phi[1], phi[2], phi[3], phi[4]
    d[3] = uxx * p0 + uxy * p1 + uxz * p2 + 2.0 * p4
    d[4] = uxy * p0 + uyy * p1 + uyz * p2 - 2.0 * p3
    d[5] = uxz * p0 + uyz * p1 + uzz * p2
    return out


def pseudo_potential(r, mu):
    r = np.asarray(r)
    r1, r2 = _distances(r, mu)
    return 0.5 * (r[0] ** 2 + r[1] ** 2) + (1.0 - mu) / r1 + mu / r2


def jacobi(x, mu):
    """Jacobi constant C = 2U - v^2. Works on (6,) or (N, 6)."""
    x = np.asarray(x)
    if x.ndim == 2:
        return np.array([jacobi(xi, mu) for xi in x])
    return 2.0 * pseudo_potential(x[:3], mu) - np.dot(x[3:6], x[3:6])


def lagrange_points(mu) -> dict[str, np.ndarray]:
    """Positions of L1..L5 in the rotating frame."""

    def fx(x):
        r1 = abs(x + mu)
        r2 = abs(x - 1.0 + mu)
        return x - (1.0 - mu) * (x + mu) / r1**3 - mu * (x - 1.0 + mu) / r2**3

    eps = 1e-9
    x1 = brentq(fx, -mu + eps, 1.0 - mu - eps)
    x2 = brentq(fx, 1.0 - mu + eps, 2.0)
    x3 = brentq(fx, -2.0, -mu - eps)
    s3 = np.sqrt(3.0) / 2.0
    return {
        "L1": np.array([x1, 0.0, 0.0]),
        "L2": np.array([x2, 0.0, 0.0]),
        "L3": np.array([x3, 0.0, 0.0]),
        "L4": np.array([0.5 - mu, s3, 0.0]),
        "L5": np.array([0.5 - mu, -s3, 0.0]),
    }


@dataclass
class Trajectory:
    """A propagated trajectory in the nondimensional rotating frame."""

    t: np.ndarray  # (N,)
    x: np.ndarray  # (N, 6)
    system: System = EARTH_MOON
    label: str = ""
    stm: np.ndarray | None = field(default=None, repr=False)  # final STM (6, 6)
    frame: str = "rotating"  # or "inertial-<center>"

    @property
    def r(self):
        return self.x[:, :3]

    @property
    def v(self):
        return self.x[:, 3:6]

    @property
    def r_km(self):
        return self.system.km(self.r)

    @property
    def t_days(self):
        return self.system.days(self.t)

    @property
    def final(self):
        return self.x[-1]

    def jacobi(self):
        return jacobi(self.x, self.system.mu)


def propagate(
    x0,
    tf,
    system: System = EARTH_MOON,
    t0: float = 0.0,
    n_points: int = 1000,
    with_stm: bool = False,
    events=None,
    rtol: float = RTOL,
    atol: float = ATOL,
    label: str = "",
) -> Trajectory:
    """Propagate a state from t0 to tf (nondimensional; tf < t0 allowed).

    Returns a Trajectory sampled at n_points times. If with_stm, the final
    state transition matrix is stored in `trajectory.stm`. `events` is passed
    to solve_ivp; if a terminal event fires, the trajectory ends there.
    """
    mu = system.mu
    x0 = np.asarray(x0, dtype=float)
    if with_stm:
        y0 = np.concatenate((x0, np.eye(6).ravel()))
        fun = eom_stm
    else:
        y0 = x0
        fun = eom
    t_eval = np.linspace(t0, tf, n_points) if n_points and n_points > 1 else None
    sol = solve_ivp(
        fun, (t0, tf), y0, method="DOP853", rtol=rtol, atol=atol,
        args=(mu,), t_eval=t_eval, events=events, dense_output=events is not None,
    )
    if not sol.success:
        raise RuntimeError(f"propagation failed: {sol.message}")
    t = sol.t
    ys = sol.y.T
    if events is not None and sol.status == 1:
        # terminal event: append exact event point
        te = np.concatenate([e for e in sol.t_events if len(e)])[-1:]
        ye = sol.sol(te).T
        t = np.concatenate((t, te))
        ys = np.vstack((ys, ye))
    stm = ys[-1, 6:].reshape(6, 6) if with_stm else None
    return Trajectory(t=t, x=ys[:, :6], system=system, label=label, stm=stm)


class ImpactError(RuntimeError):
    """Raised when a trajectory hits a primary."""


def _impact_events(system: System):
    mu = system.mu
    rr1 = system.r1 / system.lstar
    rr2 = system.r2 / system.lstar

    def hit1(t, y, mu):
        return (y[0] + mu) ** 2 + y[1] ** 2 + y[2] ** 2 - rr1**2

    def hit2(t, y, mu):
        return (y[0] - 1.0 + mu) ** 2 + y[1] ** 2 + y[2] ** 2 - rr2**2

    hit1.terminal = hit2.terminal = True
    hit1.direction = hit2.direction = -1
    return [hit1, hit2]


def propagate_state(x0, dt, system: System = EARTH_MOON, with_stm: bool = False,
                    rtol: float = RTOL, atol: float = ATOL, stop_on_impact: bool = False):
    """Propagate and return only the final state (and STM if requested).

    stop_on_impact raises ImpactError if the path enters either primary.
    """
    x0 = np.asarray(x0, dtype=float)
    if dt == 0.0:
        return (x0.copy(), np.eye(6)) if with_stm else x0.copy()
    events = _impact_events(system) if stop_on_impact else None
    if with_stm:
        y0 = np.concatenate((x0, np.eye(6).ravel()))
        fun = eom_stm
    else:
        y0 = x0
        fun = eom
    sol = solve_ivp(fun, (0.0, dt), y0, method="DOP853", rtol=rtol, atol=atol,
                    args=(system.mu,), events=events)
    if sol.status == 1:
        raise ImpactError("trajectory impacts a primary")
    if not sol.success:
        raise RuntimeError(f"propagation failed: {sol.message}")
    yf = sol.y[:, -1]
    if with_stm:
        return yf[:6], yf[6:].reshape(6, 6)
    return yf
