"""Symmetric periodic orbits in the CR3BP.

All families here (planar Lyapunov, halo / NRHO, DRO) are symmetric about the
x-z plane. They are corrected from a state on that plane,
    x0 = [x, 0, z, 0, vy, 0],
by integrating to the next y = 0 crossing (half period) and driving the
crossing velocities vx, vz to zero (perpendicular crossing).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from . import cr3bp
from .constants import EARTH_MOON, SYNODIC_MONTH_DAYS, System

IX, IY, IZ, IVX, IVY, IVZ = range(6)


@dataclass
class PeriodicOrbit:
    state0: np.ndarray  # (6,) nondim, on the x-z plane
    period: float  # nondim
    system: System = EARTH_MOON
    family: str = ""

    @property
    def mu(self):
        return self.system.mu

    @property
    def jacobi(self) -> float:
        return float(cr3bp.jacobi(self.state0, self.mu))

    @property
    def period_days(self) -> float:
        return float(self.system.days(self.period))

    def trajectory(self, n_points: int = 1000, periods: float = 1.0) -> cr3bp.Trajectory:
        return cr3bp.propagate(self.state0, self.period * periods, self.system,
                               n_points=n_points, label=self.family)

    def state_at(self, t: float) -> np.ndarray:
        """State at time t (nondim) along the orbit (t is wrapped mod period).

        Uses a cached DOP853 dense-output interpolant over one period.
        """
        dense = self.__dict__.get("_dense")
        if dense is None or self.__dict__.get("_dense_key") != (tuple(self.state0), self.period):
            sol = solve_ivp(cr3bp.eom, (0.0, self.period), self.state0, method="DOP853",
                            rtol=cr3bp.RTOL, atol=cr3bp.ATOL, args=(self.mu,),
                            dense_output=True)
            dense = sol.sol
            self.__dict__["_dense"] = dense
            self.__dict__["_dense_key"] = (tuple(self.state0), self.period)
        return dense(float(np.mod(t, self.period)))

    def monodromy(self) -> np.ndarray:
        _, phi = cr3bp.propagate_state(self.state0, self.period, self.system, with_stm=True)
        return phi

    def stability_index(self) -> float:
        """nu = (|lambda_max| + 1/|lambda_max|)/2; nu > 1 means unstable."""
        lam = np.abs(np.linalg.eigvals(self.monodromy()))
        lmax = lam.max()
        return 0.5 * (lmax + 1.0 / lmax)

    def perilune_apolune_km(self) -> tuple[float, float]:
        traj = self.trajectory(n_points=4000)
        d = np.linalg.norm(traj.r - self.system.pos2, axis=1)
        return float(self.system.km(d.min())), float(self.system.km(d.max()))

    def summary(self) -> str:
        s = self.system
        rp, ra = self.perilune_apolune_km()
        x = self.state0
        return (
            f"{self.family}: x0={x[0]:.10f} z0={x[2]:.10f} vy0={x[4]:.10f} [nd]\n"
            f"  period = {self.period:.8f} nd = {self.period_days:.4f} days\n"
            f"  Jacobi = {self.jacobi:.8f}, stability index = {self.stability_index():.4f}\n"
            f"  range to {s.name2}: {rp:,.0f} - {ra:,.0f} km"
        )


# ---------------------------------------------------------------------------
# Differential correction
# ---------------------------------------------------------------------------

def _half_period_crossing(x0, mu, t_max=20.0):
    """Integrate x0 (+STM) to the first y = 0 crossing after t = 0."""
    direction = -np.sign(x0[IVY]) if x0[IVY] != 0 else 0

    def ycross(t, y, mu):
        return y[1]

    ycross.terminal = True
    ycross.direction = direction
    y0 = np.concatenate((x0, np.eye(6).ravel()))
    # Skip a tiny initial interval so the start point is not detected.
    sol = solve_ivp(cr3bp.eom_stm, (0.0, t_max), y0, method="DOP853",
                    rtol=cr3bp.RTOL, atol=cr3bp.ATOL, args=(mu,), events=ycross,
                    first_step=1e-6)
    if sol.status != 1 or len(sol.t_events[0]) == 0:
        raise RuntimeError("no y=0 crossing found")
    tc = sol.t_events[0][0]
    yc = sol.y_events[0][0]
    return tc, yc[:6], yc[6:].reshape(6, 6)


def correct_symmetric(
    state0,
    system: System = EARTH_MOON,
    free=(IX, IVY),
    targets=(IVX, IVZ),
    period: float | None = None,
    tol: float = 1e-11,
    max_iter: int = 50,
    verbose: bool = False,
) -> PeriodicOrbit:
    """Differential correction of an x-z-plane-symmetric periodic orbit.

    free    : indices of state0 allowed to vary (subset of x, z, vy).
    targets : crossing-state components forced to zero (vx, and vz for 3D).
    period  : if given, also constrain the full period to this value
              (requires one extra free variable).
    Uses a minimum-norm Newton step when free variables outnumber constraints.
    """
    mu = system.mu
    x0 = np.asarray(state0, dtype=float).copy()
    x0[[IY, IVX, IVZ]] = 0.0
    free = list(free)
    targets = list(targets)
    for it in range(max_iter):
        tc, xc, phi = _half_period_crossing(x0, mu)
        f = cr3bp.eom(tc, xc, mu)
        dt_dx = -phi[IY, free] / f[IY]  # sensitivity of crossing time
        g = [xc[i] for i in targets]
        J = [phi[i, free] + f[i] * dt_dx for i in targets]
        if period is not None:
            g.append(tc - 0.5 * period)
            J.append(dt_dx)
        g = np.array(g)
        J = np.array(J)
        err = np.linalg.norm(g)
        if verbose:
            print(f"  iter {it}: |g| = {err:.3e}")
        if err < tol:
            return PeriodicOrbit(x0, 2.0 * tc, system)
        dx = -np.linalg.lstsq(J, g, rcond=None)[0]
        # damp very large steps
        step = np.linalg.norm(dx)
        if step > 0.05:
            dx *= 0.05 / step
        x0[free] += dx
    raise RuntimeError(f"differential correction did not converge (|g| = {err:.2e})")


# ---------------------------------------------------------------------------
# Initial guesses
# ---------------------------------------------------------------------------

def _gamma(mu, point):
    lp = cr3bp.lagrange_points(mu)
    if point == 1:
        return 1.0 - mu - lp["L1"][0]
    if point == 2:
        return lp["L2"][0] - (1.0 - mu)
    raise ValueError("point must be 1 or 2")


def lyapunov_guess(system: System, point: int, ax_km: float) -> np.ndarray:
    """Linear planar Lyapunov guess about L1/L2 with x-amplitude ax_km.

    Positive ax_km starts on the side of the L-point away from the Earth.
    """
    mu = system.mu
    xl = cr3bp.lagrange_points(mu)[f"L{point}"]
    H = cr3bp.hessian_u(xl, mu)
    uxx, uyy = H[0, 0], H[1, 1]
    b = 4.0 - uxx - uyy
    w2 = 0.5 * (b + np.sqrt(b * b - 4.0 * uxx * uyy))
    A = system.nd_length(ax_km)
    vy = -(w2 + uxx) * A / 2.0
    return np.array([xl[0] + A, 0.0, 0.0, 0.0, vy, 0.0])


def richardson_halo_guess(system: System, point: int, az_km: float,
                          branch: str = "south") -> tuple[np.ndarray, float]:
    """Richardson (1980) third-order halo approximation.

    Returns (state0, period) at the y = 0 crossing at tau1 = 0, which lies on
    the Earth side of L1 / the Moon side of L2.
    branch follows the usual convention: the sign of the maximum out-of-plane
    excursion, i.e. of z at the crossing farther from the Moon ('south': z < 0
    there, like the Gateway NRHO).
    """
    mu = system.mu
    g = _gamma(mu, point)

    def c(n):
        if point == 1:
            return (mu + (-1) ** n * (1 - mu) * g ** (n + 1) / (1 - g) ** (n + 1)) / g**3
        return ((-1) ** n * mu + (-1) ** n * (1 - mu) * g ** (n + 1) / (1 + g) ** (n + 1)) / g**3

    c2, c3, c4 = c(2), c(3), c(4)
    lam = np.sqrt(0.5 * ((2 - c2) + np.sqrt((c2 - 2) ** 2 + 4 * (c2 - 1) * (1 + 2 * c2))))
    k = 2 * lam / (lam**2 + 1 - c2)
    d1 = 3 * lam**2 / k * (k * (6 * lam**2 - 1) - 2 * lam)
    d2 = 8 * lam**2 / k * (k * (11 * lam**2 - 1) - 2 * lam)
    a21 = 3 * c3 * (k**2 - 2) / (4 * (1 + 2 * c2))
    a22 = 3 * c3 / (4 * (1 + 2 * c2))
    a23 = -3 * c3 * lam / (4 * k * d1) * (3 * k**3 * lam - 6 * k * (k - lam) + 4)
    a24 = -3 * c3 * lam / (4 * k * d1) * (2 + 3 * k * lam)
    b21 = -3 * c3 * lam / (2 * d1) * (3 * k * lam - 4)
    b22 = 3 * c3 * lam / d1
    d21 = -c3 / (2 * lam**2)
    a31 = (-9 * lam / (4 * d2) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k**2))
           + (9 * lam**2 + 1 - c2) / (2 * d2) * (3 * c3 * (2 * a23 - k * b21) + c4 * (2 + 3 * k**2)))
    a32 = -1 / d2 * (9 * lam / 4 * (4 * c3 * (k * a24 - b22) + k * c4)
                     + 1.5 * (9 * lam**2 + 1 - c2) * (c3 * (k * b22 + d21 - 2 * a24) - c4))
    b31 = 3 / (8 * d2) * (8 * lam * (3 * c3 * (k * b21 - 2 * a23) - c4 * (2 + 3 * k**2))
                          + (9 * lam**2 + 1 + 2 * c2) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k**2)))
    b32 = 1 / d2 * (9 * lam * (c3 * (k * b22 + d21 - 2 * a24) - c4)
                    + 3 / 8 * (9 * lam**2 + 1 + 2 * c2) * (4 * c3 * (k * a24 - b22) + k * c4))
    d31 = 3 / (64 * lam**2) * (4 * c3 * a24 + c4)
    d32 = 3 / (64 * lam**2) * (4 * c3 * (a23 - d21) + c4 * (4 + k**2))
    den = 2 * lam * (lam * (1 + k**2) - 2 * k)
    s1 = (1.5 * c3 * (2 * a21 * (k**2 - 2) - a23 * (k**2 + 2) - 2 * k * b21)
          - 3 / 8 * c4 * (3 * k**4 - 8 * k**2 + 8)) / den
    s2 = (1.5 * c3 * (2 * a22 * (k**2 - 2) + a24 * (k**2 + 2) + 2 * k * b22 + 5 * d21)
          + 3 / 8 * c4 * (12 - k**2)) / den
    l1 = -1.5 * c3 * (2 * a21 + a23 + 5 * d21) - 3 / 8 * c4 * (12 - k**2) + 2 * lam**2 * s1
    l2 = 1.5 * c3 * (a24 - 2 * a22) + 9 / 8 * c4 + 2 * lam**2 * s2
    delta = lam**2 - c2

    Az = system.nd_length(az_km) / g
    Ax2 = -(l2 * Az**2 + delta) / l1
    if Ax2 < 0:
        raise ValueError("Az below the halo bifurcation amplitude")
    Ax = np.sqrt(Ax2)
    nu = 1 + s1 * Ax**2 + s2 * Az**2
    if branch not in ("north", "south"):
        raise ValueError("branch must be 'north' or 'south'")
    dn = 1.0 if branch == "north" else -1.0
    if point == 2:
        dn = -dn  # tau1 = 0 is the Moon-side crossing for L2
    # tau1 = 0
    x = a21 * Ax**2 + a22 * Az**2 - Ax + (a23 * Ax**2 - a24 * Az**2) + (a31 * Ax**3 - a32 * Ax * Az**2)
    z = dn * Az + dn * d21 * Ax * Az * (1 - 3) + dn * (d32 * Az * Ax**2 - d31 * Az**3)
    vy = lam * nu * (k * Ax + 2 * (b21 * Ax**2 - b22 * Az**2) + 3 * (b31 * Ax**3 - b32 * Ax * Az**2))
    xl = cr3bp.lagrange_points(mu)[f"L{point}"][0]
    state = np.array([xl + g * x, 0.0, g * z, 0.0, g * vy, 0.0])
    period = 2 * np.pi / (lam * nu)
    return state, period


# ---------------------------------------------------------------------------
# Family constructors
# ---------------------------------------------------------------------------

def lyapunov(point: int = 1, ax_km: float = 10000.0, system: System = EARTH_MOON,
             step_km: float = 1000.0) -> PeriodicOrbit:
    """Planar Lyapunov orbit about L1/L2, continued from a small linear orbit.

    ax_km is the x-offset of the initial crossing from the L-point (signed).
    """
    small = np.sign(ax_km) * min(abs(ax_km), 2000.0)
    orb = correct_symmetric(lyapunov_guess(system, point, small), system,
                            free=(IVY,), targets=(IVX,))
    xl = cr3bp.lagrange_points(system.mu)[f"L{point}"][0]
    target_x = xl + system.nd_length(ax_km)
    step = system.nd_length(step_km)
    orb = continue_to(orb, IX, target_x, step, free=(IVY,), targets=(IVX,))
    orb.family = f"L{point} Lyapunov"
    return orb


def halo(point: int = 2, az_km: float = 10000.0, branch: str = "south",
         system: System = EARTH_MOON) -> PeriodicOrbit:
    """Halo orbit about L1/L2 with approximate out-of-plane amplitude az_km.

    Corrected with z0 fixed, varying x0 and vy0.
    """
    state, _ = richardson_halo_guess(system, point, az_km, branch)
    orb = correct_symmetric(state, system, free=(IX, IVY), targets=(IVX, IVZ))
    orb.family = f"L{point} {branch}ern halo"
    return orb


def continue_family(orbit: PeriodicOrbit, param: int, stop, step: float,
                    free=(IX, IVY), targets=(IVX, IVZ), max_orbits: int = 2000,
                    min_step: float = 1e-6):
    """Natural-parameter continuation in state0[param].

    stop : callable(PeriodicOrbit) -> bool, ends continuation when True.
    Returns the list of orbits (including the starting orbit). The step is
    halved on failure and the previous step direction is used for a secant
    predictor of the free variables.
    """
    fam = [orbit]
    free = list(free)
    while len(fam) < max_orbits and not stop(fam[-1]):
        guess = fam[-1].state0.copy()
        if len(fam) >= 2:
            d = fam[-1].state0 - fam[-2].state0
            if abs(d[param]) > 0:
                guess[free] += d[free] * step / d[param]
        guess[param] += step
        try:
            nxt = correct_symmetric(guess, orbit.system, free=free, targets=targets)
            nxt.family = orbit.family
            fam.append(nxt)
        except RuntimeError:
            step *= 0.5
            if abs(step) < min_step:
                break
    return fam


def continue_to(orbit: PeriodicOrbit, param: int, target: float, step: float,
                free=(IX, IVY), targets=(IVX, IVZ), min_step: float = 1e-7,
                return_path: bool = False):
    """Continue an orbit until state0[param] == target (secant predictor,
    step halving on failure). Returns the corrected orbit at the target, or
    every converged orbit along the way if return_path is True."""
    step = abs(step) * np.sign(target - orbit.state0[param])
    prev, cur = None, orbit
    path = [orbit]
    free = list(free)
    while abs(target - cur.state0[param]) > 1e-14:
        h = step if abs(step) < abs(target - cur.state0[param]) else target - cur.state0[param]
        guess = cur.state0.copy()
        if prev is not None:
            d = cur.state0 - prev.state0
            guess[free] += d[free] * h / d[param]
        guess[param] += h
        try:
            nxt = correct_symmetric(guess, orbit.system, free=free, targets=targets)
        except RuntimeError:
            step *= 0.5
            if abs(step) < min_step:
                raise
            continue
        nxt.family = orbit.family
        prev, cur = cur, nxt
        path.append(cur)
        step *= 1.2
    return path if return_path else cur


def _residual(x0, mu, free, targets):
    tc, xc, phi = _half_period_crossing(x0, mu)
    f = cr3bp.eom(tc, xc, mu)
    dt_dx = -phi[IY, free] / f[IY]
    g = np.array([xc[i] for i in targets])
    J = np.array([phi[i, free] + f[i] * dt_dx for i in targets])
    return g, J, tc


def _null_direction(J, ref=None):
    tau = np.linalg.svd(J)[2][-1]
    if ref is not None and np.dot(tau, ref) < 0:
        tau = -tau
    return tau


def continue_arclength(orbit: PeriodicOrbit, ds: float, stop, direction,
                       free=(IX, IZ, IVY), targets=(IVX, IVZ), max_orbits: int = 500,
                       tol: float = 1e-11, ds_max: float | None = None,
                       min_ds: float = 1e-7) -> list[PeriodicOrbit]:
    """Pseudo-arclength continuation of a symmetric family (passes folds).

    direction : vector in the space of `free` variables giving the initial
                sense of travel (e.g. (0, -1, 0) for decreasing z0).
    stop      : callable(PeriodicOrbit) -> bool.
    """
    mu = orbit.system.mu
    free = list(free)
    ds_max = ds_max or 4 * ds
    _, J, _ = _residual(orbit.state0, mu, free, targets)
    tau = _null_direction(J, np.asarray(direction, float))
    fam = [orbit]
    while len(fam) < max_orbits and not stop(fam[-1]):
        X0 = fam[-1].state0[free]
        X = X0 + ds * tau
        ok = False
        for _ in range(15):
            x0 = fam[-1].state0.copy()
            x0[free] = X
            try:
                g, J, tc = _residual(x0, mu, free, targets)
            except RuntimeError:
                break
            G = np.append(g, np.dot(X - X0, tau) - ds)
            if np.linalg.norm(G) < tol:
                ok = True
                break
            X = X - np.linalg.solve(np.vstack((J, tau)), G)
        if not ok:
            ds *= 0.5
            if ds < min_ds:
                break
            continue
        nxt = PeriodicOrbit(x0, 2.0 * tc, orbit.system, orbit.family)
        tau = _null_direction(J, tau)
        fam.append(nxt)
        ds = min(ds * 1.3, ds_max)
    return fam


def nrho(resonance: tuple[int, int] = (9, 2), branch: str = "south",
         point: int = 2, system: System = EARTH_MOON) -> PeriodicOrbit:
    """Near Rectilinear Halo Orbit with period = (q / p) synodic months.

    The default (9, 2) is the Gateway reference orbit: 9 revolutions per
    2 synodic months (~6.56 days), southern L2 family.
    """
    p, q = resonance
    target = system.nd_time_days(SYNODIC_MONTH_DAYS * q / p)
    # Seed near the 9:2 L2 NRHO; for other resonances continue in z.
    sgn = -1.0 if branch == "south" else 1.0
    seed = np.array([1.0219, 0.0, sgn * 0.1821, 0.0, -0.1031, 0.0])
    if point == 1:
        raise NotImplementedError("L1 NRHO seed not provided; use halo()+continue_family")
    orb = correct_symmetric(seed, system, free=(IX, IVY), targets=(IVX, IVZ))
    # walk z0 toward the target period, then solve with the period constraint
    for _ in range(200):
        err = orb.period - target
        if abs(err) < 0.02:
            break
        # period decreases as |z0| grows ... estimate local slope numerically
        s = orb.state0.copy()
        dz = 0.002 * sgn
        s[IZ] += dz
        o2 = correct_symmetric(s, system, free=(IX, IVY), targets=(IVX, IVZ))
        slope = (o2.period - orb.period) / dz
        step = np.clip(-err / slope, -0.01, 0.01)
        s = orb.state0.copy()
        s[IZ] += step
        orb = correct_symmetric(s, system, free=(IX, IVY), targets=(IVX, IVZ))
    orb = correct_symmetric(orb.state0, system, free=(IX, IZ, IVY),
                            targets=(IVX, IVZ), period=target)
    orb.family = f"L2 {branch}ern {p}:{q} NRHO"
    return orb


def dro(x_offset_km: float = 70000.0, system: System = EARTH_MOON,
        step_km: float = 3000.0) -> PeriodicOrbit:
    """Distant Retrograde Orbit about the Moon.

    x_offset_km: distance from the Moon of the crossing on the Earth-facing side.
    Continued outward from a small (near-Keplerian) retrograde orbit.
    """
    mu = system.mu
    d0 = min(x_offset_km, 15000.0)

    def guess(d_km):
        d = system.nd_length(d_km)
        vy = np.sqrt(mu / d) + d  # retrograde circular speed in rotating frame
        return np.array([1.0 - mu - d, 0.0, 0.0, 0.0, vy, 0.0])

    orb = correct_symmetric(guess(d0), system, free=(IVY,), targets=(IVX,))
    target_x = 1.0 - mu - system.nd_length(x_offset_km)
    orb = continue_to(orb, IX, target_x, system.nd_length(step_km),
                      free=(IVY,), targets=(IVX,))
    orb.family = "DRO"
    return orb


# ---------------------------------------------------------------------------
# Multi-crossing (fixed-time) correction and Earth-Moon resonant orbits
# ---------------------------------------------------------------------------

def correct_symmetric_period(state0, period_guess: float, system: System = EARTH_MOON,
                             free=(IVY,), tol: float = 1e-11, max_iter: int = 50,
                             max_step: float = 0.05, verbose: bool = False) -> PeriodicOrbit:
    """Differential correction of an x-z-plane-symmetric orbit at a fixed half period.

    Unlike correct_symmetric (which stops at the first y = 0 crossing), this
    integrates for exactly T/2 and drives y, vx (and vz for 3D states) to zero
    there, with T itself a free variable. Use it for orbits that cross the
    x-axis several times per period, such as Earth-Moon resonant orbits.

    free : indices of state0 allowed to vary (the half period always varies).
    """
    mu = system.mu
    x0 = np.asarray(state0, dtype=float).copy()
    x0[[IY, IVX, IVZ]] = 0.0
    planar = abs(x0[IZ]) < 1e-14 and abs(x0[IVZ]) < 1e-14
    targets = [IY, IVX] if planar else [IY, IVX, IVZ]
    free = list(free)
    half = 0.5 * float(period_guess)
    err = np.inf
    for it in range(max_iter):
        xf, phi = cr3bp.propagate_state(x0, half, system, with_stm=True)
        f = cr3bp.eom(half, xf, mu)
        g = xf[targets]
        err = np.linalg.norm(g)
        if verbose:
            print(f"  iter {it}: |g| = {err:.3e}")
        if err < tol:
            return PeriodicOrbit(x0, 2.0 * half, system)
        J = np.hstack((phi[np.ix_(targets, free)], f[targets][:, None]))
        dx = -np.linalg.lstsq(J, g, rcond=None)[0]
        step = np.linalg.norm(dx)
        if step > max_step:
            dx *= max_step / step
        x0[free] += dx[:-1]
        half += dx[-1]
    raise RuntimeError(f"fixed-time correction did not converge (|g| = {err:.2e})")


def resonant_guess(p: int, q: int, e: float, system: System = EARTH_MOON):
    """Keplerian initial guess for a planar prograde p:q Earth resonant orbit.

    The spacecraft makes p revolutions about the Earth while the Moon makes q.
    Starts at perigee on the far side of the Earth from the Moon (-x), where the
    inertial velocity is perpendicular to the x-axis. Returns (state0, period).
    """
    mu = system.mu
    gm = 1.0 - mu
    a = gm ** (1.0 / 3.0) * (q / p) ** (2.0 / 3.0)
    rp = a * (1.0 - e)
    vp = np.sqrt(gm * (1.0 + e) / rp)
    # inertial velocity (0, -vp) at (-rp, 0) relative to Earth; v_rot = v_in - w x r
    state = np.array([-mu - rp, 0.0, 0.0, 0.0, -vp + rp, 0.0])
    return state, 2.0 * np.pi * q


def resonant(p: int, q: int, e: float | None = None,
             system: System = EARTH_MOON, homotopy_steps: int = 40) -> PeriodicOrbit:
    """Planar p:q Earth-Moon resonant periodic orbit ("rosette").

    p : spacecraft revolutions about the Earth per q lunar revolutions.
        p > q gives an interior orbit (inside the Moon's orbit), p < q exterior.
    e : eccentricity of the Keplerian guess (default 0.5 interior, 0.2 exterior).
    Corrected in the CR3BP with the perigee distance fixed and the perigee
    velocity and period free. In the rotating frame an interior orbit traces a
    rosette with p lobes that closes after one period (about q sidereal months).

    If correcting the Keplerian guess directly fails (the Moon perturbs long arcs
    strongly), a mass homotopy is used: the Moon's mass is raised from ~0, where
    the guess is exactly periodic, to its true value, re-correcting at each step.
    The perigee distance from the Earth is held fixed throughout.
    """
    if p < 1 or q < 1:
        raise ValueError("p and q must be positive integers")
    if e is None:
        e = 0.5 if p > q else 0.2
    state, period = resonant_guess(p, q, e, system)

    def on_family(o):  # reject convergence onto some other periodic orbit
        return abs(o.period / period - 1.0) < 0.1

    try:
        orb = correct_symmetric_period(state, period, system, free=(IVY,))
        if not on_family(orb):
            raise RuntimeError("converged to a different periodic orbit")
    except RuntimeError:
        try:
            orb = _resonant_homotopy(p, q, e, system, homotopy_steps)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}; this p:q / eccentricity combination may not exist "
                               f"in the Earth-Moon CR3BP - try a different e") from None
        if not on_family(orb):
            raise RuntimeError(f"{p}:{q} resonant: correction left the resonance (period "
                               f"{orb.period / (2 * np.pi):.3f} months); try a different e")
    orb.family = f"{p}:{q} resonant"
    return orb


def _resonant_homotopy(p, q, e, system: System, n_steps: int) -> PeriodicOrbit:
    """Continue a p:q resonant orbit in the Moon's mass from ~0 to its true value."""
    from dataclasses import replace

    def sys_at(frac):
        return replace(system, gm2=system.gm2 * frac)

    fracs = list(np.geomspace(1e-4, 1.0, n_steps))
    frac = fracs.pop(0)
    sub = sys_at(frac)
    state, period = resonant_guess(p, q, e, sub)
    orb = correct_symmetric_period(state, period, sub, free=(IVY,))
    while fracs:
        nxt = fracs[0]
        sub_n = sys_at(nxt)
        guess = orb.state0.copy()
        guess[IX] += sub.mu - sub_n.mu  # keep the perigee distance from the Earth fixed
        try:
            orb_n = correct_symmetric_period(guess, orb.period, sub_n, free=(IVY,))
        except RuntimeError:
            mid = np.sqrt(frac * nxt)  # refine the step geometrically
            if nxt / frac < 1.001:
                raise RuntimeError(f"{p}:{q} resonant: homotopy stalled at "
                                   f"{frac:.4f} of the Moon's mass") from None
            fracs.insert(0, mid)
            continue
        fracs.pop(0)
        orb, sub, frac = orb_n, sub_n, nxt
    return PeriodicOrbit(orb.state0, orb.period, system)
