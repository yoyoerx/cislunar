"""Statistical Delta-v: trajectory correction maneuvers (TCMs) and station-keeping.

Error growth along a trajectory is captured by its state transition matrix
(STM). A TCM at time tc that re-targets the nominal arrival position costs,
to first order,

    dv_tcm = -Phi_rv(tf, tc)^-1 Phi_rr(tf, tc) dr(tc) - dv(tc),

which is linear in the error at injection. So statistics (mean, 95th / 99th
percentile) follow from sampling a linear map: cheap, and exact in scaling
(double the injection error, double the TCM). `monte_carlo_tcm` checks this
with nonlinear truth propagation plus navigation and execution errors.

Station-keeping keeps a spacecraft near an unstable reference orbit: at each
tracking pass, estimate the deviation and cancel only its component along the
orbit's unstable Floquet mode with the smallest possible burn. The CR3BP has no
unmodelled forces, so navigation and execution errors are the only noise:
absolute values are lower bounds; trends versus interval and orbit are robust.

Units: covariances and states are nondimensional; inputs ending _km / _mps
are physical.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import cr3bp
from .constants import EARTH_MOON, System
from .periodic import PeriodicOrbit


# ---------------------------------------------------------------------------
# Error models
# ---------------------------------------------------------------------------

def injection_covariance(sigma_v_mps: float, sigma_r_km: float = 0.0,
                         system: System = EARTH_MOON) -> np.ndarray:
    """Isotropic 6x6 covariance (nondim): 1-sigma per axis in position and velocity."""
    P = np.zeros((6, 6))
    P[:3, :3] = np.eye(3) * system.nd_length(sigma_r_km) ** 2
    P[3:, 3:] = np.eye(3) * system.nd_velocity(sigma_v_mps / 1000.0) ** 2
    return P


def burn_execution_covariance(dv, mag_frac: float = 0.01, pointing_deg: float = 0.5,
                              system: System = EARTH_MOON) -> np.ndarray:
    """6x6 covariance (nondim) of a burn's execution error.

    dv: the burn vector (nondim). mag_frac: 1-sigma magnitude error as a fraction;
    pointing_deg: 1-sigma pointing error per perpendicular axis.
    """
    dv = np.asarray(dv, float)
    m = np.linalg.norm(dv)
    u = dv / m
    s_par = mag_frac * m
    s_perp = m * np.tan(np.radians(pointing_deg))
    P = np.zeros((6, 6))
    P[3:, 3:] = s_par ** 2 * np.outer(u, u) + s_perp ** 2 * (np.eye(3) - np.outer(u, u))
    return P


def _stms(x0, times, system):
    """Nominal states and STMs Phi(t, 0) at increasing times (nondim)."""
    states, phis = [], []
    x, phi, t_prev = np.asarray(x0, float).copy(), np.eye(6), 0.0
    for t in times:
        if t > t_prev:
            x, dphi = cr3bp.propagate_state(x, t - t_prev, system, with_stm=True)
            phi = dphi @ phi
        states.append(x.copy())
        phis.append(phi.copy())
        t_prev = t
    return states, phis


def _guidance(phi_fc):
    """Linear TCM law matrix G (3x6): dv_tcm = G @ dx(tc)."""
    A = np.linalg.solve(phi_fc[:3, 3:], phi_fc[:3, :3])
    return np.hstack((-A, -np.eye(3)))


def _pct(a, q):
    return float(np.percentile(a, q))


# ---------------------------------------------------------------------------
# Linear TCM analysis
# ---------------------------------------------------------------------------

@dataclass
class TcmResult:
    t_tcm: np.ndarray  # nondim, (K,)
    tcm_mean: np.ndarray  # m/s, (K,)
    tcm_p50: np.ndarray
    tcm_p95: np.ndarray
    tcm_p99: np.ndarray
    total_p99: np.ndarray  # |TCM| + change in arrival burn, 99th percentile, m/s
    total_mean: np.ndarray
    miss_p99_km: float  # final position error with no TCM at all
    system: System = EARTH_MOON

    @property
    def t_tcm_days(self):
        return self.system.days(self.t_tcm)

    def table(self) -> str:
        lines = ["  TCM day   mean [m/s]   p95 [m/s]   p99 [m/s]   p99 incl. arrival [m/s]"]
        for k, t in enumerate(self.t_tcm_days):
            lines.append(f"  {t:7.2f} {self.tcm_mean[k]:12.2f} {self.tcm_p95[k]:11.2f} "
                         f"{self.tcm_p99[k]:11.2f} {self.total_p99[k]:14.2f}")
        lines.append(f"  (no TCM: 99% of arrivals miss by up to {self.miss_p99_km:,.0f} km)")
        return "\n".join(lines)


def linear_tcm(x_post, tof: float, P0, t_tcm, system: System = EARTH_MOON,
               dv_arrival=None, n_samples: int = 20000, seed: int = 0) -> TcmResult:
    """Statistics of a single TCM at each candidate time, for injection covariance P0.

    x_post     : nominal state just after the departure burn (nondim).
    tof        : time of flight to the aim point (nondim).
    t_tcm      : candidate TCM times after departure (nondim, each < tof).
    dv_arrival : nominal arrival burn (nondim vector), to include how the TCM changes
                 it; None for a flyby (no arrival burn).
    """
    t_tcm = np.sort(np.atleast_1d(np.asarray(t_tcm, float)))
    if np.any(t_tcm >= tof) or np.any(t_tcm <= 0):
        raise ValueError("TCM times must lie strictly between departure and arrival")
    _, phis = _stms(x_post, list(t_tcm) + [tof], system)
    phi_f = phis[-1]
    rng = np.random.default_rng(seed)
    dx0 = rng.multivariate_normal(np.zeros(6), P0, size=n_samples)  # (n, 6)
    miss = system.km(np.linalg.norm(dx0 @ phi_f[:3, :].T, axis=1))
    K = len(t_tcm)
    out = {k: np.zeros(K) for k in ("mean", "p50", "p95", "p99", "tot99", "totmean")}
    for k in range(K):
        phi_c = phis[k]
        phi_fc = phi_f @ np.linalg.inv(phi_c)
        dxc = dx0 @ phi_c.T
        tcm = dxc @ _guidance(phi_fc).T  # (n, 3)
        mag = system.mps(np.linalg.norm(tcm, axis=1))
        tot = mag.copy()
        if dv_arrival is not None:
            dvf = dxc[:, :3] @ phi_fc[3:, :3].T + (dxc[:, 3:] + tcm) @ phi_fc[3:, 3:].T
            dva = np.asarray(dv_arrival, float)
            tot = tot + system.mps(np.linalg.norm(dva[None, :] - dvf, axis=1) - np.linalg.norm(dva))
        out["mean"][k], out["p50"][k] = mag.mean(), _pct(mag, 50)
        out["p95"][k], out["p99"][k] = _pct(mag, 95), _pct(mag, 99)
        out["tot99"][k], out["totmean"][k] = _pct(tot, 99), tot.mean()
    return TcmResult(t_tcm, out["mean"], out["p50"], out["p95"], out["p99"], out["tot99"],
                     out["totmean"], _pct(miss, 99), system)


# ---------------------------------------------------------------------------
# Nonlinear Monte Carlo of a TCM strategy
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    t_tcm: np.ndarray  # nondim
    tcm_mps: np.ndarray  # (n, K) executed TCM magnitudes
    arrival_extra_mps: np.ndarray  # (n,) arrival burn minus nominal (0 for flybys)
    miss_km: np.ndarray  # (n,) residual position error at arrival
    system: System = EARTH_MOON

    @property
    def total_mps(self):
        return self.tcm_mps.sum(axis=1) + self.arrival_extra_mps

    def percentile(self, q: float) -> float:
        return _pct(self.total_mps, q)

    def summary(self) -> str:
        t = self.system.days(self.t_tcm)
        per = ", ".join(f"day {d:.2f}: mean {m:.2f} / p99 {_pct(c, 99):.2f}"
                        for d, m, c in zip(t, self.tcm_mps.mean(0), self.tcm_mps.T))
        return (f"Monte Carlo ({len(self.miss_km)} runs): TCMs [{per}] m/s\n"
                f"  total statistical dv: mean {self.total_mps.mean():.2f}, "
                f"p95 {self.percentile(95):.2f}, p99 {self.percentile(99):.2f} m/s\n"
                f"  residual miss at arrival: median {np.median(self.miss_km):.2f} km, "
                f"p99 {_pct(self.miss_km, 99):.2f} km")


def monte_carlo_tcm(x_post, tof: float, P0, t_tcm, system: System = EARTH_MOON,
                    target_state=None, n: int = 300, nav_sigma_km: float = 1.0,
                    nav_sigma_mps: float = 0.01, exec_frac: float = 0.01,
                    exec_floor_mps: float = 0.002, seed: int = 0) -> MonteCarloResult:
    """Nonlinear Monte Carlo of a TCM strategy.

    Each run samples an injection error from P0, propagates the true trajectory
    nonlinearly, and at each TCM time computes the linear re-targeting burn from a
    *noisy estimate* of the state (nav_sigma_km / nav_sigma_mps per axis), then
    executes it with a proportional plus floor execution error. With target_state
    (the target's state at arrival) the arrival burn to match it is included.
    """
    t_tcm = np.sort(np.atleast_1d(np.asarray(t_tcm, float)))
    states, phis = _stms(x_post, list(t_tcm) + [tof], system)
    x_nom_f, phi_f = states[-1], phis[-1]
    G = [_guidance(phi_f @ np.linalg.inv(p)) for p in phis[:-1]]
    rng = np.random.default_rng(seed)
    sr, sv = system.nd_length(nav_sigma_km), system.nd_velocity(nav_sigma_mps / 1000.0)
    floor = system.nd_velocity(exec_floor_mps / 1000.0)
    dva_nom = None if target_state is None else np.asarray(target_state)[3:] - x_nom_f[3:]
    tcms = np.zeros((n, len(t_tcm)))
    extra, miss = np.zeros(n), np.zeros(n)
    for i in range(n):
        x = np.asarray(x_post, float) + rng.multivariate_normal(np.zeros(6), P0)
        t_prev = 0.0
        for k, tc in enumerate(t_tcm):
            x = cr3bp.propagate_state(x, tc - t_prev, system)
            t_prev = tc
            est = x + np.concatenate((rng.normal(0, sr, 3), rng.normal(0, sv, 3)))
            cmd = G[k] @ (est - states[k])
            m = np.linalg.norm(cmd)
            err = rng.normal(0.0, 1.0, 3) * np.sqrt((exec_frac * m) ** 2 + floor ** 2)
            x[3:] += cmd + err
            tcms[i, k] = system.mps(np.linalg.norm(cmd + err))
        x = cr3bp.propagate_state(x, tof - t_prev, system)
        miss[i] = system.km(np.linalg.norm(x[:3] - x_nom_f[:3]))
        if dva_nom is not None:
            dva = np.asarray(target_state)[3:] - x[3:]
            extra[i] = system.mps(np.linalg.norm(dva) - np.linalg.norm(dva_nom))
    return MonteCarloResult(t_tcm, tcms, extra, miss, system)


# ---------------------------------------------------------------------------
# Station-keeping Monte Carlo
# ---------------------------------------------------------------------------

@dataclass
class StationKeepingResult:
    label: str
    interval_days: float
    duration_days: float
    dv_total_mps: np.ndarray  # (n,)  NaN for runs that were lost
    n_burns: np.ndarray  # (n,)
    max_dev_km: np.ndarray  # (n,) largest distance from the reference orbit's path
    lost: np.ndarray  # (n,) bool: deviation exceeded loss_km (control failed)
    efold_days: float = np.inf
    history_t_days: np.ndarray = field(repr=False, default=None)  # first run
    history_dev_km: np.ndarray = field(repr=False, default=None)
    history_burn_days: np.ndarray = field(repr=False, default=None)

    @property
    def per_year_mps(self):
        return self.dv_total_mps / self.duration_days * 365.25

    @property
    def success_rate(self) -> float:
        return float(np.mean(~self.lost))

    def summary(self) -> str:
        ok = ~self.lost
        if not ok.any():
            return (f"{self.label}: tracking every {self.interval_days:.2f} d -> control LOST in all "
                    f"{len(self.lost)} runs (interval too long vs e-folding time "
                    f"{self.efold_days:.2f} d)")
        py = self.per_year_mps[ok]
        lost = f"; LOST in {int(self.lost.sum())}/{len(self.lost)} runs" if self.lost.any() else ""
        return (f"{self.label}: tracking every {self.interval_days:.2f} d -> "
                f"{py.mean():.2f} m/s/yr mean, {_pct(py, 95):.2f} p95; "
                f"{self.n_burns[ok].mean():.1f} burns in {self.duration_days:.0f} d; "
                f"max deviation {_pct(self.max_dev_km[ok], 95):.1f} km (p95){lost}")


def stationkeeping_mc(orbit: PeriodicOrbit, interval_days: float, duration_days: float = 180.0,
                      nav_sigma_km: float = 1.0, nav_sigma_mps: float = 0.01,
                      exec_frac: float = 0.02, exec_floor_mps: float = 0.002,
                      min_burn_mps: float = 0.01, n: int = 20, seed: int = 0,
                      samples_per_interval: int = 20, loss_km: float = 5000.0,
                      efficiency_floor: float = 0.25) -> StationKeepingResult:
    """Monte Carlo of Floquet-mode station-keeping on a periodic orbit.

    Small deviations from a periodic orbit decompose into Floquet modes; only the
    unstable mode grows. Every interval_days the controller estimates the state
    (truth + navigation error), measures the unstable-mode coordinate
    alpha = w(t)^T dx with the left eigenvector w(t) of the monodromy matrix
    transported to the current phase, and applies the minimum-norm burn that
    cancels it: dv = -alpha w_v / |w_v|^2. Burns below min_burn_mps are skipped.
    Stable orbits have no unstable mode and need no burns.

    Controllability varies around the orbit: where a burn barely moves the
    unstable coordinate (|w_v| small, e.g. near an NRHO's perilune) the burn would
    be large and would amplify navigation noise. Passes where the efficiency
    |w_v|/|w| is below efficiency_floor times its best value around the orbit are
    skipped; the deviation is then corrected at the next efficient pass.

    Deviation is measured as distance from the reference orbit's *path* (phase
    drift along the orbit is harmless). A run is "lost" if that exceeds loss_km:
    the tracking interval is too long compared with the e-folding time.
    """
    system = orbit.system
    T = orbit.period
    dt = system.nd_time_days(interval_days)
    n_int = int(np.ceil(system.nd_time_days(duration_days) / dt))
    M = orbit.monodromy()
    w_eig, W = np.linalg.eig(M.T)  # left eigenvectors of M
    iu = int(np.argmax(np.abs(w_eig)))
    lam = float(np.abs(w_eig[iu]))
    from .manifolds import STABLE_TOL
    unstable = lam > 1.0 + STABLE_TOL  # numerically 1.000x counts as stable
    w0 = np.real(W[:, iu])
    efold = system.days(T / np.log(lam)) if unstable else np.inf
    cache = {}

    def mode(t):
        key = round(float(np.mod(t, T)), 9)
        if key not in cache:
            if key == 0.0:
                phi = np.eye(6)
            else:
                _, phi = cr3bp.propagate_state(orbit.state0, key, system, with_stm=True)
            w = np.linalg.solve(phi.T, w0)  # w(t)^T = w0^T Phi(t,0)^-1
            cache[key] = w / np.linalg.norm(w)
        return cache[key]

    eff_best = 0.0
    if unstable:
        for tau in np.linspace(0.0, T, 120, endpoint=False):
            w = mode(tau)
            eff_best = max(eff_best, float(np.linalg.norm(w[3:])))

    ref_path = orbit.trajectory(4000).r
    seg_a, seg_b = ref_path, np.roll(ref_path, -1, axis=0)  # closed polyline

    def path_dist(r):  # (m, 3) -> (m,) distance to the reference orbit's path
        # nearest vertex, then exact distance to its two adjacent segments (samples
        # are spaced unevenly in space: ~230 km apart near an NRHO perilune)
        d2 = ((r[:, None, :] - ref_path[None, :, :]) ** 2).sum(-1)
        k = d2.argmin(axis=1)
        best = np.sqrt(d2[np.arange(len(r)), k])
        for kk in (k, (k - 1) % len(ref_path)):
            a, b = seg_a[kk], seg_b[kk]
            ab = b - a
            u = np.clip(((r - a) * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
            best = np.minimum(best, np.linalg.norm(r - (a + u[:, None] * ab), axis=1))
        return best

    rng = np.random.default_rng(seed)
    sr, sv = system.nd_length(nav_sigma_km), system.nd_velocity(nav_sigma_mps / 1000.0)
    floor = system.nd_velocity(exec_floor_mps / 1000.0)
    thr = system.nd_velocity(min_burn_mps / 1000.0)
    loss = system.nd_length(loss_km)
    tot, nb, mx = np.zeros(n), np.zeros(n, int), np.zeros(n)
    lost = np.zeros(n, bool)
    hist_t, hist_d, hist_b = [], [], []
    for i in range(n):
        x = orbit.state0 + np.concatenate((rng.normal(0, sr, 3), rng.normal(0, sv, 3)))
        t = 0.0
        for _ in range(n_int):
            if unstable:
                est = x + np.concatenate((rng.normal(0, sr, 3), rng.normal(0, sv, 3)))
                w = mode(t)
                alpha = w @ (est - orbit.state_at(t))
                wv = w[3:]
                cmd = -alpha * wv / (wv @ wv)
                m = np.linalg.norm(cmd)
                efficient = np.linalg.norm(wv) >= efficiency_floor * eff_best
                if m >= thr and efficient:
                    err = rng.normal(0.0, 1.0, 3) * np.sqrt((exec_frac * m) ** 2 + floor ** 2)
                    x[3:] += cmd + err
                    tot[i] += system.mps(np.linalg.norm(cmd + err))
                    nb[i] += 1
                    if i == 0:
                        hist_b.append(float(system.days(t)))
            seg = cr3bp.propagate(x, t + dt, system, t0=t, n_points=samples_per_interval + 1)
            dev = path_dist(seg.r)
            mx[i] = max(mx[i], float(system.km(dev.max())))
            if i == 0:
                hist_t.extend(system.days(seg.t[:-1]).tolist())
                hist_d.extend(system.km(dev[:-1]).tolist())
            x, t = seg.x[-1].copy(), t + dt
            if dev.max() > loss:
                lost[i] = True
                break
    tot[lost] = np.nan
    return StationKeepingResult(orbit.family or "orbit", interval_days, n_int * interval_days,
                                tot, nb, mx, lost, float(efold), np.array(hist_t),
                                np.array(hist_d), np.array(hist_b))
