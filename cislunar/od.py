"""Orbit determination and orbit identification from sparse observations.

Given a handful of observations of an unknown target -- a position fix, angles
from a telescope, a range or range-rate, a direction of travel or the angular
rate of a streak in an image -- estimate its state with uncertainty, predict
where it goes, and identify which periodic (possibly stable) orbit it is on.

Pipeline (see architecture.md, ADR-25/26):

1. ``obs_*`` constructors build :class:`Observation` objects (user units: days,
   km, km/s, degrees, arcseconds) seen by an :class:`Observer` (Earth or Moon
   centre, a ground site on the rotating Earth, or another spacecraft) in
   ``"rotating"``, ``"inertial"`` or ``"equatorial"`` axes.
2. :func:`iod_guesses` builds initial states directly from the data where it
   can (position + velocity at one time; CR3BP Lambert between two fixes).
3. :class:`OrbitCatalog` -- periodic-orbit families sampled in phase -- supplies
   hypotheses when the data alone cannot fix all six state components.
4. :func:`fit` is a weighted batch least-squares estimator (Levenberg-Marquardt,
   STM partials, optional Gaussian prior) returning an :class:`ODResult` with
   covariance and a :meth:`ODResult.predict` method.
5. :func:`determine_orbit` runs it all and ranks the hypotheses;
   :func:`nearest_periodic` finds the closest exact periodic orbit (family,
   stability index, insertion Delta-v) and :func:`assess_fate` samples the
   covariance to say whether the object stays, escapes or impacts.

All internal quantities are nondimensional CR3BP in the barycentric rotating
frame; conversions happen in the constructors and in the reporting helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.stats import chi2 as _chi2

from . import cr3bp, frames, manifolds
from . import periodic as P
from .constants import EARTH_MOON, R_EARTH, SECONDS_PER_DAY, System
from .periodic import IVX, IVY, IVZ, IX, IZ, PeriodicOrbit

EPS_DEG = 23.44  # Earth equator to Earth-Moon plane (ADR-25)
OMEGA_EARTH = 7.2921159e-5  # rad/s, sidereal
ARCSEC = np.pi / (180.0 * 3600.0)
_ZHAT = np.array([0.0, 0.0, 1.0])


KINDS = {"position": 3, "velocity": 3, "angles": 2, "range": 1, "range_rate": 1,
         "velocity_direction": 2, "angle_rates": 2}
FRAMES = ("rotating", "inertial", "equatorial")


def _c_eq2in():
    e = np.radians(EPS_DEG)
    # equatorial -> Earth-Moon-plane inertial; the Earth's pole maps to (0, sin e, cos e)
    return np.array([[1.0, 0.0, 0.0], [0.0, np.cos(e), np.sin(e)], [0.0, -np.sin(e), np.cos(e)]]).T


_C_EQ2IN = _c_eq2in()


# ---------------------------------------------------------------------------
# Observers
# ---------------------------------------------------------------------------

class Observer:
    """Where an observation is taken from: ``state(t)`` gives its rotating-frame
    state (nondim) at nondim time t."""

    def __init__(self, fn, name: str = "observer"):
        self._fn = fn
        self.name = name

    def state(self, t: float) -> np.ndarray:
        return np.asarray(self._fn(float(t)), float)

    def __repr__(self):
        return f"Observer({self.name})"

    @classmethod
    def body(cls, which: str = "earth", system: System = EARTH_MOON) -> "Observer":
        """Centre of the Earth, the Moon or the barycentre (fixed in the rotating frame)."""
        x = np.concatenate((frames.center_position(system, which), np.zeros(3)))
        return cls(lambda t: x, f"{which} centre")

    @classmethod
    def ground_site(cls, lat_deg: float, lon_deg: float, alt_km: float = 0.0,
                    gst0_deg: float = 0.0, theta0: float = 0.0,
                    system: System = EARTH_MOON) -> "Observer":
        """A telescope or radar on the rotating Earth.

        The Earth spins at the sidereal rate about a pole tilted EPS_DEG from the
        Earth-Moon plane normal; ``gst0_deg`` is the sidereal angle of the site's
        meridian reference at t = 0 (equinox along the inertial x-axis, ADR-25).
        """
        rr = system.nd_length(R_EARTH + alt_km)
        w = OMEGA_EARTH * system.tstar  # nondim spin rate
        lat, lon, g0 = np.radians([lat_deg, lon_deg, gst0_deg])

        def fn(t):
            th = g0 + lon + w * t
            p = rr * np.array([np.cos(lat) * np.cos(th), np.cos(lat) * np.sin(th), np.sin(lat)])
            v = w * np.array([-p[1], p[0], 0.0])
            x_in = np.concatenate((_C_EQ2IN @ p, _C_EQ2IN @ v))
            return frames.inertial_to_rotating(t, x_in, system, "earth", theta0)

        return cls(fn, f"site {lat_deg:+.1f}/{lon_deg:+.1f}")

    @classmethod
    def spacecraft(cls, x0, t0: float = 0.0, system: System = EARTH_MOON,
                   name: str = "spacecraft") -> "Observer":
        """An observing spacecraft with rotating-frame state x0 (nondim) at time t0."""
        x0 = np.asarray(x0, float)
        cache = {}

        def fn(t):
            if t not in cache:
                cache[t] = cr3bp.propagate_state(x0, t - t0, system)
            return cache[t]

        return cls(fn, name)

    @classmethod
    def on_orbit(cls, orbit: PeriodicOrbit, phase: float = 0.0, name: str | None = None):
        """An observer flying a periodic orbit, at time ``phase`` along it when t = 0."""
        return cls(lambda t: orbit.state_at(phase + t), name or f"on {orbit.family}")


def _observer(o, system):
    if isinstance(o, Observer):
        return o
    if isinstance(o, str):
        return Observer.body(o, system)
    raise TypeError("observer must be an Observer or 'earth' / 'moon' / 'barycenter'")


# ---------------------------------------------------------------------------
# Observations and measurement models
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """One measurement (nondimensional / radians internally).

    kind  : one of KINDS. Vectors are target-minus-observer, in ``frame`` axes.
    z     : measured value, or None for a template to be filled by :func:`simulate`.
    sigma : 1-sigma noise per component (same units as z).
    """

    t: float
    kind: str
    z: np.ndarray | None
    sigma: np.ndarray
    observer: Observer
    frame: str = "rotating"
    theta0: float = 0.0
    label: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown observation kind {self.kind!r}; use one of {sorted(KINDS)}")
        if self.frame not in FRAMES:
            raise ValueError(f"unknown frame {self.frame!r}; use one of {FRAMES}")
        n = KINDS[self.kind]
        self.sigma = np.broadcast_to(np.asarray(self.sigma, float), (n,)).copy()
        if self.z is not None:
            self.z = np.asarray(self.z, float).reshape(n)

    @property
    def size(self) -> int:
        return KINDS[self.kind]

    def axes(self, t=None) -> np.ndarray:
        """Matrix taking rotating-frame vector components into measurement axes."""
        t = self.t if t is None else t
        if self.frame == "rotating":
            return np.eye(3)
        m = frames.rot_z(t + self.theta0)
        return _C_EQ2IN.T @ m if self.frame == "equatorial" else m

    def relative(self, X):
        """Target-minus-observer (rho, rhodot) in measurement axes, X (N, 6)."""
        X = np.atleast_2d(X)
        o = self.observer.state(self.t)
        rho = X[:, :3] - o[:3]
        rhod = X[:, 3:] - o[3:]
        if self.frame != "rotating":
            rhod = rhod + np.cross(_ZHAT, rho)
        m = self.axes()
        return rho @ m.T, rhod @ m.T

    def predict(self, X) -> np.ndarray:
        """Model value(s) h(X) for rotating-frame state(s) X (N, 6) -> (N, size)."""
        rho, rhod = self.relative(X)
        return _measure(self.kind, rho, rhod)

    def residual(self, X) -> np.ndarray:
        """(z - h(X)) with angle wrapping, (N, size)."""
        d = self.z - self.predict(X)
        if self.kind in ("angles", "velocity_direction"):
            d[:, 0] = (d[:, 0] + np.pi) % (2 * np.pi) - np.pi
        return d

    def jacobian(self, x, h: float = 1e-7) -> np.ndarray:
        """dh/dx at a single state (size, 6), central differences."""
        x = np.asarray(x, float)
        E = np.eye(6) * h
        Y = self.predict(np.vstack((x + E, x - E)))
        J = (Y[:6] - Y[6:]) / (2 * h)
        if self.kind in ("angles", "velocity_direction"):  # wrapped azimuth difference
            d = Y[:6, 0] - Y[6:, 0]
            J[:, 0] = ((d + np.pi) % (2 * np.pi) - np.pi) / (2 * h)
        return J.T


def _angles(v):
    r = np.linalg.norm(v, axis=1)
    return np.column_stack((np.arctan2(v[:, 1], v[:, 0]), np.arcsin(np.clip(v[:, 2] / r, -1, 1))))


def _measure(kind, rho, rhod):
    if kind == "position":
        return rho
    if kind == "velocity":
        return rhod
    if kind == "angles":
        return _angles(rho)
    if kind == "velocity_direction":
        return _angles(rhod)
    r = np.linalg.norm(rho, axis=1)
    if kind == "range":
        return r[:, None]
    rdot = np.einsum("ij,ij->i", rho, rhod) / r
    if kind == "range_rate":
        return rdot[:, None]
    # angle_rates: d(ra)/dt, d(dec)/dt
    x, y, z = rho.T
    vx, vy, vz = rhod.T
    rxy2 = x * x + y * y
    ra_dot = (x * vy - y * vx) / rxy2
    dec_dot = (vz * r - z * rdot) / (r * np.sqrt(rxy2))
    return np.column_stack((ra_dot, dec_dot))


def _mk(t_days, kind, z, sigma, observer, frame, theta0, system, label):
    return Observation(float(system.nd_time_days(t_days)), kind, z, sigma,
                       _observer(observer, system), frame, theta0, label)


def obs_position(t_days, r_km=None, sigma_km=10.0, observer="earth", frame="rotating",
                 theta0=0.0, system: System = EARTH_MOON, label=""):
    """Target position relative to the observer [km] (e.g. radar range + angles)."""
    z = None if r_km is None else system.nd_length(np.asarray(r_km, float))
    return _mk(t_days, "position", z, system.nd_length(sigma_km), observer, frame, theta0, system, label)


def obs_velocity(t_days, v_kms=None, sigma_mps=10.0, observer="earth", frame="rotating",
                 theta0=0.0, system: System = EARTH_MOON, label=""):
    """Target velocity relative to the observer [km/s], in ``frame`` (inertial velocities
    include the frame-rotation term)."""
    z = None if v_kms is None else system.nd_velocity(np.asarray(v_kms, float))
    return _mk(t_days, "velocity", z, system.nd_velocity(np.asarray(sigma_mps) / 1000.0),
               observer, frame, theta0, system, label)


def obs_angles(t_days, ra_deg=None, dec_deg=None, sigma_arcsec=2.0, observer="earth",
               frame="equatorial", theta0=0.0, system: System = EARTH_MOON, label=""):
    """Line-of-sight angles (right ascension / declination style) of the target."""
    z = None if ra_deg is None else np.radians([ra_deg, dec_deg])
    return _mk(t_days, "angles", z, np.asarray(sigma_arcsec) * ARCSEC, observer, frame, theta0,
               system, label)


def obs_range(t_days, range_km=None, sigma_km=1.0, observer="earth", system: System = EARTH_MOON,
              label=""):
    z = None if range_km is None else [system.nd_length(range_km)]
    return _mk(t_days, "range", z, system.nd_length(sigma_km), observer, "rotating", 0.0, system, label)


def obs_range_rate(t_days, rdot_mps=None, sigma_mps=1.0, observer="earth",
                   system: System = EARTH_MOON, label=""):
    z = None if rdot_mps is None else [system.nd_velocity(rdot_mps / 1000.0)]
    return _mk(t_days, "range_rate", z, system.nd_velocity(np.asarray(sigma_mps) / 1000.0),
               observer, "rotating", 0.0, system, label)


def obs_velocity_direction(t_days, az_deg=None, el_deg=None, sigma_deg=2.0, observer="moon",
                           frame="rotating", theta0=0.0, system: System = EARTH_MOON, label=""):
    """Direction of travel only (no speed), as azimuth/elevation of the velocity
    relative to the observer. With observer='moon' this is the Moon-relative heading."""
    z = None if az_deg is None else np.radians([az_deg, el_deg])
    return _mk(t_days, "velocity_direction", z, np.radians(sigma_deg), observer, frame, theta0,
               system, label)


def obs_angle_rates(t_days, ra_rate_arcsec_s=None, dec_rate_arcsec_s=None, sigma_arcsec_s=0.05,
                    observer="earth", frame="equatorial", theta0=0.0, system: System = EARTH_MOON,
                    label=""):
    """Angular rates of the line of sight, e.g. from a streak's length and direction
    in one exposure [arcsec/s]."""
    k = ARCSEC * system.tstar  # arcsec/s -> rad per nondim time
    z = None if ra_rate_arcsec_s is None else np.array([ra_rate_arcsec_s, dec_rate_arcsec_s]) * k
    return _mk(t_days, "angle_rates", z, np.asarray(sigma_arcsec_s) * k, observer, frame, theta0,
               system, label)


def simulate(observations, x_epoch, epoch: float = 0.0, system: System = EARTH_MOON,
             rng=None, noise: bool = True):
    """Fill templates with synthetic measurements of a true state x_epoch (nondim, at
    nondim time ``epoch``), adding Gaussian noise with each observation's sigma."""
    rng = np.random.default_rng(rng)
    times = [o.t for o in observations]
    states = _propagate_many(x_epoch, epoch, times, system, with_stm=False)
    out = []
    for o, (x, _) in zip(observations, states):
        z = o.predict(x)[0]
        if noise:
            z = z + rng.normal(0.0, o.sigma)
        out.append(Observation(o.t, o.kind, z, o.sigma.copy(), o.observer, o.frame, o.theta0, o.label))
    return out


# ---------------------------------------------------------------------------
# Propagation to observation times (with STM)
# ---------------------------------------------------------------------------

def _propagate_many(x0, t0, times, system: System, with_stm: bool = True):
    """States (and STMs) at each of ``times`` from x0 at t0. Returns list of (x, phi)."""
    x0 = np.asarray(x0, float)
    times = np.asarray(times, float)
    out = [None] * len(times)
    for sign in (1.0, -1.0):
        idx = [i for i, t in enumerate(times) if (t - t0) * sign > 0]
        for i, t in enumerate(times):
            if t == t0:
                out[i] = (x0.copy(), np.eye(6))
        if not idx:
            continue
        dts = np.unique([times[i] - t0 for i in idx])
        dts = dts if sign > 0 else dts[::-1]
        y0 = np.concatenate((x0, np.eye(6).ravel())) if with_stm else x0
        fun = cr3bp.eom_stm if with_stm else cr3bp.eom
        sol = solve_ivp(fun, (0.0, dts[-1]), y0, method="DOP853", rtol=cr3bp.RTOL,
                        atol=cr3bp.ATOL, args=(system.mu,), t_eval=dts,
                        events=cr3bp._impact_events(system))
        if sol.status == 1:
            raise cr3bp.ImpactError("trajectory impacts a primary between observations")
        if not sol.success:
            raise RuntimeError(f"propagation failed: {sol.message}")
        lookup = {float(t): sol.y[:, k] for k, t in enumerate(sol.t)}
        for i in idx:
            y = lookup[float(times[i] - t0)]
            out[i] = (y[:6], y[6:].reshape(6, 6) if with_stm else None)
    return out


# ---------------------------------------------------------------------------
# Batch least squares
# ---------------------------------------------------------------------------

@dataclass
class ODResult:
    """A fitted epoch state with covariance."""

    x: np.ndarray  # epoch state, nondim rotating
    epoch: float  # nondim time
    P: np.ndarray  # 6x6 covariance, nondim
    chi2: float  # data chi-square
    chi2_prior: float
    n_meas: int
    converged: bool
    iterations: int
    prior_used: bool
    info_rank: int  # number of state directions the data alone constrains better than the prior scale
    residuals: list = field(default_factory=list)  # normalised residual vector per observation
    source: str = ""
    system: System = EARTH_MOON
    match: "OrbitMatch | None" = None
    weight: float = 1.0

    @property
    def score(self) -> float:
        return self.chi2 + self.chi2_prior

    @property
    def dof(self) -> int:
        return max(self.n_meas - (0 if self.prior_used else 6), 0)

    @property
    def p_value(self) -> float:
        """Probability of a chi-square at least this large if the model is right."""
        return float(_chi2.sf(self.chi2, self.dof)) if self.dof > 0 else float("nan")

    @property
    def rms(self) -> float:
        return float(np.sqrt(self.chi2 / max(self.n_meas, 1)))

    def state_km(self) -> np.ndarray:
        return self.system.state_to_dim(self.x)

    def sigma_pos_km(self) -> float:
        """RSS 1-sigma position uncertainty at epoch [km]."""
        return float(self.system.km(np.sqrt(np.trace(self.P[:3, :3]))))

    def sigma_vel_mps(self) -> float:
        return float(self.system.mps(np.sqrt(np.trace(self.P[3:, 3:]))))

    def predict(self, t_days):
        """Predicted states [km, km/s] and RSS 1-sigma position [km] at times t_days
        (days from the time origin of the observations). Linear covariance mapping."""
        s = self.system
        t = np.atleast_1d(s.nd_time_days(np.asarray(t_days, float)))
        res = _propagate_many(self.x, self.epoch, t, s, with_stm=True)
        X = np.array([s.state_to_dim(x) for x, _ in res])
        sig = np.array([s.km(np.sqrt(np.trace((phi @ self.P @ phi.T)[:3, :3]))) for _, phi in res])
        return X, sig

    def trajectory(self, span_days: float, n_points: int = 800) -> cr3bp.Trajectory:
        tr = cr3bp.propagate(self.x, self.system.nd_time_days(span_days), self.system,
                             n_points=n_points, label=self.source)
        tr.t = tr.t + self.epoch
        return tr

    def samples(self, n: int, rng=None) -> np.ndarray:
        """n epoch states drawn from N(x, P)."""
        rng = np.random.default_rng(rng)
        w, V = np.linalg.eigh(0.5 * (self.P + self.P.T))
        L = V * np.sqrt(np.clip(w, 0.0, None))
        return self.x + rng.standard_normal((n, 6)) @ L.T


def _stack(observations, x, epoch, system):
    res = _propagate_many(x, epoch, [o.t for o in observations], system, with_stm=True)
    r, A, per = [], [], []
    for o, (xk, phi) in zip(observations, res):
        rk = o.residual(xk)[0] / o.sigma
        Ak = (o.jacobian(xk) @ phi) / o.sigma[:, None]
        r.append(rk)
        A.append(Ak)
        per.append(rk)
    return np.concatenate(r), np.vstack(A), per


def _prior_scale(system, pos_km, vel_mps):
    return np.concatenate((np.full(3, system.nd_length(pos_km)),
                           np.full(3, system.nd_velocity(vel_mps / 1000.0))))


def info_rank(observations, x, epoch: float = None, system: System = EARTH_MOON,
              pos_km: float = 2000.0, vel_mps: float = 50.0) -> int:
    """How many state directions the observations constrain more tightly than
    (pos_km, vel_mps) would -- 6 means the data alone determines the orbit."""
    epoch = min(o.t for o in observations) if epoch is None else epoch
    _, A, _ = _stack(observations, x, epoch, system)
    s = np.linalg.svd(A * _prior_scale(system, pos_km, vel_mps), compute_uv=False)
    return int(np.sum(s > 1.0))


def fit(observations, x_guess, epoch: float | None = None, system: System = EARTH_MOON,
        prior=None, max_iter: int = 40, source: str = "") -> ODResult:
    """Weighted batch least squares for the epoch state.

    observations : list of Observation (with z filled)
    x_guess      : initial epoch state (nondim)
    epoch        : nondim epoch time (default: earliest observation)
    prior        : optional (x_bar, sigma) with sigma a (6,) array of 1-sigma (nondim),
                   added as pseudo-measurements of the state.
    """
    obs = [o for o in observations if o.z is not None]
    if not obs:
        raise ValueError("no observations with measured values")
    epoch = min(o.t for o in obs) if epoch is None else epoch
    n_meas = sum(o.size for o in obs)
    if prior is not None:
        xbar, psig = np.asarray(prior[0], float), np.asarray(prior[1], float)
        Pi = np.diag(1.0 / psig**2)
    else:
        xbar, Pi = None, np.zeros((6, 6))

    def prior_cost(x):
        return 0.0 if xbar is None else float(np.sum(((x - xbar) / psig) ** 2))

    def evaluate(x):
        try:
            r, A, per = _stack(obs, x, epoch, system)
        except (cr3bp.ImpactError, RuntimeError):
            return None
        return r, A, per, float(r @ r) + prior_cost(x)

    x = np.asarray(x_guess, float).copy()
    ev = evaluate(x)
    if ev is None:
        raise RuntimeError("initial guess impacts a primary or cannot be propagated")
    r, A, per, cost = ev
    lam, converged, it = 1e-3, False, 0
    for it in range(1, max_iter + 1):
        H = A.T @ A + Pi
        g = A.T @ r + (Pi @ (xbar - x) if xbar is not None else 0.0)
        d = np.diag(H).copy()
        d = np.maximum(d, 1e-12 * max(d.max(), 1.0))
        accepted = False
        while lam < 1e10:
            try:
                dx = np.linalg.solve(H + lam * np.diag(d), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            ev = evaluate(x + dx)
            if ev is not None and ev[3] <= cost:
                accepted = True
                break
            lam *= 10
        if not accepted:
            converged = cost < 1e30 and np.linalg.norm(g) < 1e-6 * (1 + cost)
            break
        drop = cost - ev[3]
        x = x + dx
        r, A, per, cost = ev
        lam = max(lam / 10, 1e-9)
        if drop <= 1e-10 * (1.0 + cost) or np.linalg.norm(dx) < 1e-13:
            converged = True
            break
    H = A.T @ A + Pi
    try:
        Pcov = np.linalg.inv(H)
        if not np.all(np.isfinite(Pcov)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        Pcov = np.linalg.pinv(H)
    s = np.linalg.svd(A * _prior_scale(system, 2000.0, 50.0), compute_uv=False)
    chi_d = float(r @ r)
    return ODResult(x, epoch, Pcov, chi_d, prior_cost(x), n_meas, bool(converged), it,
                    xbar is not None, int(np.sum(s > 1.0)), per, source, system)


# ---------------------------------------------------------------------------
# Initial orbit determination
# ---------------------------------------------------------------------------

def _fixes(observations):
    """Rotating-frame target positions (and velocities) recoverable from the data,
    keyed by time: {t: {"r": (3,), "v": (3,) optional}}."""
    out = {}
    for o in observations:
        if o.z is None:
            continue
        m = o.axes()
        ob = o.observer.state(o.t)
        if o.kind == "position":
            out.setdefault(o.t, {})["rho"] = (m.T @ o.z, o)
    # angles + range at the same time and observer
    for o in observations:
        if o.kind != "angles" or o.z is None:
            continue
        for q in observations:
            if q.kind == "range" and q.t == o.t and q.observer is o.observer and q.z is not None:
                ra, dec = o.z
                u = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
                out.setdefault(o.t, {}).setdefault("rho", (o.axes().T @ (q.z[0] * u), o))
    fixes = {}
    for t, d in out.items():
        if "rho" not in d:
            continue
        rho_rot, o = d["rho"]
        ob = o.observer.state(t)
        fixes[t] = {"r": ob[:3] + rho_rot}
        for q in observations:
            if q.kind == "velocity" and q.t == t and q.z is not None:
                obq = q.observer.state(t)
                rho_q = fixes[t]["r"] - obq[:3]
                v = q.axes().T @ q.z
                if q.frame != "rotating":
                    v = v - np.cross(_ZHAT, rho_q)
                fixes[t]["v"] = obq[3:] + v
    return fixes


def iod_guesses(observations, epoch: float | None = None, system: System = EARTH_MOON):
    """Initial epoch states computed directly from the data: (label, x_epoch) list.

    * position + velocity at the same time -> that state;
    * two position fixes -> CR3BP Lambert (single shooting seeded by a straight line
      and by Earth/Moon two-body Lambert), for a few pairs of fixes.
    """
    from . import targeting  # local import: targeting imports heavy modules

    obs = [o for o in observations if o.z is not None]
    epoch = min(o.t for o in obs) if epoch is None else epoch
    fx = _fixes(obs)
    out = []

    def to_epoch(x, t):
        try:
            return cr3bp.propagate_state(x, epoch - t, system, stop_on_impact=True)
        except (cr3bp.ImpactError, RuntimeError):
            return None

    for t, d in sorted(fx.items()):
        if "v" in d:
            x = to_epoch(np.concatenate((d["r"], d["v"])), t)
            if x is not None:
                out.append(("IOD: position + velocity", x))
    ts = sorted(fx)
    pairs = []
    if len(ts) >= 2:
        pairs = list(dict.fromkeys([(ts[0], ts[-1]), (ts[0], ts[1]), (ts[-2], ts[-1])]))
    for t1, t2 in pairs:
        r1, r2 = fx[t1]["r"], fx[t2]["r"]
        tof = t2 - t1
        seeds = [("line", (r2 - r1) / tof)] + targeting.lambert_seeds(
            np.concatenate((r1, np.zeros(3))), r2, tof, system, t1)
        for name, v0 in seeds:
            try:
                v, _ = targeting.shoot(np.concatenate((r1, v0)), r2, tof, system)
            except RuntimeError:
                continue
            x = to_epoch(np.concatenate((r1, v)), t1)
            if x is not None:
                out.append((f"IOD: Lambert between fixes ({name} seed)", x))
    # drop duplicates
    uniq = []
    for lab, x in out:
        if all(np.linalg.norm(x - y) > 1e-7 for _, y in uniq):
            uniq.append((lab, x))
    return uniq


# ---------------------------------------------------------------------------
# Periodic-orbit catalogue
# ---------------------------------------------------------------------------

CATALOG_VERSION = 1
DEFAULT_FAMILIES = ("L1 Lyapunov", "L2 Lyapunov", "L1 southern halo", "L1 northern halo",
                    "L2 southern halo", "L2 northern halo", "DRO")
# correction recipe per family type: (fixed param, free, targets), with a fallback
_PLANAR = [(IX, (IVY,), (IVX,))]
_SPATIAL = [(IZ, (IX, IVY), (IVX, IVZ)), (IX, (IZ, IVY), (IVX, IVZ))]


def _sweep(start: PeriodicOrbit, values, step: float, free, targets) -> list[PeriodicOrbit]:
    """Members of a family at each state0[IX] in ``values`` (natural continuation)."""
    out, cur = [], start
    for v in values:
        try:
            cur = P.continue_to(cur, IX, v, step, free=free, targets=targets)
        except RuntimeError:
            break
        out.append(cur)
    return out


def _family_orbits(name: str, system: System) -> list[PeriodicOrbit]:
    """Continue one family and return its members (unsampled)."""
    lp = cr3bp.lagrange_points(system.mu)
    if name.endswith("Lyapunov"):
        pt = int(name[1])
        start = P.lyapunov(pt, 2000.0, system)
        xl = lp[f"L{pt}"][0]
        lim = system.nd_length(32000.0)
        fam = P.continue_arclength(start, 0.003, stop=lambda o: abs(o.state0[IX] - xl) > lim,
                                   free=(IX, IVY), targets=(IVX,), direction=(1.0, 0.0),
                                   ds_max=0.006)
    elif "halo" in name:
        pt = int(name[1])
        if "northern" in name:
            return [_mirror(o, name) for o in _family_orbits(name.replace("northern", "southern"), system)]
        start = P.halo(pt, 5000.0, "south", system)
        st = {"min": np.inf}

        def stop(o):  # down the family to the NRHOs; stop once the period turns back up
            st["min"] = min(st["min"], o.period_days)
            return o.period_days < 6.0 or (st["min"] < 10.0 and o.period_days > st["min"] + 0.5)

        fam = P.continue_arclength(start, 0.004, stop=stop, ds_max=0.008,
                                   direction=(0.0, 1.0 if pt == 2 else -1.0, 0.0))
    elif name == "DRO":
        start = P.dro(5000.0, system)
        vals = 1.0 - system.mu - system.nd_length(np.geomspace(5000.0, 120000.0, 40))
        fam = _sweep(start, vals, system.nd_length(1500.0), (IVY,), (IVX,))
    else:
        raise ValueError(f"unknown family {name!r}")
    for o in fam:
        o.family = name
    return fam


def _mirror(o: PeriodicOrbit, name: str) -> PeriodicOrbit:
    s = o.state0.copy()
    s[IZ], s[IVZ] = -s[IZ], -s[IVZ]
    return PeriodicOrbit(s, o.period, o.system, name)


def _thin(fam, n_max):
    """Keep about n_max members, evenly spaced in (state0, period)."""
    if len(fam) <= n_max:
        return fam
    Z = np.array([np.append(o.state0, o.period / 10.0) for o in fam])
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(Z, axis=0), axis=1))))
    keep = sorted(set(int(np.argmin(np.abs(s - v))) for v in np.linspace(0, s[-1], n_max)))
    return [fam[i] for i in keep]


def _build_family_task(args):
    name, system, n_members, n_phase = args
    fam = _thin(_family_orbits(name, system), n_members)
    rows = []
    for o in fam:
        tr = cr3bp.propagate(o.state0, o.period, system, n_points=n_phase + 1)
        X = tr.x[:-1]
        F = np.array([cr3bp.eom(0.0, x, system.mu) for x in X])
        dm = np.linalg.norm(tr.r - system.pos2, axis=1)
        de = np.linalg.norm(tr.r - system.pos1, axis=1)
        if system.km(dm.min()) < system.r2 + 50.0 or system.km(de.min()) < system.r1 + 100.0:
            continue  # grazes or hits a primary
        rows.append(dict(state0=o.state0, period=o.period, jacobi=o.jacobi,
                         nu=o.stability_index(), X=X, F=F,
                         rp=float(system.km(dm.min())), ra=float(system.km(dm.max()))))
    return name, rows


@dataclass
class CatalogMember:
    index: int
    family: str
    state0: np.ndarray
    period: float
    jacobi: float
    nu: float
    perilune_km: float
    apolune_km: float
    system: System = EARTH_MOON

    @property
    def period_days(self) -> float:
        return float(self.system.days(self.period))

    @property
    def stable(self) -> bool:
        return self.nu <= 1.0 + 1e-3

    def orbit(self) -> PeriodicOrbit:
        return PeriodicOrbit(self.state0.copy(), self.period, self.system, self.family)

    def describe(self) -> str:
        return (f"{self.family}, T = {self.period_days:.2f} d, Moon range "
                f"{self.perilune_km:,.0f}-{self.apolune_km:,.0f} km, nu = {self.nu:.3g}")


class OrbitCatalog:
    """Periodic-orbit families sampled uniformly in time (phase) along each member.

    ``X[m, j]`` is member m's state at time ``j * period_m / n_phase``; ``F`` holds
    the state derivatives for cubic Hermite interpolation between samples.
    """

    def __init__(self, members: list[CatalogMember], X: np.ndarray, F: np.ndarray,
                 system: System = EARTH_MOON):
        self.members = members
        self.X = X
        self.F = F
        self.system = system
        self.period = np.array([m.period for m in members])
        self.fam_of = np.array([m.family for m in members])

    def __len__(self):
        return len(self.members)

    @property
    def families(self) -> list[str]:
        return list(dict.fromkeys(self.fam_of))

    @property
    def n_phase(self) -> int:
        return self.X.shape[1]

    # --- construction / persistence -------------------------------------
    @classmethod
    def build(cls, families=DEFAULT_FAMILIES, n_members: int = 40, n_phase: int = 512,
              system: System = EARTH_MOON, workers: int = 1, verbose: bool = False) -> "OrbitCatalog":
        """Continue each family and sample it. workers > 1 uses processes (Windows:
        call from under ``if __name__ == "__main__":``)."""
        tasks = [(f, system, n_members, n_phase) for f in families]
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(min(workers, len(tasks))) as ex:
                results = list(ex.map(_build_family_task, tasks))
        else:
            results = [_build_family_task(t) for t in tasks]
        members, Xs, Fs = [], [], []
        for name, rows in results:
            if verbose:
                print(f"  catalogue: {name}: {len(rows)} members")
            for r in rows:
                members.append(CatalogMember(len(members), name, r["state0"], r["period"], r["jacobi"],
                                             r["nu"], r["rp"], r["ra"], system))
                Xs.append(r["X"])
                Fs.append(r["F"])
        return cls(members, np.array(Xs), np.array(Fs), system)

    def save(self, path) -> None:
        m = self.members
        np.savez_compressed(
            path, version=CATALOG_VERSION, mu=self.system.mu, X=self.X, F=self.F,
            family=np.array([x.family for x in m]), state0=np.array([x.state0 for x in m]),
            period=self.period, jacobi=np.array([x.jacobi for x in m]),
            nu=np.array([x.nu for x in m]), rp=np.array([x.perilune_km for x in m]),
            ra=np.array([x.apolune_km for x in m]))

    @classmethod
    def load(cls, path, system: System = EARTH_MOON) -> "OrbitCatalog":
        d = np.load(path, allow_pickle=False)
        if int(d["version"]) != CATALOG_VERSION or abs(float(d["mu"]) - system.mu) > 1e-12:
            raise ValueError("catalogue file is from another version or system")
        members = [CatalogMember(i, str(f), s, float(p), float(j), float(n), float(a), float(b), system)
                   for i, (f, s, p, j, n, a, b) in enumerate(zip(d["family"], d["state0"], d["period"],
                                                                 d["jacobi"], d["nu"], d["rp"], d["ra"]))]
        return cls(members, d["X"], d["F"], system)

    @classmethod
    def default(cls, path=None, workers: int = 1, verbose: bool = False,
                system: System = EARTH_MOON) -> "OrbitCatalog":
        """Load the default catalogue from ``~/.cache/cislunar`` or build it (about a
        minute per family) and save it there."""
        path = Path(path) if path else Path.home() / ".cache" / "cislunar" / f"od_catalog_v{CATALOG_VERSION}.npz"
        if path.exists():
            try:
                return cls.load(path, system)
            except (ValueError, KeyError, OSError):
                pass
        if verbose:
            print(f"building orbit catalogue -> {path}")
        cat = cls.build(system=system, workers=workers, verbose=verbose)
        path.parent.mkdir(parents=True, exist_ok=True)
        cat.save(path)
        return cat

    # --- interpolation ----------------------------------------------------
    def states(self, m, tau) -> np.ndarray:
        """States of member(s) m at time(s) tau along the orbit (nondim), Hermite."""
        m = np.asarray(m)
        tau = np.asarray(tau, float)
        n = self.n_phase
        T = self.period[m]
        h = T / n
        u = np.mod(tau, T) / h
        j = np.floor(u).astype(int) % n
        s = (u - np.floor(u))[..., None]
        j1 = (j + 1) % n
        x0, x1 = self.X[m, j], self.X[m, j1]
        f0, f1 = self.F[m, j], self.F[m, j1]
        hh = h[..., None]
        s2, s3 = s * s, s * s * s
        return ((2 * s3 - 3 * s2 + 1) * x0 + (s3 - 2 * s2 + s) * hh * f0
                + (-2 * s3 + 3 * s2) * x1 + (s3 - s2) * hh * f1)

    # --- matching ------------------------------------------------------------
    def match(self, observations, epoch: float | None = None, top: int = 8,
              model_pos_km: float = 2000.0, model_vel_mps: float = 50.0, per_family: int = 2):
        """Rank catalogue states by consistency with the observations.

        Each member is tried at every sampled phase as the epoch state; observation
        noise is inflated by the catalogue's own resolution (model_pos_km,
        model_vel_mps). Returns [(member index, epoch phase tau, chi2)] with at most
        ``per_family`` entries per family among the first picks.
        """
        obs = [o for o in observations if o.z is not None]
        epoch = min(o.t for o in obs) if epoch is None else epoch
        s = self.system
        sp, sv = s.nd_length(model_pos_km), s.nd_velocity(model_vel_mps / 1000.0)
        M, n = len(self.members), self.n_phase
        mm = np.repeat(np.arange(M), n)
        tau0 = (np.tile(np.arange(n), M) * (self.period / n)[mm])
        chi = np.zeros(M * n)
        for o in obs:
            X = self.states(mm, tau0 + (o.t - epoch))
            d = o.residual(X)
            rho, rhod = o.relative(X)
            sig = np.broadcast_to(_model_sigma(o.kind, rho, rhod, sp, sv), d.shape)
            chi += np.sum(d**2 / (o.sigma**2 + sig**2), axis=1)
        chi = chi.reshape(M, n)
        jbest = np.argmin(chi, axis=1)
        cbest = chi[np.arange(M), jbest]
        order = np.argsort(cbest)
        picks, count = [], {}
        for i in order:
            f = self.fam_of[i]
            if count.get(f, 0) < per_family:
                picks.append(i)
                count[f] = count.get(f, 0) + 1
            if len(picks) >= top:
                break
        for i in order:  # fill with the next best overall
            if len(picks) >= top:
                break
            if i not in picks:
                picks.append(i)
        picks.sort(key=lambda i: cbest[i])
        return [(int(i), float(jbest[i] * self.period[i] / n), float(cbest[i])) for i in picks]

    def nearest_samples(self, x, pos_km: float = 1000.0, vel_mps: float = 20.0, k: int = 5):
        """The k members whose sampled states come closest to x (scaled distance)."""
        sc = _prior_scale(self.system, pos_km, vel_mps)
        d2 = np.sum(((self.X - x) / sc) ** 2, axis=2)
        j = np.argmin(d2, axis=1)
        best = d2[np.arange(len(self.members)), j]
        order = np.argsort(best)[:k]
        return [(int(i), float(j[i] * self.period[i] / self.n_phase), float(np.sqrt(best[i])))
                for i in order]


def _model_sigma(kind, rho, rhod, sp, sv):
    """Catalogue-resolution noise mapped into each measurement kind."""
    r = np.linalg.norm(rho, axis=1)[:, None]
    v = np.linalg.norm(rhod, axis=1)[:, None]
    if kind == "position":
        return np.full_like(rho, sp)
    if kind == "velocity":
        return np.full_like(rho, sv)
    if kind == "angles":
        return np.repeat(sp / r, 2, axis=1)
    if kind == "range":
        return np.full_like(r, sp)
    if kind == "range_rate":
        return np.full_like(r, sv)
    if kind == "velocity_direction":
        return np.repeat(np.minimum(sv / np.maximum(v, 1e-12), np.pi), 2, axis=1)
    return np.repeat(sv / r + sp * v / r**2, 2, axis=1)  # angle_rates


# ---------------------------------------------------------------------------
# Nearest exact periodic orbit
# ---------------------------------------------------------------------------

@dataclass
class OrbitMatch:
    """The periodic orbit closest to a state, and what it would take to join it."""

    family: str
    orbit: PeriodicOrbit
    phase: float  # nondim time along the orbit of the closest point
    dr_km: float  # position offset from that point
    dv_mps: float  # velocity offset there (an insertion Delta-v estimate)
    nu: float  # stability index of the orbit
    member: int  # nearest catalogue member
    refined: bool

    @property
    def stable(self) -> bool:
        return self.nu <= 1.0 + 1e-3

    @property
    def period_days(self) -> float:
        return float(self.orbit.period_days)

    def describe(self) -> str:
        stab = "linearly stable" if self.stable else f"unstable (nu = {self.nu:.3g})"
        return (f"{self.family}, T = {self.period_days:.2f} d, {stab}; offset {self.dr_km:,.0f} km, "
                f"{self.dv_mps:,.1f} m/s")


def _recipe(family):
    return _PLANAR if ("Lyapunov" in family or family == "DRO") else _SPATIAL


def _correct_between(a: CatalogMember, b: CatalogMember, s: float) -> PeriodicOrbit | None:
    guess = (1 - s) * a.state0 + s * b.state0
    for _, free, targets in _recipe(a.family):
        try:
            o = P.correct_symmetric(guess, a.system, free=free, targets=targets)
            o.family = a.family
            return o
        except RuntimeError:
            continue
    return None


def _closest_on(orbit: PeriodicOrbit, x, sc, n: int = 1500):
    tr = cr3bp.propagate(orbit.state0, orbit.period, orbit.system, n_points=n)
    d = np.linalg.norm((tr.x - x) / sc, axis=1)
    k = int(np.argmin(d))
    h = orbit.period / (n - 1)
    f = lambda t: np.linalg.norm((orbit.state_at(t) - x) / sc)  # noqa: E731
    r = minimize_scalar(f, bounds=(tr.t[k] - h, tr.t[k] + h), method="bounded",
                        options={"xatol": 1e-9})
    return float(r.x % orbit.period), float(r.fun)


def nearest_periodic(x, catalog: OrbitCatalog, refine: bool = True, pos_km: float = 1000.0,
                     vel_mps: float = 20.0) -> OrbitMatch:
    """The periodic orbit closest to state x (distance scaled by pos_km, vel_mps).

    The nearest catalogue sample picks a member; with refine=True the family is
    re-corrected between that member and its neighbours (bounded 1-D search) to
    find the closest exact member.
    """
    s = catalog.system
    sc = _prior_scale(s, pos_km, vel_mps)
    m, _, _ = catalog.nearest_samples(x, pos_km, vel_mps, k=1)[0]
    mem = catalog.members[m]
    best_orbit, refined = mem.orbit(), False
    tbest, dbest = _closest_on(best_orbit, x, sc)
    if refine:
        fam_idx = [i for i in range(len(catalog)) if catalog.fam_of[i] == mem.family]
        k = fam_idx.index(m)
        for nb in (k - 1, k + 1):
            if not 0 <= nb < len(fam_idx):
                continue
            other = catalog.members[fam_idx[nb]]
            cache = {}

            def cost(u, other=other, cache=cache):
                o = _correct_between(mem, other, u)
                if o is None:
                    return 1e9
                t, d = _closest_on(o, x, sc, n=800)
                cache[u] = (o, t, d)
                return d

            r = minimize_scalar(cost, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-3,
                                                                                    "maxiter": 20})
            if r.x in cache and cache[r.x][2] < dbest:
                o, _, _ = cache[r.x]
                t, d = _closest_on(o, x, sc)
                if d < dbest:
                    best_orbit, tbest, dbest, refined = o, t, d, True
    xo = best_orbit.state_at(tbest)
    nu = best_orbit.stability_index() if refined else mem.nu
    return OrbitMatch(mem.family, best_orbit, tbest, float(s.km(np.linalg.norm(xo[:3] - x[:3]))),
                      float(s.mps(np.linalg.norm(xo[3:] - x[3:]))), float(nu), m, refined)


def _orbit_dense(o: PeriodicOrbit):
    sol = solve_ivp(cr3bp.eom, (0.0, o.period), o.state0, method="DOP853", rtol=cr3bp.RTOL,
                    atol=cr3bp.ATOL, args=(o.system.mu,), dense_output=True)
    return lambda t: sol.sol(np.mod(np.atleast_1d(t), o.period)).T


def _obs_chi2_on(o: PeriodicOrbit, obs, epoch: float, n: int = 1000):
    """Best epoch phase of exact periodic orbit o for the observations: (tau, chi2)."""
    f = _orbit_dense(o)

    def chi(taus):
        c = np.zeros(len(taus))
        for ob in obs:
            c += np.sum((ob.residual(f(taus + (ob.t - epoch))) / ob.sigma) ** 2, axis=1)
        return c

    taus = np.linspace(0.0, o.period, n, endpoint=False)
    c = chi(taus)
    j = int(np.argmin(c))
    h = o.period / n
    r = minimize_scalar(lambda t: chi(np.array([t]))[0], bounds=(taus[j] - h, taus[j] + h),
                        method="bounded", options={"xatol": 1e-10})
    return float(r.x % o.period), float(r.fun)


def refine_hypothesis(catalog: OrbitCatalog, m: int, observations, epoch: float | None = None,
                      max_moves: int = 6):
    """Slide along member m's family (re-correcting interpolated members) to the exact
    periodic orbit and phase that best explain the observations.

    Returns (orbit, tau, chi2, x_epoch). Used when the data cannot determine the
    state, so that a family is judged by whether *any* of its members fits, not by
    how close the catalogue's sampling happens to fall.
    """
    obs = [o for o in observations if o.z is not None]
    epoch = min(o.t for o in obs) if epoch is None else epoch
    fam_idx = [i for i in range(len(catalog)) if catalog.fam_of[i] == catalog.fam_of[m]]
    k = fam_idx.index(m)
    orbit = catalog.members[m].orbit()
    tau, best = _obs_chi2_on(orbit, obs, epoch)
    for _ in range(max_moves):
        mem = catalog.members[fam_idx[k]]
        step = None
        for nb in (k - 1, k + 1):
            if not 0 <= nb < len(fam_idx):
                continue
            other = catalog.members[fam_idx[nb]]
            cache = {}

            def cost(u, other=other, cache=cache, mem=mem):
                o = _correct_between(mem, other, u)
                if o is None:
                    return np.inf
                t, c = _obs_chi2_on(o, obs, epoch, n=600)
                cache[u] = (o, t, c)
                return c

            r = minimize_scalar(cost, bounds=(0.0, 1.0), method="bounded",
                                options={"xatol": 2e-3, "maxiter": 14})
            if r.x in cache and cache[r.x][2] < best:
                o, _, _ = cache[r.x]
                t, c = _obs_chi2_on(o, obs, epoch)
                if c < best:
                    orbit, tau, best = o, t, c
                    step = nb if r.x > 0.98 else None
        if step is None:
            break
        k = step  # optimum sits at the neighbour: keep walking along the family
    return orbit, tau, best, orbit.state_at(tau)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

@dataclass
class ODReport:
    hypotheses: list  # list[ODResult], best first, with .weight and .match
    epoch: float
    n_meas: int
    determined: bool  # True if the data alone fixes all six state components
    system: System = EARTH_MOON

    @property
    def best(self) -> ODResult:
        return self.hypotheses[0]

    def table(self) -> str:
        s = self.system
        lines = [f"{len(self.hypotheses)} hypotheses from {self.n_meas} scalar measurements "
                 f"({'data determine the orbit' if self.determined else 'under-determined: catalogue prior used'})"]
        ruled = []
        for k, h in enumerate(self.hypotheses):
            if h.weight < 1e-3:
                ruled.append(h)
                continue
            lines.append(f" {k + 1}. weight {h.weight:5.3f}  chi2 {h.chi2:9.2f} (+prior {h.chi2_prior:6.2f})  "
                         f"sigma {h.sigma_pos_km():9.1f} km / {h.sigma_vel_mps():8.2f} m/s  [{h.source}]")
            if h.match is not None:
                lines.append(f"      nearest periodic orbit: {h.match.describe()}")
            lines.append(f"      Moon range at epoch {s.km(np.linalg.norm(h.x[:3] - s.pos2)):,.0f} km, "
                         f"Jacobi {cr3bp.jacobi(h.x, s.mu):.5f}")
        for h in ruled:
            lines.append(f"    ruled out (weight {h.weight:.1e}, chi2 + prior {h.score:,.1f}): {h.source}")
        return "\n".join(lines)

    def plausible(self, min_weight: float = 1e-3) -> list:
        """Hypotheses that the data have not ruled out."""
        return [h for h in self.hypotheses if h.weight >= min_weight]


def _same(a: ODResult, b: ODResult) -> bool:
    d = a.x - b.x
    try:
        m2 = float(d @ np.linalg.solve(a.P + b.P, d))
    except np.linalg.LinAlgError:
        m2 = np.inf
    return m2 < 16.0 or np.linalg.norm(d) < 1e-9


def determine_orbit(observations, catalog: OrbitCatalog | None = None, epoch: float | None = None,
                    n_hypotheses: int = 6, prior_pos_km: float = 2000.0, prior_vel_mps: float = 50.0,
                    refine_matches: int = 2, refine_catalog: bool = True,
                    system: System = EARTH_MOON) -> ODReport:
    """Estimate the target's orbit from observations, ranking alternative hypotheses.

    Hypotheses come from direct IOD and (if a catalogue is given) from the
    catalogue states that best explain the data. Each is refined by :func:`fit`;
    when the data cannot determine the full state, a prior centred on the
    hypothesis (prior_pos_km, prior_vel_mps) closes the problem. Duplicates are
    merged and weights are proportional to exp(-score/2). With refine_catalog, an
    under-determined catalogue hypothesis first slides along its family to the exact
    member that best fits the data (:func:`refine_hypothesis`), so families are
    compared on physics rather than on catalogue sampling.
    """
    obs = sorted([o for o in observations if o.z is not None], key=lambda o: o.t)
    if not obs:
        raise ValueError("no observations")
    epoch = obs[0].t if epoch is None else epoch
    psig = _prior_scale(system, prior_pos_km, prior_vel_mps)
    guesses = [(lab, x, False) for lab, x in iod_guesses(obs, epoch, system)]
    if catalog is not None:
        for m, tau, _ in catalog.match(obs, epoch, top=n_hypotheses):
            x = catalog.states(m, tau)
            lab = f"catalogue: {catalog.members[m].describe()}"
            try:
                under = info_rank(obs, x, epoch, system, prior_pos_km, prior_vel_mps) < 6
            except (cr3bp.ImpactError, RuntimeError):
                continue
            if under and refine_catalog:
                orbit, tau_r, _, x = refine_hypothesis(catalog, m, obs, epoch)
                lab = (f"family search: {orbit.family}, T = {orbit.period_days:.2f} d "
                       f"(from catalogue member {m})")
            guesses.append((lab, x, True))
    if not guesses:
        raise ValueError("the observations do not determine a state on their own (need e.g. two "
                         "position fixes, or position + velocity); pass an OrbitCatalog for hypotheses")
    results, determined = [], False
    for lab, x, _from_cat in guesses:
        try:
            rank = info_rank(obs, x, epoch, system, prior_pos_km, prior_vel_mps)
        except (cr3bp.ImpactError, RuntimeError):
            continue
        prior = None if rank >= 6 else (x.copy(), psig)
        try:
            r = fit(obs, x, epoch, system, prior=prior, source=lab)
        except RuntimeError:
            continue
        if prior is None and r.info_rank < 6:
            try:
                r = fit(obs, x, epoch, system, prior=(x.copy(), psig), source=lab)
            except RuntimeError:
                continue
        determined = determined or not r.prior_used
        results.append(r)
    if not results:
        raise RuntimeError("no hypothesis could be fitted")
    results.sort(key=lambda r: r.score)
    # drop fits that went nowhere (e.g. a Lambert seed on the wrong branch)
    results = [r for r in results if r is results[0] or r.score < max(1e4 * results[0].score, 1e4)]
    kept = []
    for r in results:
        if not any(_same(r, k) for k in kept):
            kept.append(r)
    kept = kept[:n_hypotheses]
    s0 = kept[0].score
    w = np.array([np.exp(-0.5 * (r.score - s0)) for r in kept])
    for r, wi in zip(kept, w / w.sum()):
        r.weight = float(wi)
    if catalog is not None:
        for k, r in enumerate(kept):
            r.match = nearest_periodic(r.x, catalog, refine=k < refine_matches)
    return ODReport(kept, epoch, sum(o.size for o in obs), determined, system)


def assess_fate(result: ODResult, n_samples: int = 64, horizon_days: float = 90.0, rng=0,
                gate_margin_km: float = manifolds.GATE_MARGIN_KM) -> dict:
    """Propagate samples of the fitted covariance and tally their fates
    (captured / impact Moon or Earth / exit through L1 or L2) within the horizon.
    Returns probabilities, median time to the event [days] and the samples' fates."""
    s = result.system
    H = s.nd_time_days(horizon_days)
    X = result.samples(n_samples, rng)
    fates = [manifolds.fate(x, H, s, gate_margin_km=gate_margin_km)[0] for x in X]
    out = {"n": n_samples, "horizon_days": horizon_days, "fates": fates, "p": {}, "median_days": {}}
    for k in manifolds.FATES:
        ts = [s.days(f.t) for f in fates if f.kind == k]
        out["p"][k] = len(ts) / n_samples
        if ts:
            out["median_days"][k] = float(np.median(ts))
    return out


def position_from_km(r_km, center: str = "earth", system: System = EARTH_MOON) -> np.ndarray:
    """Helper: a rotating-frame position given in km relative to ``center``."""
    return frames.center_position(system, center) + system.nd_length(np.asarray(r_km, float))


__all__ = [
    "Observer", "Observation", "obs_position", "obs_velocity", "obs_angles", "obs_range",
    "obs_range_rate", "obs_velocity_direction", "obs_angle_rates", "simulate", "fit", "info_rank",
    "iod_guesses", "OrbitCatalog", "CatalogMember", "OrbitMatch", "nearest_periodic", "ODResult",
    "ODReport", "determine_orbit", "refine_hypothesis", "assess_fate", "position_from_km", "SECONDS_PER_DAY",
]
