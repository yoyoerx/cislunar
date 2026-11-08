"""Real-world dynamics and reference frames for working with actual observations.

The rest of the toolkit is CR3BP-only. Fitting real astrometry needs the real
Moon and Sun (see architecture.md, ADR-27/28):

* :class:`Ephemeris` - JPL Horizons geocentric ICRF vector tables, fetched once,
  cached as JSON and Hermite-interpolated.
* time scales: UTC <-> TT/TDB Julian dates (leap seconds 1972-2026).
* Earth orientation: GMST/GAST, IAU-1976 precession, leading IAU-1980 nutation
  terms -> :func:`station_j2000` for MPC observatory parallax constants.
* :func:`selenographic` - IAU (WGCCRE 2009) lunar rotation model.
* :class:`NBodyDynamics` - Earth-centred J2000 point-mass Earth + Moon + Sun
  (+ Earth J2, optional planets and solar radiation pressure) with
  state-transition matrix, Moon/Earth impact events and a close-approach finder
  for long (decades) propagations.
* :class:`CR3BPInFrame` - the CR3BP aligned with the real Moon at an epoch, but
  exposing states in the same Earth-centred J2000 frame, so a CR3BP fit and an
  ephemeris fit of the same data can be compared like for like.

Both dynamics classes use the CR3BP nondimensional units (L*, T* of the System)
and measure time t from a reference TDB Julian date ``jd0``; they plug into
:func:`cislunar.od.fit` through its ``dynamics`` argument.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from scipy.integrate import solve_ivp

from . import cr3bp, frames
from .constants import EARTH_MOON, R_EARTH, R_MOON, SECONDS_PER_DAY, System

GM_SUN = 1.32712440041e11  # km^3/s^2 (DE440)
J2_EARTH = 1.08262668e-3
C_KMS = 299792.458
AU_KM = 149597870.7
SOLAR_PRESSURE_1AU = 4.56e-6  # N/m^2 (solar flux / c) at 1 AU
OMEGA_EARTH = 7.292115855e-5  # rad/s
JD_J2000 = 2451545.0
ARCSEC = np.pi / 648000.0
HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
# (first UTC JD the offset applies, TAI-UTC seconds), 1972 Jan 1 onward (IERS Bulletin C)
_LEAP = [(2441317.5, 10.0), (2441499.5, 11.0), (2441683.5, 12.0), (2442048.5, 13.0), (2442413.5, 14.0),
         (2442778.5, 15.0), (2443144.5, 16.0), (2443509.5, 17.0), (2443874.5, 18.0), (2444239.5, 19.0),
         (2444786.5, 20.0), (2445151.5, 21.0), (2445516.5, 22.0), (2446247.5, 23.0), (2447161.5, 24.0),
         (2447892.5, 25.0), (2448257.5, 26.0), (2448804.5, 27.0), (2449169.5, 28.0), (2449534.5, 29.0),
         (2450083.5, 30.0), (2450630.5, 31.0), (2451179.5, 32.0), (2453736.5, 33.0), (2454832.5, 34.0),
         (2456109.5, 35.0), (2457204.5, 36.0), (2457754.5, 37.0)]
# planetary GM [km^3/s^2] (DE440; system values for planets with moons) and Horizons IDs
PLANETS = {"mercury": (22031.868551, "199"), "venus": (324858.592, "299"), "mars": (42828.375816, "4"),
           "jupiter": (126712764.1, "5"), "saturn": (37940584.8418, "6"), "uranus": (5794556.4, "7"),
           "neptune": (6836527.10058, "8")}


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def jd_from_calendar(year: int, month: int, day: float) -> float:
    """Julian date of a Gregorian calendar date (day may be fractional)."""
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    return float(int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5)


def tai_minus_utc(jd_utc: float) -> float:
    """TAI - UTC [s]. Before 1972 UTC used variable-length seconds; there the 1972
    value (10 s) is returned, good to a few seconds, which is ample for dating
    events decades back but not for reducing pre-1972 astrometry."""
    if jd_utc < _LEAP[0][0]:
        return _LEAP[0][1]
    return [s for j, s in _LEAP if jd_utc >= j][-1]


def utc_to_tdb(jd_utc: float) -> float:
    """UTC Julian date -> TDB Julian date (TDB - TT periodic term included, ~1.7 ms)."""
    jd_tt = jd_utc + (tai_minus_utc(jd_utc) + 32.184) / SECONDS_PER_DAY
    g = np.radians(357.53 + 0.9856003 * (jd_tt - JD_J2000))
    return float(jd_tt + (0.001657 * np.sin(g) + 0.000014 * np.sin(2 * g)) / SECONDS_PER_DAY)


def tdb_to_utc(jd_tdb: float) -> float:
    jd = jd_tdb - 69.184 / SECONDS_PER_DAY
    for _ in range(3):
        jd += jd_tdb - utc_to_tdb(jd)
    return float(jd)


# ---------------------------------------------------------------------------
# Rotations and Earth orientation
# ---------------------------------------------------------------------------

def _r1(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def _r2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def _r3(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def precession_matrix(jd_tt: float) -> np.ndarray:
    """IAU-1976 precession: J2000 mean equator -> mean equator of date (coordinate transform)."""
    t = (jd_tt - JD_J2000) / 36525.0
    zeta = (2306.2181 * t + 0.30188 * t**2 + 0.017998 * t**3) * ARCSEC
    z = (2306.2181 * t + 1.09468 * t**2 + 0.018203 * t**3) * ARCSEC
    theta = (2004.3109 * t - 0.42665 * t**2 - 0.041833 * t**3) * ARCSEC
    return _r3(-z) @ _r2(theta) @ _r3(-zeta)


def nutation(jd_tt: float):
    """Leading IAU-1980 nutation terms: (dpsi, deps, mean obliquity) [rad]; error < 0.5"."""
    t = (jd_tt - JD_J2000) / 36525.0
    om = np.radians(125.04452 - 1934.136261 * t)
    ls = np.radians(280.4665 + 36000.7698 * t)
    lm = np.radians(218.3165 + 481267.8813 * t)
    dpsi = (-17.20 * np.sin(om) - 1.32 * np.sin(2 * ls) - 0.23 * np.sin(2 * lm) + 0.21 * np.sin(2 * om)) * ARCSEC
    deps = (9.20 * np.cos(om) + 0.57 * np.cos(2 * ls) + 0.10 * np.cos(2 * lm) - 0.09 * np.cos(2 * om)) * ARCSEC
    eps0 = (84381.448 - 46.8150 * t - 0.00059 * t**2 + 0.001813 * t**3) * ARCSEC
    return dpsi, deps, eps0


def nutation_matrix(jd_tt: float) -> np.ndarray:
    """Mean equator of date -> true equator of date."""
    dpsi, deps, eps0 = nutation(jd_tt)
    return _r1(-(eps0 + deps)) @ _r3(-dpsi) @ _r1(eps0)


def gmst(jd_ut1: float) -> float:
    """Greenwich mean sidereal time [rad] (IAU 1982)."""
    tu = (jd_ut1 - JD_J2000) / 36525.0
    s = 67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * tu + 0.093104 * tu**2 - 6.2e-6 * tu**3
    return float(np.mod(s % 86400.0 / 240.0, 360.0) * np.pi / 180.0)


def gast(jd_ut1: float, jd_tt: float) -> float:
    dpsi, deps, eps0 = nutation(jd_tt)
    return gmst(jd_ut1) + dpsi * np.cos(eps0 + deps)


def station_j2000(east_lon_deg: float, rho_cos: float, rho_sin: float, jd_utc: float):
    """Position [km] and velocity [km/s] of an MPC observatory in the J2000 (mean
    equator, ~GCRS) frame. UT1 is taken as UTC and polar motion is ignored."""
    lam = np.radians(east_lon_deg)
    r_itrf = R_EARTH * np.array([rho_cos * np.cos(lam), rho_cos * np.sin(lam), rho_sin])
    jd_tt = jd_utc + (tai_minus_utc(jd_utc) + 32.184) / SECONDS_PER_DAY
    th = gast(jd_utc, jd_tt)
    c, s = np.cos(th), np.sin(th)
    r_tod = np.array([c * r_itrf[0] - s * r_itrf[1], s * r_itrf[0] + c * r_itrf[1], r_itrf[2]])
    v_tod = OMEGA_EARTH * np.array([-r_tod[1], r_tod[0], 0.0])
    m = (nutation_matrix(jd_tt) @ precession_matrix(jd_tt)).T  # true of date -> J2000
    return m @ r_tod, m @ v_tod


# ---------------------------------------------------------------------------
# Moon orientation
# ---------------------------------------------------------------------------

def moon_orientation(jd_tdb: float) -> np.ndarray:
    """Rotation J2000 -> Moon body-fixed (IAU WGCCRE 2009 model, approximates the
    mean-Earth/polar-axis frame to ~100 m)."""
    d = jd_tdb - JD_J2000
    t = d / 36525.0
    e = np.radians([125.045 - 0.0529921 * d, 250.089 - 0.1059842 * d, 260.008 + 13.0120009 * d,
                    176.625 + 13.3407154 * d, 357.529 + 0.9856003 * d, 311.589 + 26.4057084 * d,
                    134.963 + 13.0649930 * d, 276.617 + 0.3287146 * d, 34.226 + 1.7484877 * d,
                    15.134 - 0.1589763 * d, 119.743 + 0.0036096 * d, 239.961 + 0.1643573 * d,
                    25.053 + 12.9590088 * d])
    s, c = np.sin(e), np.cos(e)
    ra = (269.9949 + 0.0031 * t - 3.8787 * s[0] - 0.1204 * s[1] + 0.0700 * s[2] - 0.0172 * s[3]
          + 0.0072 * s[5] - 0.0052 * s[9] + 0.0043 * s[12])
    dec = (66.5392 + 0.0130 * t + 1.5419 * c[0] + 0.0239 * c[1] - 0.0278 * c[2] + 0.0068 * c[3]
           - 0.0029 * c[5] + 0.0009 * c[6] + 0.0008 * c[9] - 0.0009 * c[12])
    w = (38.3213 + 13.17635815 * d - 1.4e-12 * d**2 + 3.5610 * s[0] + 0.1208 * s[1] - 0.0642 * s[2]
         + 0.0158 * s[3] + 0.0252 * s[4] - 0.0066 * s[5] - 0.0047 * s[6] - 0.0046 * s[7] + 0.0028 * s[8]
         + 0.0052 * s[9] + 0.0040 * s[10] + 0.0019 * s[11] - 0.0044 * s[12])
    return _r3(np.radians(w)) @ _r1(np.radians(90.0 - dec)) @ _r3(np.radians(90.0 + ra))


def selenographic(r_rel_moon_km, jd_tdb: float):
    """(latitude, east longitude in [0, 360), radius) [deg, deg, km] of a Moon-centred
    J2000 position."""
    b = moon_orientation(jd_tdb) @ np.asarray(r_rel_moon_km, float)
    r = np.linalg.norm(b)
    return (float(np.degrees(np.arcsin(b[2] / r))), float(np.degrees(np.arctan2(b[1], b[0])) % 360.0),
            float(r))


def surface_distance_km(lat1, lon1, lat2, lon2, radius_km: float = R_MOON) -> float:
    p1, p2 = np.radians([lat1, lat2])
    dl = np.radians(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return float(radius_km * np.arccos(np.clip(c, -1.0, 1.0)))


# ---------------------------------------------------------------------------
# Horizons tables
# ---------------------------------------------------------------------------

def horizons_vectors(command: str, start: str, stop: str, step: str = "1h", center: str = "500@399",
                     timeout: float = 60.0):
    """Fetch an ICRF (equatorial) vector table from JPL Horizons.

    Returns (jd_tdb, r_km, v_kms) arrays.
    """
    q = dict(format="json", COMMAND=f"'{command}'", EPHEM_TYPE="VECTORS", CENTER=f"'{center}'",
             START_TIME=f"'{start}'", STOP_TIME=f"'{stop}'", STEP_SIZE=f"'{step}'", VEC_TABLE="'2'",
             REF_SYSTEM="'ICRF'", REF_PLANE="'FRAME'", CSV_FORMAT="'YES'", OUT_UNITS="'KM-S'",
             VEC_LABELS="'NO'", MAKE_EPHEM="'YES'", OBJ_DATA="'NO'")
    req = Request(f"{HORIZONS_URL}?{urlencode(q)}", headers={"User-Agent": "cislunar-toolkit"})
    with urlopen(req, timeout=timeout) as f:
        res = json.loads(f.read().decode("utf-8"))["result"]
    if "$$SOE" not in res:
        raise RuntimeError(f"Horizons returned no table:\n{res[:800]}")
    body = res.split("$$SOE")[1].split("$$EOE")[0]
    rows = [ln.split(",") for ln in body.strip().splitlines() if ln.strip()]
    a = np.array([[float(x) for x in (r[0], *r[2:8])] for r in rows])
    return a[:, 0], a[:, 1:4], a[:, 4:7]


class Ephemeris:
    """Tabulated geocentric J2000 states of perturbing bodies, Hermite-interpolated."""

    BODIES = {"moon": "301", "sun": "10"}

    def __init__(self, tables: dict, source: str = ""):
        self.tables = {k: (np.asarray(j, float), np.asarray(r, float), np.asarray(v, float))
                       for k, (j, r, v) in tables.items()}
        self.source = source

    @classmethod
    def fetch(cls, start: str, stop: str, step: str = "1h", bodies=None) -> "Ephemeris":
        bodies = bodies or cls.BODIES
        return cls({name: horizons_vectors(cmd, start, stop, step) for name, cmd in bodies.items()},
                   source=f"JPL Horizons {start} to {stop} step {step}, geocentric ICRF, TDB")

    def save(self, path) -> None:
        """Write as JSON (gzip-compressed if the name ends in .gz)."""
        d = {"source": self.source,
             "tables": {k: {"jd_tdb": j.tolist(), "r_km": r.tolist(), "v_kms": v.tolist()}
                        for k, (j, r, v) in self.tables.items()}}
        text = json.dumps(d)
        if str(path).endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(text)
        else:
            Path(path).write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Ephemeris":
        if str(path).endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                d = json.loads(f.read())
        else:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({k: (v["jd_tdb"], v["r_km"], v["v_kms"]) for k, v in d["tables"].items()}, d["source"])

    def merged(self, other: "Ephemeris") -> "Ephemeris":
        """A new Ephemeris holding this one's tables plus ``other``'s (other wins on clashes)."""
        return Ephemeris({**self.tables, **other.tables}, f"{self.source}; {other.source}")

    @classmethod
    def cached(cls, path, start: str, stop: str, step: str = "1h", bodies=None) -> "Ephemeris":
        """Load from ``path`` if present, else fetch from Horizons and save there.

        ``bodies`` maps table names to Horizons IDs (default Moon and Sun; planets
        are in :data:`PLANETS`, e.g. ``{"venus": "299"}``)."""
        p = Path(path)
        if p.exists():
            return cls.load(p)
        eph = cls.fetch(start, stop, step, bodies)
        p.parent.mkdir(parents=True, exist_ok=True)
        eph.save(p)
        return eph

    def span(self):
        j = np.concatenate([t[0] for t in self.tables.values()])
        return float(max(t[0][0] for t in self.tables.values())), float(min(t[0][-1] for t in self.tables.values()))

    def state(self, body: str, jd_tdb):
        """Position [km] and velocity [km/s] at TDB Julian date(s) (cubic Hermite)."""
        j, r, v = self.tables[body]
        jd = np.atleast_1d(np.asarray(jd_tdb, float))
        if jd.min() < j[0] - 1e-9 or jd.max() > j[-1] + 1e-9:
            raise ValueError(f"{body}: JD {jd.min():.3f}-{jd.max():.3f} outside the table "
                             f"{j[0]:.3f}-{j[-1]:.3f}")
        k = np.clip(np.searchsorted(j, jd) - 1, 0, len(j) - 2)
        h = (j[k + 1] - j[k]) * SECONDS_PER_DAY
        s = ((jd - j[k]) * SECONDS_PER_DAY / h)[:, None]
        hh = h[:, None]
        p0, p1, v0, v1 = r[k], r[k + 1], v[k], v[k + 1]
        s2, s3 = s * s, s * s * s
        pos = ((2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * hh * v0
               + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * hh * v1)
        vel = ((6 * s2 - 6 * s) * p0 / hh + (3 * s2 - 4 * s + 1) * v0
               + (-6 * s2 + 6 * s) * p1 / hh + (3 * s2 - 2 * s) * v1)
        if np.ndim(jd_tdb) == 0:
            return pos[0], vel[0]
        return pos, vel


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def _propagate_generic(rhs, x0, t0, times, with_stm, events, args=()):
    """States (and STMs) at ``times`` (nondim) from x0 at t0: list of (x, phi)."""
    x0 = np.asarray(x0, float)
    times = np.asarray(times, float)
    out = [None] * len(times)
    for i, t in enumerate(times):
        if t == t0:
            out[i] = (x0.copy(), np.eye(6))
    for sign in (1.0, -1.0):
        idx = [i for i, t in enumerate(times) if (t - t0) * sign > 0]
        if not idx:
            continue
        tt = np.unique(times[idx])
        tt = tt if sign > 0 else tt[::-1]
        y0 = np.concatenate((x0, np.eye(6).ravel())) if with_stm else x0
        sol = solve_ivp(rhs, (t0, tt[-1]), y0, method="DOP853", rtol=1e-11, atol=1e-13, t_eval=tt,
                        events=events, args=(with_stm,) + tuple(args))
        if sol.status == 1:
            raise cr3bp.ImpactError("trajectory impacts a body between observations")
        if not sol.success:
            raise RuntimeError(f"propagation failed: {sol.message}")
        lookup = {float(t): sol.y[:, k] for k, t in enumerate(sol.t)}
        for i in idx:
            y = lookup[float(times[i])]
            out[i] = (y[:6], y[6:].reshape(6, 6) if with_stm else None)
    return out


class NBodyDynamics:
    """Earth-centred J2000 dynamics: Earth (+J2), Moon and Sun point masses, and
    optional cannonball solar radiation pressure (``amr`` = C_r A/m in m^2/kg;
    no shadowing; left out of the STM partials).

    ``planets`` names extra point-mass perturbers from :data:`PLANETS` (their
    geocentric tables must be in ``eph``); needed for heliocentric arcs of years.
    The Earth-centred frame stays exact for distant objects: every perturber's
    pull on the Earth is subtracted (the indirect term). ``gr`` adds the Sun's
    leading general-relativity term (Schwarzschild, PPN beta = gamma = 1), worth
    ~1e4 km over a decade for an object near 1 AU.

    States are nondimensional (system L*, V*) J2000 Earth-centred; t is
    nondimensional time since TDB Julian date ``jd0``.
    """

    frame = "J2000 Earth-centred"

    def __init__(self, eph: Ephemeris, jd0: float, system: System = EARTH_MOON, sun: bool = True,
                 moon: bool = True, j2: bool = True, amr: float = 0.0, planets=(), gr: bool = False):
        self.eph, self.jd0, self.system = eph, float(jd0), system
        gm = system.gm1 + system.gm2
        self.mu_e = system.gm1 / gm
        self.bodies = []
        if moon:
            self.bodies.append(("moon", system.gm2 / gm))
        if sun:
            self.bodies.append(("sun", GM_SUN / gm))
        for name in planets:
            self.bodies.append((name, PLANETS[name][0] / gm))
        self.j2 = j2
        # solar radiation pressure: amr = C_r * area / mass [m^2/kg]; cannonball, no shadow
        self.amr = float(amr)
        self.gr = bool(gr)
        self._c2 = (C_KMS / system.vstar) ** 2
        self._mu_sun = GM_SUN / gm
        acc_1au = SOLAR_PRESSURE_1AU * self.amr / 1000.0  # km/s^2 at 1 AU
        self._srp = acc_1au / (system.lstar / system.tstar**2) * (AU_KM / system.lstar) ** 2
        self.re = R_EARTH / system.lstar
        self.rm = R_MOON / system.lstar

    # time helpers
    def jd(self, t) -> np.ndarray:
        return self.jd0 + np.asarray(self.system.days(t))

    def t_of_jd(self, jd_tdb):
        return self.system.nd_time_days(np.asarray(jd_tdb) - self.jd0)

    def body_state(self, name: str, t):
        r, v = self.eph.state(name, self.jd(t))
        return self.system.state_to_nd(np.concatenate((r, v), axis=-1))

    def _perturbers(self, t):
        return [(mu, self.body_state(name, t)[:3]) for name, mu in self.bodies]

    def accel(self, t, r, v=None):
        """Acceleration at position r (the relativity term needs the velocity v)."""
        rn = np.linalg.norm(r)
        a = -self.mu_e * r / rn**3
        if self.j2:
            z2 = (r[2] / rn) ** 2
            k = -1.5 * J2_EARTH * self.mu_e * self.re**2 / rn**5
            a = a + k * np.array([r[0] * (1 - 5 * z2), r[1] * (1 - 5 * z2), r[2] * (3 - 5 * z2)])
        for mu, rk in self._perturbers(t):
            d = rk - r
            a = a + mu * (d / np.linalg.norm(d) ** 3 - rk / np.linalg.norm(rk) ** 3)
        if self.amr:
            a = a + self._srp_accel(t, r)
        if self.gr and v is not None:
            a = a + self._gr_accel(t, r, v)
        return a

    def _gr_accel(self, t, r, v):
        """Sun's Schwarzschild term on the object minus the same term on the Earth
        (the Earth-centred frame accelerates with the Earth; leaving the Earth's
        ~1.8e-13 km/s^2 term out forces ~1-year orbits resonantly)."""
        sun = self.body_state("sun", t)
        mu, c2 = self._mu_sun, self._c2

        def term(rh, vh):
            rn = np.linalg.norm(rh)
            return mu / (c2 * rn**3) * ((4 * mu / rn - vh @ vh) * rh + 4 * (rh @ vh) * vh)
        return term(r - sun[:3], v - sun[3:]) - term(-sun[:3], -sun[3:])

    def _srp_accel(self, t, r):
        d = r - self.body_state("sun", t)[:3]
        dn = np.linalg.norm(d)
        return self._srp * d / dn**3

    def _rhs(self, t, y, with_stm):
        r, v = y[:3], y[3:6]
        rn = np.linalg.norm(r)
        a = -self.mu_e * r / rn**3
        if self.amr:
            a = a + self._srp_accel(t, r)
        if self.gr:
            a = a + self._gr_accel(t, r, v)
        if self.j2:
            z2 = (r[2] / rn) ** 2
            k = -1.5 * J2_EARTH * self.mu_e * self.re**2 / rn**5
            a = a + k * np.array([r[0] * (1 - 5 * z2), r[1] * (1 - 5 * z2), r[2] * (3 - 5 * z2)])
        pert = self._perturbers(t)
        for mu, rk in pert:
            d = rk - r
            a = a + mu * (d / np.linalg.norm(d) ** 3 - rk / np.linalg.norm(rk) ** 3)
        if not with_stm:
            return np.concatenate((v, a))
        eye = np.eye(3)
        G = -self.mu_e * (eye / rn**3 - 3.0 * np.outer(r, r) / rn**5)
        for mu, rk in pert:
            d = r - rk
            dn = np.linalg.norm(d)
            G = G - mu * (eye / dn**3 - 3.0 * np.outer(d, d) / dn**5)
        phi = y[6:].reshape(6, 6)
        dphi = np.vstack((phi[3:], G @ phi[:3]))
        return np.concatenate((v, a, dphi.ravel()))

    def _events(self, moon_impact=True, earth_impact=True, moon_radius_km=None):
        evs = []
        rm = self.rm if moon_radius_km is None else moon_radius_km / self.system.lstar

        if earth_impact:
            def hit_earth(t, y, *a):
                return np.linalg.norm(y[:3]) - self.re
            hit_earth.terminal, hit_earth.direction = True, -1
            evs.append(hit_earth)
        if moon_impact and any(n == "moon" for n, _ in self.bodies):
            def hit_moon(t, y, *a):
                return np.linalg.norm(y[:3] - self.body_state("moon", t)[:3]) - rm
            hit_moon.terminal, hit_moon.direction = True, -1
            evs.append(hit_moon)
        return evs

    def propagate_many(self, x0, t0, times, with_stm: bool = True):
        return _propagate_generic(self._rhs, x0, t0, times, with_stm, self._events())

    def propagate(self, x0, t0: float, t1: float, n_points: int = 2000, moon_radius_km: float | None = None):
        """Trajectory from t0 to t1 (stops at an impact).

        Returns (t, X, impact) with impact = None or a dict with the body, time t,
        TDB/UTC Julian date, state, and for the Moon the selenographic lat/lon.
        """
        evs = self._events(moon_radius_km=moon_radius_km)
        sol = solve_ivp(self._rhs, (t0, t1), np.asarray(x0, float), method="DOP853", rtol=1e-11,
                        atol=1e-13, args=(False,), events=evs, dense_output=True)
        impact = None
        t_end = t1
        if sol.status == 1:
            for ev, te, ye in zip(evs, sol.t_events, sol.y_events):
                if len(te):
                    t_end = float(te[0])
                    body = "moon" if ev.__name__ == "hit_moon" else "earth"
                    jd = float(self.jd(t_end))
                    impact = {"body": body, "t": t_end, "jd_tdb": jd, "jd_utc": tdb_to_utc(jd), "x": ye[0]}
                    if body == "moon":
                        rel = self.system.km(ye[0][:3] - self.body_state("moon", t_end)[:3])
                        lat, lon, _ = selenographic(rel, jd)
                        vrel = self.system.kms(ye[0][3:] - self.body_state("moon", t_end)[3:])
                        up = rel / np.linalg.norm(rel)
                        impact.update(lat=lat, lon=lon, speed_kms=float(np.linalg.norm(vrel)),
                                      angle_from_vertical_deg=float(np.degrees(np.arccos(
                                          -np.dot(vrel, up) / np.linalg.norm(vrel)))))
                    break
        t = np.linspace(t0, t_end, n_points)
        return t, sol.sol(t).T, impact


    def close_approaches(self, x0, t0: float, t1: float, r_max_km=5.0e6, body="earth",
                         rtol: float = 1e-10, max_step_days: float | None = None, times=None):
        """Propagate from t0 to t1 (either direction; no STM) and list every local
        minimum of the distance to ``body`` closer than ``r_max_km``.

        ``body`` is "earth", a table name such as "moon", or a tuple of them (then
        ``r_max_km`` may be a matching tuple). Returns (approaches, x1, impact):
        approaches is a list of dicts with body, t, jd_tdb, r_km and the state x
        at each minimum, in time order of the integration; x1 is the final state;
        impact is None or {"body", "jd_tdb"} if the run hit the Earth or Moon.
        With ``times`` (nondim, inside [t0, t1]) the second item is instead the
        array of states at those times (rows of NaN past an impact).
        """
        bodies = (body,) if isinstance(body, str) else tuple(body)
        limits = (r_max_km,) * len(bodies) if np.isscalar(r_max_km) else tuple(r_max_km)

        def rel(name, t, y):
            if name == "earth":
                return y[:3], y[3:6]
            b = self.body_state(name, t)
            return y[:3] - b[:3], y[3:6] - b[3:]

        def make(name):
            def rdot(t, y, *a):
                r, v = rel(name, t, y)
                return float(np.dot(r, v))
            # r.v crosses zero upward (in time) at a minimum; integrating backwards
            # the solver sees it crossing downward
            rdot.direction = 1.0 if t1 > t0 else -1.0
            return rdot
        evs = [make(n) for n in bodies]
        imp = self._events()
        kw = {}
        if max_step_days:
            kw["max_step"] = float(self.system.nd_time_days(max_step_days))
        if times is not None:
            times = np.asarray(times, float)
            order = np.argsort(times) if t1 > t0 else np.argsort(-times)
            kw["t_eval"] = times[order]
        sol = solve_ivp(self._rhs, (t0, t1), np.asarray(x0, float), method="DOP853", rtol=rtol,
                        atol=rtol * 1e-2, args=(False,), events=evs + imp, **kw)
        out = []
        for name, lim, tes, yes in zip(bodies, limits, sol.t_events, sol.y_events):
            for te, ye in zip(tes, yes):
                r = self.system.km(np.linalg.norm(rel(name, te, ye)[0]))
                if r < lim:
                    out.append({"body": name, "t": float(te), "jd_tdb": float(self.jd(te)), "r_km": float(r),
                                "x": ye.copy()})
        out.sort(key=lambda c: c["t"], reverse=t1 < t0)
        impact = None
        if sol.status == 1:
            for ev, te in zip(imp, sol.t_events[len(bodies):]):
                if len(te):
                    impact = {"body": ev.__name__.replace("hit_", ""), "jd_tdb": float(self.jd(te[0]))}
        if times is None:
            return out, sol.y[:, -1].copy(), impact
        X = np.full((len(times), 6), np.nan)
        Y = np.asarray(sol.y, float).reshape(6, -1)  # empty if an impact came before the first sample
        X[order[:Y.shape[1]]] = Y.T
        return out, X, impact


def shoot_velocity(dyn, r0, v_guess, t0: float, t1: float, r_target, tol_km: float = 0.1, max_iter: int = 30):
    """Velocity at (r0, t0) whose trajectory under ``dyn`` reaches r_target at t1
    (Newton iteration on the state-transition matrix; an N-body Lambert solver).

    r0, v_guess, r_target are nondimensional (dyn units). Returns (v0, miss_km).
    Use dynamics without the target body's gravity when aiming at a point near it.
    """
    r0 = np.asarray(r0, float)
    v = np.asarray(v_guess, float).copy()
    target = np.asarray(r_target, float)
    miss = np.inf
    for _ in range(max_iter):
        (x1, phi), = dyn.propagate_many(np.concatenate((r0, v)), t0, [t1], with_stm=True)
        d = x1[:3] - target
        miss = float(dyn.system.km(np.linalg.norm(d)))
        if miss < tol_km:
            break
        dv = np.linalg.solve(phi[:3, 3:], -d)
        step = np.linalg.norm(dv)
        lim = 0.2 * np.linalg.norm(v)
        v = v + (dv if step < lim else dv * lim / step)
    return v, miss


class CR3BPInFrame:
    """CR3BP dynamics expressed in Earth-centred J2000 (nondim), for comparison.

    The rotating frame is aligned with the real Moon at TDB Julian date ``jd0``
    (x toward the Moon, z along its orbital angular momentum); the CR3BP then
    assumes a circular Moon at the system's mean distance.
    """

    frame = "J2000 Earth-centred (CR3BP dynamics)"

    def __init__(self, eph: Ephemeris, jd0: float, system: System = EARTH_MOON):
        self.eph, self.jd0, self.system = eph, float(jd0), system
        r, v = eph.state("moon", jd0)
        x = r / np.linalg.norm(r)
        z = np.cross(r, v)
        z /= np.linalg.norm(z)
        self.A = np.column_stack((x, np.cross(z, x), z))  # EM-inertial axes -> J2000

    def jd(self, t):
        return self.jd0 + np.asarray(self.system.days(t))

    def t_of_jd(self, jd_tdb):
        return self.system.nd_time_days(np.asarray(jd_tdb) - self.jd0)

    def to_rotating(self, x, t):
        q = np.concatenate((self.A.T @ x[:3], self.A.T @ x[3:]))
        return frames.inertial_to_rotating(t, q, self.system, "earth")

    def from_rotating(self, x, t):
        q = frames.rotating_to_inertial(t, x, self.system, "earth")
        return np.concatenate((self.A @ q[:3], self.A @ q[3:]))

    def _T(self, t):
        b = self.to_rotating(np.zeros(6), t)
        return np.column_stack([self.to_rotating(e, t) - b for e in np.eye(6)])

    def propagate_many(self, x0, t0, times, with_stm: bool = True):
        from .od import _propagate_many
        xr0 = self.to_rotating(x0, t0)
        res = _propagate_many(xr0, t0, times, self.system, with_stm=with_stm)
        T0 = self._T(t0)
        out = []
        for t, (xr, phi) in zip(times, res):
            x = self.from_rotating(xr, t)
            out.append((x, np.linalg.solve(self._T(t), phi @ T0) if with_stm else None))
        return out


def station_observer(east_lon_deg: float, rho_cos: float, rho_sin: float, dynamics, name: str = ""):
    """An :class:`cislunar.od.Observer` for an MPC station, in the dynamics' J2000 frame."""
    from .od import Observer
    s = dynamics.system
    cache = {}

    def fn(t):
        if t not in cache:
            jd_utc = tdb_to_utc(float(dynamics.jd(t)))
            r, v = station_j2000(east_lon_deg, rho_cos, rho_sin, jd_utc)
            cache[t] = s.state_to_nd(np.concatenate((r, v)))
        return cache[t]

    return Observer(fn, name or f"station {east_lon_deg:.3f}")


__all__ = ["Ephemeris", "NBodyDynamics", "CR3BPInFrame", "horizons_vectors", "utc_to_tdb", "tdb_to_utc",
           "jd_from_calendar", "station_j2000", "station_observer", "selenographic", "moon_orientation",
           "surface_distance_km", "precession_matrix", "nutation_matrix", "gmst", "gast", "C_KMS", "GM_SUN"]
