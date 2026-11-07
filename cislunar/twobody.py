"""Two-body (Keplerian) utilities. Units are whatever mu is in (km & km^3/s^2,
or nondimensional). Angles in radians."""
from __future__ import annotations

import numpy as np


def stumpff_c(z):
    if z > 1e-6:
        return (1.0 - np.cos(np.sqrt(z))) / z
    if z < -1e-6:
        return (np.cosh(np.sqrt(-z)) - 1.0) / (-z)
    return 0.5 - z / 24.0 + z * z / 720.0


def stumpff_s(z):
    if z > 1e-6:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / sz**3
    if z < -1e-6:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / sz**3
    return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0


def _rot1(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot3(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def coe2rv(a, e, i, raan, argp, nu, mu):
    """Classical orbital elements -> position, velocity (inertial)."""
    p = a * (1.0 - e * e)
    r_pf = p / (1.0 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    Q = _rot3(raan) @ _rot1(i) @ _rot3(argp)
    return Q @ r_pf, Q @ v_pf


def rv2coe(r, v, mu) -> dict:
    """Position, velocity -> classical orbital elements (dict)."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    n = np.cross([0.0, 0.0, 1.0], h)
    nn = np.linalg.norm(n)
    evec = ((vn**2 - mu / rn) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(evec)
    energy = vn**2 / 2 - mu / rn
    a = -mu / (2 * energy) if abs(e - 1.0) > 1e-12 else np.inf
    i = np.arccos(np.clip(h[2] / hn, -1, 1))
    raan = np.arctan2(n[1], n[0]) % (2 * np.pi) if nn > 1e-12 else 0.0
    if e > 1e-10:
        if nn > 1e-12:
            argp = np.arccos(np.clip(np.dot(n, evec) / (nn * e), -1, 1))
            if evec[2] < 0:
                argp = 2 * np.pi - argp
        else:
            argp = np.arctan2(evec[1], evec[0]) % (2 * np.pi)
            if h[2] < 0:
                argp = (2 * np.pi - argp) % (2 * np.pi)
        nu = np.arccos(np.clip(np.dot(evec, r) / (e * rn), -1, 1))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu
    else:
        argp = 0.0
        ref = n / nn if nn > 1e-12 else np.array([1.0, 0.0, 0.0])
        nu = np.arccos(np.clip(np.dot(ref, r) / rn, -1, 1))
        if np.dot(np.cross(ref, r), h) < 0:
            nu = 2 * np.pi - nu
    return {"a": a, "e": e, "i": i, "raan": raan, "argp": argp, "nu": nu,
            "h": hn, "energy": energy}


def kepler_propagate(r0, v0, dt, mu, tol=1e-12, max_iter=100):
    """Universal-variable Kepler propagation (any conic). Returns r, v."""
    r0 = np.asarray(r0, float)
    v0 = np.asarray(v0, float)
    r0n = np.linalg.norm(r0)
    vr0 = np.dot(r0, v0) / r0n
    alpha = 2.0 / r0n - np.dot(v0, v0) / mu
    smu = np.sqrt(mu)
    chi = smu * abs(alpha) * dt if abs(alpha) > 1e-12 else smu * dt / r0n
    for _ in range(max_iter):
        z = alpha * chi**2
        C, S = stumpff_c(z), stumpff_s(z)
        F = (r0n * vr0 / smu * chi**2 * C + (1 - alpha * r0n) * chi**3 * S
             + r0n * chi - smu * dt)
        dF = r0n * vr0 / smu * chi * (1 - z * S) + (1 - alpha * r0n) * chi**2 * C + r0n
        d = F / dF
        chi -= d
        if abs(d) < tol * max(1.0, abs(chi)):
            break
    z = alpha * chi**2
    C, S = stumpff_c(z), stumpff_s(z)
    f = 1 - chi**2 / r0n * C
    g = dt - chi**3 / smu * S
    r = f * r0 + g * v0
    rn = np.linalg.norm(r)
    fdot = smu / (rn * r0n) * (alpha * chi**3 * S - chi)
    gdot = 1 - chi**2 / rn * C
    v = fdot * r0 + gdot * v0
    return r, v


def circular_orbit(radius, mu, inc=0.0, raan=0.0, u=0.0):
    """State on a circular orbit; u = argument of latitude."""
    return coe2rv(radius, 0.0, inc, raan, 0.0, u, mu)
