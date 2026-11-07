"""Single-revolution Lambert solver (universal variables, Bate-Mueller-White /
Curtis formulation, solved with a bracketing root finder)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .twobody import stumpff_c, stumpff_s


def lambert(r1, r2, tof, mu, prograde: bool = True, normal=None):
    """Solve Lambert's problem for the single-revolution transfer.

    r1, r2 : position vectors; tof > 0 : time of flight; mu : GM.
    The transfer direction is prograde with respect to `normal` (default +z)
    unless prograde=False.
    Returns (v1, v2).
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    if tof <= 0:
        raise ValueError("tof must be positive")
    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
    normal = np.array([0.0, 0.0, 1.0]) if normal is None else np.asarray(normal, float)
    cross = np.cross(r1, r2)
    cos_dth = np.clip(np.dot(r1, r2) / (n1 * n2), -1.0, 1.0)
    dth = np.arccos(cos_dth)
    short_way = np.dot(cross, normal) >= 0
    if not prograde:
        short_way = not short_way
    if not short_way:
        dth = 2 * np.pi - dth
    if abs(np.sin(dth)) < 1e-10:
        raise ValueError("transfer angle is 0 or 180 deg; Lambert plane undefined")
    A = np.sin(dth) * np.sqrt(n1 * n2 / (1.0 - cos_dth))
    smu = np.sqrt(mu)

    def y(z):
        return n1 + n2 + A * (z * stumpff_s(z) - 1.0) / np.sqrt(stumpff_c(z))

    def F(z):
        yz = y(z)
        if yz < 0:  # outside the valid domain: on the "too short" side
            return -smu * tof
        return (yz / stumpff_c(z)) ** 1.5 * stumpff_s(z) + A * np.sqrt(yz) - smu * tof

    # F is increasing in z on the valid domain (y > 0). Bracket the root.
    z_hi = (2 * np.pi) ** 2 - 1e-6
    z_lo = -4.0
    while y(z_lo) > 0 and F(z_lo) > 0:
        z_lo *= 2.0
        if z_lo < -1e8:
            raise RuntimeError("Lambert: no hyperbolic bracket")
    if y(z_lo) <= 0:
        # move up to just inside the y > 0 boundary, where F < 0
        a, b = z_lo, z_hi
        for _ in range(200):
            m = 0.5 * (a + b)
            if y(m) > 0:
                b = m
            else:
                a = m
        z_lo = b
    if F(z_hi) < 0:
        raise RuntimeError("Lambert: tof too long for a single-revolution solution")
    z = brentq(F, z_lo, z_hi, xtol=1e-14, maxiter=500)
    yz = y(z)
    f = 1 - yz / n1
    g = A * np.sqrt(yz / mu)
    gdot = 1 - yz / n2
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    return v1, v2

