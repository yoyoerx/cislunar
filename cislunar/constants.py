"""Physical constants and CR3BP system definitions.

All CR3BP computations are nondimensional: length unit = primary separation (L*),
time unit = 1/mean motion (T*), so the primaries' angular rate is 1 and the
synodic period is 2*pi.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GM_EARTH = 398600.435436  # km^3/s^2 (DE440)
GM_MOON = 4902.800066  # km^3/s^2 (DE440)
R_EARTH = 6378.137  # km
R_MOON = 1737.4  # km
EARTH_MOON_DISTANCE = 384400.0  # km (mean)
SYNODIC_MONTH_DAYS = 29.530589
SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class System:
    """A CR3BP system: two primaries in circular orbit about their barycentre."""

    name: str
    gm1: float  # km^3/s^2, larger primary
    gm2: float  # km^3/s^2, smaller primary
    lstar: float  # km, primary separation
    r1: float  # km, radius of larger primary
    r2: float  # km, radius of smaller primary
    name1: str = "Earth"
    name2: str = "Moon"

    @property
    def mu(self) -> float:
        return self.gm2 / (self.gm1 + self.gm2)

    @property
    def tstar(self) -> float:
        """Characteristic time [s]."""
        return float(np.sqrt(self.lstar**3 / (self.gm1 + self.gm2)))

    @property
    def vstar(self) -> float:
        """Characteristic velocity [km/s]."""
        return self.lstar / self.tstar

    @property
    def pos1(self) -> np.ndarray:
        """Position of the larger primary in the rotating frame (nondim)."""
        return np.array([-self.mu, 0.0, 0.0])

    @property
    def pos2(self) -> np.ndarray:
        """Position of the smaller primary in the rotating frame (nondim)."""
        return np.array([1.0 - self.mu, 0.0, 0.0])

    # --- unit conversions -------------------------------------------------
    def km(self, nd):
        return np.asarray(nd) * self.lstar

    def kms(self, nd):
        return np.asarray(nd) * self.vstar

    def mps(self, nd):
        return np.asarray(nd) * self.vstar * 1000.0

    def days(self, nd):
        return np.asarray(nd) * self.tstar / SECONDS_PER_DAY

    def seconds(self, nd):
        return np.asarray(nd) * self.tstar

    def nd_length(self, km):
        return np.asarray(km) / self.lstar

    def nd_velocity(self, kms):
        return np.asarray(kms) / self.vstar

    def nd_time_days(self, days):
        return np.asarray(days) * SECONDS_PER_DAY / self.tstar

    def nd_time_seconds(self, s):
        return np.asarray(s) / self.tstar

    def state_to_dim(self, x):
        """Nondim state (..., 6) -> km, km/s."""
        x = np.asarray(x, dtype=float)
        out = x.copy()
        out[..., :3] *= self.lstar
        out[..., 3:6] *= self.vstar
        return out

    def state_to_nd(self, x):
        """km, km/s state (..., 6) -> nondim."""
        x = np.asarray(x, dtype=float)
        out = x.copy()
        out[..., :3] /= self.lstar
        out[..., 3:6] /= self.vstar
        return out


EARTH_MOON = System(
    name="Earth-Moon",
    gm1=GM_EARTH,
    gm2=GM_MOON,
    lstar=EARTH_MOON_DISTANCE,
    r1=R_EARTH,
    r2=R_MOON,
)
