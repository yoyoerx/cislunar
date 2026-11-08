"""Optical astrometry: parsing, observatories, and angles-only initial orbits.

* :func:`parse` reads MPC 80-column lines (``C2022 01 15.75603 hh mm ss.ss
  +dd mm ss.s``) and Find_Orb's high-precision form (``CK220117:185918
  ddd.ddddd +dd.ddddd``, time yymmdd:hhmmss[fraction], angles in degrees).
  Columns that ran together (6-decimal dates followed by RA) are handled.
* :func:`load_stations` reads MPC observatory parallax constants.
* :func:`to_observations` turns records into :class:`cislunar.od.Observation`
  angle measurements in a dynamics model's J2000 frame, with light-time and the
  Earth's barycentric motion (astrometric directions, ADR-28).
* :func:`gauss_iod` is Gauss's angles-only method (two-body, Earth-centred) for
  a first orbit from three observations; refine it with :func:`cislunar.od.fit`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ephemeris as E
from . import od
from .constants import EARTH_MOON, System

_MPC = re.compile(r"[CB](\d{4}) (\d\d) (\d\d\.\d{5,6}) ?(\d\d) (\d\d) (\d\d\.\d+) *([+-])(\d\d) (\d\d) (\d\d\.\d+)")
_HP = re.compile(r"CK(\d\d)(\d\d)(\d\d):(\d\d)(\d\d)(\d\d)(\d*) +(\d+\.\d+) +([+-]\d+\.\d+)")


@dataclass
class Record:
    jd_utc: float
    ra: float  # rad, J2000
    dec: float  # rad
    mag: float | None
    station: str
    line: str


def parse(source, strict: bool = True, skipped: list | None = None) -> list[Record]:
    """Parse astrometry from a file path, a string, or a list of lines (# = comment).

    Only ground-based CCD/CMOS positions (MPC note "C" or "B") are read. With strict=False,
    other lines (satellite, radar, roving-observer, garbled) are skipped and, if
    ``skipped`` is a list, appended to it.
    """
    if isinstance(source, (str, Path)) and Path(str(source)).exists():
        lines = Path(source).read_text(encoding="utf-8").splitlines()
    elif isinstance(source, str):
        lines = source.splitlines()
    else:
        lines = list(source)
    out = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        station = s[-3:]  # observatory code: always the last three characters
        m = _HP.search(s)
        if m:
            yy, mo, dd, hh, mi, ss, frac = m.groups()[:7]
            sec = float(f"{ss}.{frac}") if frac else float(ss)
            day = int(dd) + (int(hh) + int(mi) / 60 + sec / 3600) / 24
            jd = E.jd_from_calendar(2000 + int(yy), int(mo), day)
            ra, dec = np.radians(float(m.group(8))), np.radians(float(m.group(9)))
            tail = s[m.end():]
        else:
            m = _MPC.search(s)
            if not m:
                if strict:
                    raise ValueError(f"cannot parse astrometry line: {ln!r}")
                if skipped is not None:
                    skipped.append(ln)
                continue
            y, mo, d, rh, rm, rs, sg, dd_, dm, ds = m.groups()
            jd = E.jd_from_calendar(int(y), int(mo), float(d))
            ra = np.radians(15.0 * (int(rh) + int(rm) / 60 + float(rs) / 3600))
            dec = np.radians((1 if sg == "+" else -1) * (int(dd_) + int(dm) / 60 + float(ds) / 3600))
            tail = s[m.end():]
        mag = re.search(r"(\d+\.\d)", tail[:-3])
        out.append(Record(jd, float(ra), float(dec), float(mag.group(1)) if mag else None, station, ln))
    return out


MPC_OBSCODES_URL = "https://www.minorplanetcenter.net/iau/lists/ObsCodes.html"


def mpc_stations(codes, cache_path=None) -> dict:
    """Parallax constants for MPC observatory ``codes`` from the MPC's ObsCodes list
    (fetched once and cached as text at ``cache_path``). Codes without ground
    coordinates (spacecraft, roving observers) are left out."""
    from urllib.request import Request, urlopen
    text = None
    if cache_path and Path(cache_path).exists():
        text = Path(cache_path).read_text(encoding="utf-8")
    if text is None:
        req = Request(MPC_OBSCODES_URL, headers={"User-Agent": "cislunar-toolkit"})
        text = urlopen(req, timeout=60).read().decode("utf-8", errors="replace")
        if cache_path:
            Path(cache_path).write_text(text, encoding="utf-8")
    st = {}
    for ln in text.splitlines():
        code = ln[:3]
        if code in codes and len(ln) > 30 and ln[4:13].strip():
            try:
                st[code] = (float(ln[4:13]), float(ln[13:21]), float(ln[21:30]), ln[30:].strip())
            except ValueError:
                continue
    return st


def write_stations(st: dict, path, source: str = MPC_OBSCODES_URL) -> None:
    lines = [f"# MPC observatory codes ({source})", "# code  east_lon_deg  rho_cos_phi  rho_sin_phi  name"]
    lines += [f"{k} {v[0]:10.5f} {v[1]:9.6f} {v[2]:+10.6f}  {v[3]}" for k, v in sorted(st.items())]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_stations(path) -> dict:
    """{code: (east_lon_deg, rho_cos_phi, rho_sin_phi, name)} from an MPC-style table."""
    st = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        f = ln.split()
        st[f[0]] = (float(f[1]), float(f[2]), float(f[3]), " ".join(f[4:]))
    return st


def to_observations(records, stations: dict, dynamics, sigma_arcsec: float = 1.0,
                    sigma_by_station: dict | None = None) -> list:
    """od angle observations (J2000 RA/Dec) in the dynamics' frame and time base.

    sigma_by_station overrides the per-axis noise for given station codes.
    """
    s: System = dynamics.system
    c = E.C_KMS / s.vstar
    observers = {code: E.station_observer(*stations[code][:3], dynamics, name=code)
                 for code in {r.station for r in records}}
    out = []
    for r in records:
        jd = E.utc_to_tdb(r.jd_utc)
        t = float(dynamics.t_of_jd(jd))
        v_e = -dynamics.eph.state("sun", jd)[1]  # Earth's barycentric velocity (Sun's ~13 m/s ignored)
        sig = (sigma_by_station or {}).get(r.station, sigma_arcsec) * od.ARCSEC
        out.append(od.Observation(t, "angles", np.array([r.ra, r.dec]), np.array([sig, sig]),
                                  observers[r.station], "native", 0.0, r.station, c,
                                  s.nd_velocity(v_e)))
    return out


def residuals_arcsec(result, observations):
    """Per-observation (dRA cos dec, dDec) residuals [arcsec] of a fit."""
    res = od._prop(result.dynamics, result.x, result.epoch, [o.t for o in observations], result.system,
                   with_stm=False)
    out = []
    for o, (x, _) in zip(observations, res):
        d = o.residual(x)[0]
        out.append((d[0] * np.cos(o.z[1]) / od.ARCSEC, d[1] / od.ARCSEC))
    return np.array(out)


def gauss_iod(records, stations: dict, mu_kms: float = 398600.435436):
    """Gauss angles-only IOD from three records (two-body about the Earth's centre).

    Returns a list of candidate (jd_utc of the middle record, r_km, v_kms) in J2000,
    one per positive real root of the eighth-degree range polynomial.
    """
    if len(records) != 3:
        raise ValueError("need exactly three records")
    rec = sorted(records, key=lambda r: r.jd_utc)
    L, R = [], []
    for r in rec:
        L.append(np.array([np.cos(r.dec) * np.cos(r.ra), np.cos(r.dec) * np.sin(r.ra), np.sin(r.dec)]))
        R.append(E.station_j2000(*stations[r.station][:3], r.jd_utc)[0])
    t = [(r.jd_utc - rec[1].jd_utc) * 86400.0 for r in rec]
    t1, t3 = t[0], t[2]
    tau = t3 - t1
    p = [np.cross(L[1], L[2]), np.cross(L[0], L[2]), np.cross(L[0], L[1])]
    d0 = L[0] @ p[0]
    D = np.array([[R[i] @ p[j] for j in range(3)] for i in range(3)])
    A = (-D[0, 1] * t3 / tau + D[1, 1] + D[2, 1] * t1 / tau) / d0
    B = (D[0, 1] * (t3**2 - tau**2) * t3 / tau + D[2, 1] * (tau**2 - t1**2) * t1 / tau) / (6 * d0)
    Ev = R[1] @ L[1]
    a = -(A**2 + 2 * A * Ev + R[1] @ R[1])
    b = -2 * mu_kms * B * (A + Ev)
    c = -(mu_kms * B) ** 2
    roots = np.roots([1, 0, a, 0, 0, b, 0, 0, c])
    out = []
    for x in roots:
        if abs(x.imag) > 1e-6 * abs(x) or x.real <= 0:
            continue
        r2 = x.real
        rho1 = ((6 * (D[2, 0] * t1 / t3 + D[1, 0] * tau / t3) * r2**3
                 + mu_kms * D[2, 0] * (tau**2 - t1**2) * t1 / t3)
                / (6 * r2**3 + mu_kms * (tau**2 - t3**2)) - D[0, 0]) / d0
        rho2 = A + mu_kms * B / r2**3
        rho3 = ((6 * (D[0, 2] * t3 / t1 - D[1, 2] * tau / t1) * r2**3
                 + mu_kms * D[0, 2] * (tau**2 - t3**2) * t3 / t1)
                / (6 * r2**3 + mu_kms * (tau**2 - t1**2)) - D[2, 2]) / d0
        if min(rho1, rho2, rho3) <= 0:
            continue
        r1v, r2v, r3v = R[0] + rho1 * L[0], R[1] + rho2 * L[1], R[2] + rho3 * L[2]
        f1 = 1 - 0.5 * mu_kms * t1**2 / r2**3
        f3 = 1 - 0.5 * mu_kms * t3**2 / r2**3
        g1 = t1 - mu_kms * t1**3 / (6 * r2**3)
        g3 = t3 - mu_kms * t3**3 / (6 * r2**3)
        v2 = (-f3 * r1v + f1 * r3v) / (f1 * g3 - f3 * g1)
        out.append((rec[1].jd_utc, r2v, v2))
    return out


def _scan_seed(args):
    """One range-scan seed (top level so it can run in a worker process)."""
    from .constants import EARTH_MOON as S
    records, stations, dynamics, ends, p1, p2, max_rms = args
    (j1, R1, L1), (j2, R2, L2) = ends
    r1, r2 = R1 + p1 * L1, R2 + p2 * L2
    x1 = S.state_to_nd(np.concatenate((r1, (r2 - r1) / ((j2 - j1) * 86400.0))))
    try:
        obs = to_observations(records, stations, dynamics)
        x0 = dynamics.propagate_many(x1, float(dynamics.t_of_jd(j1)), [0.0], with_stm=False)[0][0]
        f = od.fit(obs, x0, 0.0, S, dynamics=dynamics, max_iter=30)
    except Exception:  # impacts, divergence: not a viable seed
        return None
    if not f.converged or f.rms > max_rms:
        return None
    return f.x, f.chi2, float(p1), float(p2)


def range_scan_iod(records, stations: dict, dynamics, ranges_km=None, top: int = 3, workers: int = 1,
                   max_obs: int = 40, max_rms: float = 5.0):
    """Angles-only first orbit by scanning trial distances (an admissible-region search).

    For the first and last record of a short arc, every pair of trial ranges
    (rho1, rho2) defines two positions and a straight-line velocity; each seed is
    refined by least squares on the arc's records (thinned to ``max_obs``) in
    ``dynamics``, so the Sun and Moon are included, unlike Gauss's two-body method.
    Seeds that don't converge or fit worse than ``max_rms`` (in units of the
    observation sigma) are dropped. workers > 1 uses processes (Windows: call from
    under ``if __name__ == "__main__":``). Returns up to ``top`` distinct
    (ODResult, rho1_km, rho2_km), best first; epoch is dynamics time 0.
    """
    from .constants import EARTH_MOON as S
    rec = sorted(records, key=lambda r: r.jd_utc)
    if len(rec) > max_obs:
        keep = sorted(set(np.linspace(0, len(rec) - 1, max_obs).round().astype(int)))
        rec = [rec[k] for k in keep]
    ranges_km = np.geomspace(2e4, 3e7, 14) if ranges_km is None else np.asarray(ranges_km, float)
    ends = []
    for r in (rec[0], rec[-1]):
        L = np.array([np.cos(r.dec) * np.cos(r.ra), np.cos(r.dec) * np.sin(r.ra), np.sin(r.dec)])
        ends.append((E.utc_to_tdb(r.jd_utc), E.station_j2000(*stations[r.station][:3], r.jd_utc)[0], L))
    tasks = [(rec, stations, dynamics, ends, p1, p2, max_rms) for p1 in ranges_km for p2 in ranges_km]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_scan_seed, tasks, chunksize=4))
    else:
        res = [_scan_seed(t) for t in tasks]
    res = sorted([r for r in res if r is not None], key=lambda r: r[1])
    obs = to_observations(rec, stations, dynamics)
    uniq = []
    for x, chi2, p1, p2 in res:
        if all(np.linalg.norm(x[:3] - g.x[:3]) > 1e-4 for g, _, _ in uniq):
            uniq.append((od.fit(obs, x, 0.0, S, dynamics=dynamics), p1, p2))
        if len(uniq) >= top:
            break
    return uniq


__all__ = ["Record", "parse", "load_stations", "mpc_stations", "write_stations", "range_scan_iod", "to_observations", "residuals_arcsec", "gauss_iod"]
