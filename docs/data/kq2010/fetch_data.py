"""Fetch the public data for the 2010 KQ case study (run once; results are committed).

* 2010 KQ astrometry: the MPC deleted the object as artificial (MPEC 2010-N23)
  and no longer serves its observations, so they come from Bill Gray's Project
  Pluto pseudo-MPEC (rk.htm), which lists them in MPC 80-column format.
* MPC observatory parallax constants for every station used.
* Moon/Sun geocentric tables from JPL Horizons for the 2010 arc (1 h step), and
  Moon, Sun and all eight other planets for 1957-2010 (backward runs) and
  2010-2050 (forward runs), gzip-compressed JSON.
"""
import html
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

from cislunar import astrometry as A
from cislunar import ephemeris as E

HERE = Path(__file__).parent
UA = {"User-Agent": "cislunar-toolkit"}
GRAY_URL = "https://www.projectpluto.com/pluto/mpecs/rk.htm"


def astrometry():
    raw = urlopen(Request(GRAY_URL, headers=UA), timeout=60).read().decode("utf-8", "replace")
    lines = [html.unescape(re.sub(r"<[^>]+>", "", ln)) for ln in raw.splitlines() if "K10K00Q" in ln]
    seen = []
    for ln in lines:
        if ln not in seen:
            seen.append(ln)
    hdr = ("# 2010 KQ = RK252A5 (deleted from the MPC catalogue as an artificial object, MPEC 2010-N23).\n"
           f"# Source: Bill Gray, Project Pluto pseudo-MPEC {GRAY_URL}\n"
           "#   (created 2010 Jun 25 with Find_Orb; fetched 2026-11-07). Exact duplicate lines removed.\n"
           "# Original observers: MPC observatory codes in columns 78-80.\n")
    (HERE / "astrometry.txt").write_text(hdr + "\n".join(seen) + "\n", encoding="utf-8")
    print("astrometry:", len(seen), "lines")


def long_table(path, start, stop):
    """Moon 12 h, Sun and Mercury 1 d, Venus-Saturn 2 d, Uranus/Neptune 4 d, gzip JSON."""
    if path.exists():
        return
    parts = [E.Ephemeris.fetch(start, stop, "12h", {"moon": "301"}),
             E.Ephemeris.fetch(start, stop, "1d", {"sun": "10", "mercury": "199"}),
             E.Ephemeris.fetch(start, stop, "2d", {k: E.PLANETS[k][1] for k in ("venus", "mars", "jupiter", "saturn")}),
             E.Ephemeris.fetch(start, stop, "4d", {"uranus": "7", "neptune": "8"})]
    eph = parts[0]
    for q in parts[1:]:
        eph = eph.merged(q)
    eph.source = (f"JPL Horizons geocentric ICRF, TDB, {start} to {stop}: Moon 12 h, Sun and Mercury 1 d, "
                  "Venus/Mars/Jupiter/Saturn 2 d, Uranus/Neptune 4 d (system barycentres for Mars and beyond)")
    eph.save(path)


def validation():
    """JPL Horizons states of two natural near-Earth asteroids (gravity-only JPL orbits) used
    to validate the decades-long propagation: yearly positions 1990-2010 and a start state."""
    path = HERE / "validation_horizons.json"
    if path.exists():
        return
    out = {"source": "JPL Horizons geocentric ICRF vectors, TDB (fetched 2026-11-07)", "objects": {}}
    for label, cmd in (("1991 VG", "DES=1991 VG;"), ("3753 Cruithne", "3753")):
        jd, r, v = E.horizons_vectors(cmd, "1990-06-01", "2010-06-01", "365d")
        j0, r0, v0 = E.horizons_vectors(cmd, "2010-06-01", "2010-06-02", "1d")
        out["objects"][label] = {"command": cmd, "jd_tdb": jd.tolist(), "r_km": r.tolist(),
                                 "start": {"jd_tdb": float(j0[0]), "r_km": r0[0].tolist(), "v_kms": v0[0].tolist()}}
    path.write_text(json.dumps(out), encoding="utf-8")


def main():
    if not (HERE / "astrometry.txt").exists():
        astrometry()
    codes = {r.station for r in A.parse(HERE / "astrometry.txt", strict=False)}
    st = A.mpc_stations(codes, cache_path=HERE / "_obscodes.html")
    A.write_stations(st, HERE / "stations.txt")
    (HERE / "_obscodes.html").unlink(missing_ok=True)
    print("stations:", sorted(st))

    E.Ephemeris.cached(HERE / "ephemeris_2010.json", "2010-05-01", "2010-07-10", "1h")
    print("2010 Moon/Sun table done")
    long_table(HERE / "ephemeris_1957_2010.json.gz", "1957-10-01", "2010-07-10")
    long_table(HERE / "ephemeris_2010_2050.json.gz", "2010-05-01", "2050-01-01")
    print("long table done")
    validation()
    print("validation states done")


if __name__ == "__main__":
    main()
