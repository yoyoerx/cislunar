"""Fetch the public data for the rock-or-rocket case study (run once; results are committed).

* 2024 PT5 astrometry from the Minor Planet Center API (OBS80 format)
* MPC observatory parallax constants for every station used (both objects)
* Moon/Sun geocentric tables from JPL Horizons covering each arc
* JPL's own orbit of 2024 PT5 (Horizons vectors and SBDB orbit summary) for comparison

2020 SO astrometry comes from Project Pluto (see 2020so_astrometry.txt header).
"""
import json
from pathlib import Path
from urllib.request import Request, urlopen

from cislunar import astrometry as A
from cislunar import ephemeris as E

HERE = Path(__file__).parent
UA = {"User-Agent": "cislunar-toolkit", "Content-Type": "application/json"}


def get(url, data=None, method="GET"):
    req = Request(url, data=data, headers=UA, method=method)
    return urlopen(req, timeout=120).read().decode("utf-8")


def main():
    # 2024 PT5 observations
    body = json.dumps({"desigs": ["2024 PT5"], "output_format": ["OBS80"]}).encode()
    res = json.loads(get("https://data.minorplanetcenter.net/api/get-obs", body))
    obs80 = res[0]["OBS80"]
    hdr = ("# Astrometry of 2024 PT5 (natural mini-moon), MPC OBS80 format.\n"
           "# Source: Minor Planet Center get-obs API, https://data.minorplanetcenter.net/api/get-obs (fetched 2026-11-07).\n")
    (HERE / "2024pt5_astrometry.txt").write_text(hdr + obs80.rstrip() + "\n", encoding="utf-8")
    print("2024 PT5:", len(obs80.splitlines()), "lines")

    codes = set()
    for name in ("2020so_astrometry.txt", "2024pt5_astrometry.txt"):
        for r in A.parse(HERE / name, strict=False):
            codes.add(r.station)
    st = A.mpc_stations(codes, cache_path=HERE / "_obscodes.html")
    A.write_stations(st, HERE / "stations.txt")
    print(f"stations: {len(st)} of {len(codes)} codes have ground coordinates; missing: {sorted(codes - set(st))}")
    (HERE / "_obscodes.html").unlink(missing_ok=True)

    E.Ephemeris.cached(HERE / "ephemeris_2020_2021.json", "2020-10-01", "2021-05-01", "3h")
    E.Ephemeris.cached(HERE / "ephemeris_2024_2025.json", "2024-07-01", "2025-03-15", "3h")

    jpl = E.Ephemeris({"pt5": E.horizons_vectors("DES=2024 PT5;", "2024-07-01", "2025-03-15", "3h")},
                      source="JPL Horizons small-body 2024 PT5, geocentric ICRF, TDB (JPL orbit solution, not independent truth)")
    jpl.save(HERE / "jpl_2024pt5.json")
    sb = json.loads(get("https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=2024%20PT5&phys-par=1"))
    keep = {"object": sb.get("object"), "orbit": {k: sb["orbit"].get(k) for k in
            ("orbit_id", "epoch", "first_obs", "last_obs", "n_obs_used", "data_arc", "model_pars", "rms", "comment")},
            "phys_par": sb.get("phys_par"), "fetched": "2026-11-07", "source": "https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=2024%20PT5"}
    (HERE / "jpl_2024pt5_sbdb.json").write_text(json.dumps(keep, indent=1), encoding="utf-8")
    print("SBDB model pars:", keep["orbit"]["model_pars"])


if __name__ == "__main__":
    main()
