"""Case study: where should a satellite-servicing base station live?

Computes (or loads) the station -> client route matrix with the CR3BP toolkit, then
compares campaigns of 10, 100 and 1,000 servicing missions flown directly from LEO
against reusable servicers based at each candidate station, using a rocket-equation
mass model. The interactive report is docs/case_study_base_stations.html
(python docs/build_case_study.py).
"""
import json
from pathlib import Path

from cislunar import servicing as SV

ROOT = Path(__file__).parents[1]
CACHE = ROOT / "docs" / "_cache" / "routes.json"
SHORT = {"leo": "LEO depot", "sso": "LEO SSO", "meo": "MEO/GNSS", "geo": "GEO", "heo21": "2:1 transl.",
         "nrho": "NRHO", "dro": "DRO", "l1h": "L1 halo", "l2h": "L2 halo", "llo": "LLO"}


def main():
    costs = SV.cost_matrix(cache=str(CACHE), verbose=True)  # ~30 min the first time, parallel

    print("\nOne-way route Delta-v, station (rows) -> client (columns) [m/s]")
    print(f"{'':12s}" + "".join(f"{SHORT[c]:>11s}" for c in SV.CLIENT_KEYS))
    for s in SV.STATION_KEYS:
        print(f"{SHORT[s]:12s}" + "".join(f"{costs[f'{s}->{c}']['total_mps']:11.0f}" for c in SV.CLIENT_KEYS))

    mm = SV.MassModel()
    print(f"\nMass model: station {mm.station_kg / 1000:.0f} t, servicer {mm.servicer_kg:.0f} kg, kit "
          f"{mm.kit_kg:.0f} kg, Isp {mm.isp_s:.0f} s, servicer life {mm.servicer_life} missions")
    for n in (10, 100, 1000):
        print(f"\n{n} missions per client class: cheapest architecture (mass in LEO)")
        for c in SV.CLIENT_KEYS:
            options = [(SV.campaign(n, c, None, costs, mm)["imleo_kg"], "direct")]
            options += [(SV.campaign(n, c, s, costs, mm)["imleo_kg"], s) for s in SV.STATION_KEYS]
            options.sort()
            direct = next(v for v, k in options if k == "direct")
            best_v, best_k = options[0]
            print(f"  {SHORT[c]:12s} best: {SHORT.get(best_k, 'direct from LEO'):16s} {best_v / 1000:10.1f} t  "
                  f"(direct {direct / 1000:10.1f} t, saving {100 * (1 - best_v / direct):4.0f}%)")

    print("\nBreak-even mission count (station beats direct), by client:")
    for c in SV.CLIENT_KEYS:
        be = {s: SV.break_even(c, s, costs, mm) for s in SV.STATION_KEYS}
        ok = {s: v for s, v in be.items() if v is not None}
        best = min(ok, key=ok.get) if ok else None
        print(f"  {SHORT[c]:12s} " + (f"soonest: {SHORT[best]} after {ok[best]} missions" if best else "no station pays off"))

    out = ROOT / "examples" / "output" / "servicing_summary.json"
    out.write_text(json.dumps({k: v["total_mps"] for k, v in costs.items()}, indent=1), encoding="utf-8")
    print(f"\nroute table written to {out}")


if __name__ == "__main__":
    main()
