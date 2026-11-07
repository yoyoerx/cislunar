"""Delta-v budget roll-up.

A mission budget has four layers:
  deterministic    - planned burns (departure, insertion, return ...)
  statistical      - trajectory corrections that depend on errors (a percentile
                     of a TCM analysis or Monte Carlo)
  station-keeping  - per-year cost of staying on an unstable orbit x years
  margin           - a percentage of everything above (finite-burn losses,
                     contingency, model error)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CATEGORIES = ("deterministic", "statistical", "station-keeping", "margin")


@dataclass
class BudgetItem:
    name: str
    category: str
    dv_mps: float
    note: str = ""


@dataclass
class DeltaVBudget:
    title: str = "Delta-v budget"
    margin_frac: float = 0.10
    items: list = field(default_factory=list)

    def add(self, name: str, dv_mps: float, category: str = "deterministic", note: str = ""):
        if category not in CATEGORIES[:3]:
            raise ValueError(f"category must be one of {CATEGORIES[:3]}")
        self.items.append(BudgetItem(name, category, float(dv_mps), note))
        return self

    def add_transfer(self, transfer, name: str = "transfer", flyby: bool = False):
        """Add a targeting.Transfer: departure burn, plus the arrival burn unless flyby."""
        self.add(f"{name}: departure", transfer.dv1_mps, note=f"TOF {transfer.tof_days:.2f} d")
        if not flyby:
            self.add(f"{name}: arrival", transfer.dv2_mps)
        return self

    def add_tcm(self, result, name: str = "TCMs", percentile: float = 99.0, index: int = 0):
        """Add statistical Delta-v from navigation.linear_tcm (TCM at t_tcm[index],
        including the arrival-burn change) or navigation.monte_carlo_tcm."""
        if hasattr(result, "total_p99"):  # TcmResult
            if percentile != 99.0:
                raise ValueError("TcmResult stores the 99th percentile; use percentile=99")
            dv = float(result.total_p99[index])
            note = f"p99, TCM at day {result.t_tcm_days[index]:.2f}"
        else:  # MonteCarloResult
            dv = result.percentile(percentile)
            note = f"p{percentile:g} of {len(result.miss_km)} Monte Carlo runs"
        return self.add(name, dv, "statistical", note)

    def add_stationkeeping(self, result, years: float, name: str = "station-keeping",
                           percentile: float = 95.0):
        """Add navigation.StationKeepingResult: percentile of m/s/yr times years."""
        ok = ~result.lost
        if not ok.any():
            raise ValueError("station-keeping was lost in every run; shorten the interval")
        per_year = float(np.percentile(result.per_year_mps[ok], percentile))
        return self.add(name, per_year * years, "station-keeping",
                        f"p{percentile:g} {per_year:.2f} m/s/yr x {years:g} yr, tracking every "
                        f"{result.interval_days:g} d")

    def subtotal(self, category: str) -> float:
        if category == "margin":
            return self.margin_frac * sum(self.subtotal(c) for c in CATEGORIES[:3])
        return sum(i.dv_mps for i in self.items if i.category == category)

    @property
    def total(self) -> float:
        return sum(self.subtotal(c) for c in CATEGORIES)

    def to_dict(self) -> dict:
        return {"title": self.title, "margin_frac": self.margin_frac,
                "items": [vars(i) for i in self.items],
                "subtotals": {c: self.subtotal(c) for c in CATEGORIES}, "total": self.total}

    def table(self) -> str:
        w = max([len(i.name) for i in self.items] + [24])
        lines = [self.title, "-" * (w + 40)]
        for c in CATEGORIES[:3]:
            rows = [i for i in self.items if i.category == c]
            if not rows:
                continue
            lines.append(f"{c.upper()}")
            for i in rows:
                lines.append(f"  {i.name:<{w}} {i.dv_mps:10.1f} m/s  {i.note}")
            lines.append(f"  {'subtotal':<{w}} {self.subtotal(c):10.1f} m/s")
        lines.append(f"MARGIN ({self.margin_frac * 100:.0f}%)".ljust(w + 2) +
                     f" {self.subtotal('margin'):10.1f} m/s")
        lines.append("-" * (w + 40))
        lines.append(f"{'TOTAL':<{w + 2}} {self.total:10.1f} m/s")
        return "\n".join(lines)
