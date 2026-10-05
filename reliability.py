"""Reliability model for a Radial Distribution System (RDS).

Sample 6-node radial feeder:  Substation -S1- N1 -S2- N2 -S3- N3 ... -S6- N6
Each node Ni feeds one load point LPi through a fused lateral and a distribution
transformer.  All network data below is SYNTHETIC sample data (typical textbook
failure rates), meant to be replaced with the utility's own records.

Two methods are provided:
  * analytical  - classic series-system / failure-mode-effect analysis
  * monte_carlo - annual-sample simulation used to validate the analytical result
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

HOURS_PER_YEAR = 8760.0

# ----------------------------------------------------------------------------
# Sample network data
# ----------------------------------------------------------------------------
SECTION_KM = [0.75, 0.80, 0.65, 0.90, 0.60, 0.70]      # main feeder sections
LATERAL_KM = [0.60, 0.75, 0.55, 0.80, 0.50, 0.65]      # fused laterals
CUSTOMERS = [220, 310, 150, 410, 260, 180]             # customers per load point
LOAD_KW = [1100, 1500, 750, 2000, 1300, 900]           # average load per load point

LAMBDA_MAIN = 0.065      # failures / yr / km
LAMBDA_LATERAL = 0.100   # failures / yr / km
LAMBDA_TRANSFORMER = 0.015  # failures / yr / unit
R_MAIN = 5.0             # mean repair time, h
R_LATERAL = 3.0
R_TRANSFORMER = 8.0

N_LP = len(CUSTOMERS)

# Illustrative targets used only for the pass/fail rating on the dashboard
TARGET_SAIFI = 0.40
TARGET_SAIDI = 1.00


@dataclass
class Params:
    failure_scale: float = 1.0      # multiplies every failure rate
    repair_scale: float = 1.0       # multiplies every repair time
    switching_h: float = 1.0        # manual isolation / switching time
    tie_switch: bool = False        # normally-open tie to a backup feeder
    tie_transfer_h: float = 1.5     # time to restore downstream LPs via the tie
    automation: bool = False        # remote-controlled sectionalizers
    load_growth_pct: float = 0.0    # growth of load (affects ENS only)

    def effective_switching(self) -> float:
        return 0.1 if self.automation else self.switching_h

    def effective_tie(self) -> float:
        return 0.25 if self.automation else self.tie_transfer_h


def network_table(p: Params | None = None):
    """Component list with rates (used both by the model and the UI)."""
    p = p or Params()
    comps = []
    for k, km in enumerate(SECTION_KM):
        comps.append(dict(id=f"S{k+1}", kind="main", length_km=km,
                          lam=LAMBDA_MAIN * km * p.failure_scale,
                          repair_h=R_MAIN * p.repair_scale, node=k))
    for j, km in enumerate(LATERAL_KM):
        comps.append(dict(id=f"L{j+1}", kind="lateral", length_km=km,
                          lam=LAMBDA_LATERAL * km * p.failure_scale,
                          repair_h=R_LATERAL * p.repair_scale, node=j))
    for j in range(N_LP):
        comps.append(dict(id=f"T{j+1}", kind="transformer", length_km=0.0,
                          lam=LAMBDA_TRANSFORMER * p.failure_scale,
                          repair_h=R_TRANSFORMER * p.repair_scale, node=j))
    return comps


def _duration_matrix(comps, p: Params):
    """For each component: duration (h) and whether it needs a repair draw, per LP.

    Returns durations (C x LP) and need_repair (C x LP bool).
    deterministic duration is used where need_repair is False.
    """
    s, tie = p.effective_switching(), p.effective_tie()
    C = len(comps)
    dur = np.zeros((C, N_LP))
    rep = np.zeros((C, N_LP), dtype=bool)
    for c, comp in enumerate(comps):
        if comp["kind"] == "main":
            k = comp["node"]               # section k feeds node k
            for j in range(N_LP):
                if j < k:                  # upstream of fault: isolate + close breaker
                    dur[c, j] = s
                elif p.tie_switch:         # downstream: transfer to backup via tie
                    dur[c, j] = s + tie
                else:                      # downstream: wait for repair
                    dur[c, j] = comp["repair_h"]
                    rep[c, j] = True
        else:                              # fused lateral / transformer: only own LP
            j = comp["node"]
            dur[c, j] = comp["repair_h"]
            rep[c, j] = True
    return dur, rep


def _indices(freq, hours, ens_kwh):
    """freq/hours/ens per LP (or per sample x LP) -> system indices."""
    n = np.array(CUSTOMERS, float)
    N = n.sum()
    saifi = (freq * n).sum(-1) / N
    saidi = (hours * n).sum(-1) / N
    caidi = np.divide(saidi, saifi, out=np.zeros_like(saidi), where=saifi > 0)
    ens = ens_kwh.sum(-1)
    aens = ens / N
    asai = 1 - saidi / HOURS_PER_YEAR
    return saifi, saidi, caidi, ens, aens, asai


def analytical(p: Params | None = None) -> dict:
    p = p or Params()
    comps = network_table(p)
    dur, _ = _duration_matrix(comps, p)
    lam = np.array([c["lam"] for c in comps])
    # every component failure interrupts the LPs where dur > 0
    affected = (dur > 0).astype(float)
    freq = (lam[:, None] * affected).sum(0)                 # f/yr per LP
    hours = (lam[:, None] * dur).sum(0)                     # h/yr per LP
    load = np.array(LOAD_KW) * (1 + p.load_growth_pct / 100)
    ens_kwh = hours * load                                  # kWh/yr
    saifi, saidi, caidi, ens, aens, asai = _indices(freq, hours, ens_kwh)

    # contribution to SAIDI by component group
    n = np.array(CUSTOMERS, float)
    contrib = (lam[:, None] * dur * n[None, :]).sum(1) / n.sum()
    groups = {"Main feeder sections": 0.0, "Fused laterals": 0.0, "Transformers": 0.0}
    names = {"main": "Main feeder sections", "lateral": "Fused laterals",
             "transformer": "Transformers"}
    for c, comp in enumerate(comps):
        groups[names[comp["kind"]]] += float(contrib[c])
    per_component = [dict(id=comp["id"], kind=comp["kind"],
                          saidi=float(contrib[c])) for c, comp in enumerate(comps)]

    return dict(
        system=dict(SAIFI=float(saifi), SAIDI=float(saidi), CAIDI=float(caidi),
                    ENS=float(ens), AENS=float(aens), ASAI=float(asai)),
        load_points=[dict(id=f"LP{j+1}", customers=CUSTOMERS[j], load_kw=float(load[j]),
                          lam=float(freq[j]), U=float(hours[j]),
                          r=float(hours[j] / freq[j]) if freq[j] else 0.0,
                          ens=float(ens_kwh[j])) for j in range(N_LP)],
        contribution=groups,
        per_component=per_component,
    )


def monte_carlo(p: Params | None = None, years: int = 20000, seed: int = 42) -> dict:
    p = p or Params()
    rng = np.random.default_rng(seed)
    comps = network_table(p)
    dur, rep = _duration_matrix(comps, p)
    C = len(comps)
    freq = np.zeros((years, N_LP))
    hours = np.zeros((years, N_LP))
    for c, comp in enumerate(comps):
        K = rng.poisson(comp["lam"], years)                       # failures per year
        # one shared repair-time draw per (year, component) for LPs that wait for repair
        shape = np.where(K > 0, K, 1)
        repair_total = rng.gamma(shape, comp["repair_h"]) * (K > 0)
        for j in range(N_LP):
            if dur[c, j] <= 0:
                continue
            freq[:, j] += K
            hours[:, j] += repair_total if rep[c, j] else K * dur[c, j]
    load = np.array(LOAD_KW) * (1 + p.load_growth_pct / 100)
    ens_kwh = hours * load
    saifi, saidi, caidi, ens, aens, asai = _indices(freq, hours, ens_kwh)

    def stats(x):
        return dict(mean=float(x.mean()), std=float(x.std()),
                    p5=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)))

    hist, edges = np.histogram(saidi, bins=24, range=(0, float(np.percentile(saidi, 99.5))))
    caidi_stats = stats(caidi)
    caidi_stats["mean"] = float(saidi.mean() / saifi.mean())   # ratio of means, as in the analytical method
    return dict(
        years=years,
        SAIFI=stats(saifi), SAIDI=stats(saidi), CAIDI=caidi_stats,
        ENS=stats(ens), AENS=stats(aens), ASAI=stats(asai),
        saidi_hist=dict(counts=(hist / years * 100).tolist(),
                        edges=[float(e) for e in edges]),
        prob_saidi_exceeds_target=float((saidi > TARGET_SAIDI).mean()),
    )


def scenarios(base: Params | None = None) -> list[dict]:
    b = base or Params()
    defs = [("Base case", dict(tie_switch=False, automation=False)),
            ("+ Tie switch", dict(tie_switch=True, automation=False)),
            ("+ Automation", dict(tie_switch=False, automation=True)),
            ("Tie + Automation", dict(tie_switch=True, automation=True))]
    out = []
    for name, kw in defs:
        q = Params(**{**asdict(b), **kw})
        s = analytical(q)["system"]
        out.append(dict(name=name, **s))
    return out


def sensitivity() -> dict:
    xs = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    base = [analytical(Params(failure_scale=x))["system"]["SAIDI"] for x in xs]
    best = [analytical(Params(failure_scale=x, tie_switch=True, automation=True))
            ["system"]["SAIDI"] for x in xs]
    return dict(scale=xs, base=base, upgraded=best, target=TARGET_SAIDI)


def rating(system: dict) -> dict:
    ok_f = system["SAIFI"] <= TARGET_SAIFI
    ok_d = system["SAIDI"] <= TARGET_SAIDI
    label = "Meets targets" if (ok_f and ok_d) else (
        "Partially meets targets" if (ok_f or ok_d) else "Below targets")
    return dict(label=label, saifi_ok=bool(ok_f), saidi_ok=bool(ok_d),
                target_saifi=TARGET_SAIFI, target_saidi=TARGET_SAIDI)
