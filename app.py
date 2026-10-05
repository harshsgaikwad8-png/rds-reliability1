"""Flask app serving the RDS reliability dashboard."""
import os
from dataclasses import asdict
from flask import Flask, jsonify, render_template, request

import reliability as rel

app = Flask(__name__)

# Results that never change are computed once at start-up
_BASE = rel.Params()
_CACHE = {}


def _cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return "ok"


@app.get("/api/overview")
def overview():
    a = _cached("a", lambda: rel.analytical(_BASE))
    m = _cached("m", lambda: rel.monte_carlo(_BASE))
    return jsonify(system=a["system"], rating=rel.rating(a["system"]),
                   mc_years=m["years"], load_points=len(rel.CUSTOMERS),
                   customers=sum(rel.CUSTOMERS), feeder_km=sum(rel.SECTION_KM))


@app.get("/api/network")
def network():
    return jsonify(components=rel.network_table(_BASE), customers=rel.CUSTOMERS,
                   load_kw=rel.LOAD_KW, section_km=rel.SECTION_KM,
                   lateral_km=rel.LATERAL_KM)


@app.get("/api/indices")
def indices():
    a = _cached("a", lambda: rel.analytical(_BASE))
    return jsonify(load_points=a["load_points"], contribution=a["contribution"],
                   per_component=a["per_component"])


@app.get("/api/validation")
def validation():
    a = _cached("a", lambda: rel.analytical(_BASE))
    m = _cached("m", lambda: rel.monte_carlo(_BASE))
    rows = []
    for k in ["SAIFI", "SAIDI", "CAIDI", "ENS", "AENS", "ASAI"]:
        av, mv = a["system"][k], m[k]["mean"]
        err = abs(av - mv) / abs(av) * 100 if av else 0.0
        rows.append(dict(index=k, analytical=av, monte_carlo=mv, error_pct=err,
                         p5=m[k]["p5"], p95=m[k]["p95"]))
    return jsonify(rows=rows, years=m["years"], hist=m["saidi_hist"],
                   prob_exceed=m["prob_saidi_exceeds_target"],
                   target_saidi=rel.TARGET_SAIDI)


@app.get("/api/scenarios")
def scenarios():
    return jsonify(scenarios=_cached("sc", rel.scenarios),
                   sensitivity=_cached("sens", rel.sensitivity))


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}

    def num(key, default, lo, hi):
        try:
            return min(max(float(body.get(key, default)), lo), hi)
        except (TypeError, ValueError):
            return default

    p = rel.Params(
        failure_scale=num("failure_scale", 1.0, 0.1, 5.0),
        repair_scale=num("repair_scale", 1.0, 0.1, 5.0),
        switching_h=num("switching_h", 1.0, 0.05, 6.0),
        tie_switch=bool(body.get("tie_switch", False)),
        automation=bool(body.get("automation", False)),
        load_growth_pct=num("load_growth_pct", 0.0, -50, 200),
    )
    a = rel.analytical(p)
    base = _cached("a", lambda: rel.analytical(_BASE))["system"]
    return jsonify(params=asdict(p), system=a["system"], base=base,
                   rating=rel.rating(a["system"]), load_points=a["load_points"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
