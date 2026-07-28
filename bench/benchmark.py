"""Every table in the README, reproducibly.

    python bench/benchmark.py            # all of it, a few minutes
    python bench/benchmark.py --quick    # fewer seeds, for a sanity check

Each experiment changes exactly one thing and holds the rest fixed, which is
the only way the numbers mean anything. The filter bank, gate, track scoring
and initiation logic are identical across every row of every table; only the
named variable moves.
"""

import argparse
import time

import numpy as np

from tf import Config, Tracker
from tf.metrics import identity_metrics, ospa_sequence, rmse_position
from tf.scenarios import crossing, dense, manoeuvre
from tf.track import Track

AREA = {"crossing": 8000 * 6000, "manoeuvre": 8000 * 5000, "dense": 6000 * 6000}


def run(scenario, associator="jpda", single_model=False, clutter=1.0,
        area=1.0, **cfg_kw):
    Track.reset_ids()
    cfg = Config(sigma=20.0, clutter_density=clutter / area, **cfg_kw)
    t = Tracker(cfg, associator, single_model=single_model)
    t0 = time.perf_counter()
    est = t.run(scenario.scans)
    elapsed = time.perf_counter() - t0

    total, loc, card = ospa_sequence(scenario.truth, est)
    ident = identity_metrics(scenario.truth, est)
    return {
        "ospa": total, "loc": loc, "card": card,
        "rmse": rmse_position(scenario.truth, est),
        "seconds": elapsed, "stats": t.stats, **ident,
    }


def mean(rows, key):
    vals = [r[key] for r in rows if not np.isnan(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def table(title, header, rows):
    print(f"\n## {title}\n")
    widths = [max(len(str(r[i])) for r in [header] + rows)
              for i in range(len(header))]
    def line(cells):
        return "| " + " | ".join(str(c).ljust(w)
                                 for c, w in zip(cells, widths)) + " |"
    print(line(header))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        print(line(r))


# -- experiments -----------------------------------------------------------

def exp_imm(seeds):
    """IMM against a single CV filter on a manoeuvring target."""
    rows = []
    for name, single in (("single CV filter", True), ("IMM (CV + 2 turns)", False)):
        rs = [run(manoeuvre(seed=s), "jpda", single_model=single,
                  clutter=0.5, area=AREA["manoeuvre"], Pd=0.98)
              for s in range(seeds)]
        rows.append([name, f"{mean(rs,'ospa'):.2f}", f"{mean(rs,'loc'):.2f}",
                     f"{mean(rs,'rmse'):.1f}", f"{mean(rs,'coverage'):.3f}"])
    return table("IMM vs a single motion model (manoeuvring target, no association difficulty)",
                 ["filter", "OSPA", "OSPA loc", "RMSE m", "coverage"], rows)


def exp_turn_rate(seeds):
    """How the IMM's advantage scales with how hard the turn is."""
    rows = []
    for rate in (0.0, 3.0, 6.0, 12.0):
        out = {}
        for single in (True, False):
            rs = [run(manoeuvre(seed=s, turn_rate=rate), "jpda",
                      single_model=single, clutter=0.5,
                      area=AREA["manoeuvre"], Pd=0.98)
                  for s in range(seeds)]
            out[single] = mean(rs, "ospa")
        gain = (out[True] - out[False]) / out[True] * 100.0
        rows.append([f"{rate:.0f}", f"{out[True]:.2f}", f"{out[False]:.2f}",
                     f"{gain:+.1f}%"])
    return table("IMM advantage vs turn rate",
                 ["turn deg/s", "single CV", "IMM", "OSPA change"], rows)


def exp_association(seeds):
    """Soft vs hard association on a crossing."""
    rows = []
    for name in ("gnn", "jpda"):
        rs = [run(crossing(seed=s, clutter=3.0), name, clutter=3.0,
                  area=AREA["crossing"], Pd=0.95) for s in range(seeds)]
        rows.append([name.upper(), f"{mean(rs,'ospa'):.2f}",
                     f"{mean(rs,'loc'):.2f}", f"{mean(rs,'card'):.2f}",
                     f"{mean(rs,'false_tracks_per_scan'):.2f}",
                     f"{mean(rs,'coverage'):.3f}"])
    return table("Soft vs hard association (two crossing targets, light clutter)",
                 ["assoc", "OSPA", "OSPA loc", "OSPA card", "false trk/scan",
                  "coverage"], rows)


def exp_clutter(seeds, levels):
    """Where JPDA starts paying for itself, and what it costs."""
    rows = []
    for cl in levels:
        out = {}
        for name in ("gnn", "jpda"):
            rs = [run(dense(n_scans=30, n_targets=5, clutter=cl, seed=s),
                      name, clutter=cl, area=AREA["dense"], Pd=0.9)
                  for s in range(seeds)]
            # Fraction of clusters handed to the hard-assignment fallback.
            # Without this column the JPDA rows look like exact JPDA, and at
            # high clutter they are mostly not.
            fb = np.mean([r["stats"]["fallbacks"] / max(r["stats"]["clusters"], 1)
                          for r in rs])
            out[name] = (mean(rs, "ospa"), mean(rs, "coverage"),
                         mean(rs, "seconds"), fb)
        win = "JPDA" if out["jpda"][0] < out["gnn"][0] else "GNN"
        rows.append([f"{cl:.0f}", f"{out['gnn'][0]:.2f}", f"{out['jpda'][0]:.2f}",
                     win, f"{out['jpda'][1]:.3f}",
                     f"{out['jpda'][3]*100:.0f}%", f"{out['jpda'][2]:.2f}"])
    return table("Association quality vs clutter density",
                 ["false alarms/scan", "GNN OSPA", "JPDA OSPA", "better",
                  "JPDA coverage", "JPDA clusters over the size limit",
                  "JPDA s"], rows)


def exp_pd(seeds):
    """Sensitivity to detection probability."""
    rows = []
    for pd in (0.99, 0.9, 0.8, 0.6):
        rs = [run(dense(n_scans=30, n_targets=5, clutter=8.0, Pd=pd, seed=s),
                  "jpda", clutter=8.0, area=AREA["dense"], Pd=pd)
              for s in range(seeds)]
        rows.append([f"{pd:.2f}", f"{mean(rs,'ospa'):.2f}",
                     f"{mean(rs,'loc'):.2f}", f"{mean(rs,'card'):.2f}",
                     f"{mean(rs,'coverage'):.3f}"])
    return table("Sensitivity to detection probability (JPDA, 8 false alarms/scan)",
                 ["Pd", "OSPA", "OSPA loc", "OSPA card", "coverage"], rows)


def exp_cost(seeds):
    """The combinatorics, measured directly."""
    rows = []
    for n in (2, 4, 6, 8, 10):
        rs = [run(dense(n_scans=20, n_targets=n, clutter=15.0, seed=s),
                  "jpda", clutter=15.0, area=AREA["dense"], Pd=0.9)
              for s in range(seeds)]
        ev = np.mean([r["stats"]["events"] / max(r["stats"]["clusters"], 1)
                      for r in rs])
        mx = max(r["stats"]["max_cluster"] for r in rs)
        fb = sum(r["stats"]["fallbacks"] for r in rs)
        rows.append([str(n), f"{mx}", f"{ev:.1f}", str(fb),
                     f"{mean(rs,'seconds'):.2f}"])
    return table("JPDA cost vs target count (15 false alarms/scan)",
                 ["targets", "largest cluster", "events/cluster",
                  "fallbacks", "seconds"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = 2 if args.quick else 5
    levels = (2.0, 8.0, 20.0) if args.quick else (2.0, 8.0, 20.0, 40.0, 80.0)

    print(f"track-fusion benchmark, {seeds} seeds per cell")
    exp_imm(seeds)
    exp_turn_rate(seeds)
    exp_association(seeds)
    exp_clutter(seeds, levels)
    exp_pd(seeds)
    exp_cost(seeds)


if __name__ == "__main__":
    main()
