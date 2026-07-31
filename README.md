# track-fusion

Multi-target tracking built from the filter up: an IMM estimator over a bank of
motion models, joint probabilistic data association, sequential track scoring,
and OSPA-based evaluation that charges you for losing a target in the same
units it charges you for misplacing one.

[![CI](https://github.com/asp53826/track-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/asp53826/track-fusion/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![Runtime dependencies](https://img.shields.io/badge/runtime_dependencies-2-2ea44f?style=flat-square)
![Tests](https://img.shields.io/badge/tests-67-6f42c1?style=flat-square)
[![License: MIT](https://img.shields.io/badge/license-MIT-f5c518?style=flat-square)](LICENSE)

A tracker is three problems wearing a trenchcoat — estimate where a target is,
decide which measurement belongs to which target, and decide which targets
exist at all. The third one is where trackers actually fail, and it's the one
that gets left out of tracking demos.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q                 # 67 tests
python bench/benchmark.py           # every table below, ~6 minutes
python bench/benchmark.py --quick   # a faster sanity check
```

## What's in it

| piece | file | notes |
|---|---|---|
| Kalman filter | [`tf/kalman.py`](tf/kalman.py) | Cholesky throughout; no explicit inverses |
| Motion models | [`tf/models.py`](tf/models.py) | Constant velocity and coordinated turn, via `sinc` |
| IMM estimator | [`tf/imm.py`](tf/imm.py) | Mixing, mode-matched filtering, moment-matched output |
| Gating | [`tf/gating.py`](tf/gating.py) | Chi-square ellipsoidal |
| JPDA | [`tf/assoc/jpda.py`](tf/assoc/jpda.py) | Clustering, exact joint-event enumeration, bounded fallback |
| GNN | [`tf/assoc/gnn.py`](tf/assoc/gnn.py) | Hungarian assignment — the baseline JPDA has to beat |
| Track lifecycle | [`tf/track.py`](tf/track.py) | Wald sequential test on a log-likelihood-ratio score |
| Metrics | [`tf/metrics.py`](tf/metrics.py) | OSPA with localisation/cardinality split, identity metrics |

Association is swappable and everything else is held fixed, so a JPDA-vs-GNN
row differs in exactly one thing. That's the only reason the comparisons below
mean anything.

## The filter bank

A single Kalman filter has to commit to one process noise, and that choice is
a straight trade. Tight noise tracks a straight leg well and lags badly through
a turn; loose noise survives the turn and is noisy everywhere else. The IMM
runs a bank — constant velocity, and coordinated turns at ±6°/s — and lets the
measurements weight them.

One manoeuvring target, no clutter to speak of, no association difficulty. The
only variable is the filter:

| filter | OSPA | OSPA loc | RMSE m | coverage |
|---|---|---|---|---|
| single CV filter | 39.75 | 26.92 | 30.3 | 0.893 |
| IMM (CV + 2 turns) | **36.57** | **14.40** | **18.6** | **0.960** |

**Localisation error drops 47%** and coverage goes from 0.893 to 0.960. The
localisation column is the honest one here: the IMM's whole job is to be
accurate through the manoeuvre, and that is what it does.

### Where the IMM loses

The bank only helps when the manoeuvre is inside it. Sweeping the turn rate,
with the modes fixed at ±6°/s:

| turn °/s | single CV | IMM | OSPA change |
|---|---|---|---|
| 0 | **21.59** | 31.42 | **−45.5%** |
| 3 | 36.64 | **32.40** | +11.6% |
| 6 | 39.75 | **36.57** | +8.0% |
| 12 | **42.41** | 49.33 | **−16.3%** |

Both ends are losses, for opposite reasons, and both are the model being honest
about its assumptions:

- **At 0°/s the target never manoeuvres**, so the turn modes are pure cost.
  They hold prior probability mass, they carry 40x the process noise, and they
  drag the combined estimate. Paying 45% for insurance you don't use is the
  standard IMM trade and the reason you don't reach for one by default.
- **At 12°/s the turn is outside the bank.** Nothing in it can represent a
  12°/s turn, so the filter falls back on the manoeuvre modes' process noise
  rather than their dynamics — which is roughly a badly-tuned CV filter. A
  wider bank fixes this and makes the 0°/s case worse.

The useful band here is 3–6°/s. That's not a limitation of IMM in general; it's
the specific bank in [`tf/models.py`](tf/models.py), and the sweep is how you'd
size a real one against an expected threat set.

## Association

JPDA doesn't pick an assignment — it averages over every consistent one.
"Consistent" is load-bearing: one measurement has one source, one target
produces at most one detection. That constraint is the entire difference
between JPDA and running independent PDA filters that both happily consume the
same measurement.

Two targets on converging tracks that cross mid-run, light clutter:

| assoc | OSPA | OSPA loc | OSPA card | false trk/scan | coverage |
|---|---|---|---|---|---|
| GNN | **28.73** | 14.70 | **16.14** | **0.21** | 0.948 |
| JPDA | 39.55 | **13.76** | 29.86 | 0.56 | 0.945 |

**JPDA is more accurate and worse overall**, and that split is the interesting
result. Near the crossing both measurements are plausible for both tracks;
averaging over the ambiguity beats committing and being wrong half the time,
and JPDA's localisation is better on 12 of 12 seeds (paired difference
0.76 ± 0.40).

But soft association also keeps marginal tracks alive. A spurious track that
picks up partial weight from clutter every scan never looks dead, so JPDA
sustains 2.7x the false tracks, and OSPA charges for those in the cardinality
column. Localisation is a third of JPDA's error here; cardinality is
three-quarters of it.

This is the argument for OSPA over RMSE in one table. On position error alone
JPDA wins and you ship it.

### When JPDA is actually worth it

Sweeping clutter density, five manoeuvring targets in a shared volume:

| false alarms/scan | GNN OSPA | JPDA OSPA | better | JPDA coverage | clusters over size limit | JPDA s |
|---|---|---|---|---|---|---|
| 2 | **41.72** | 49.68 | GNN | 0.908 | 2% | 0.33 |
| 8 | **31.92** | 34.39 | GNN | 0.887 | 3% | 0.56 |
| 20 | **34.17** | 37.98 | GNN | 0.853 | 4% | 1.83 |
| 40 | **38.33** | 41.89 | GNN | 0.816 | 3% | 5.06 |
| 80 | 49.09 | **47.75** | **JPDA** | 0.737 | 2% | 16.48 |

JPDA only overtakes at 80 false alarms per scan, and wins by 2.7%. Everywhere
below that, hard assignment is both better and cheaper — its commitments are
mostly right, and it doesn't pay for the ones it gets wrong the way JPDA pays
for the false tracks it keeps alive.

The honest summary is that **JPDA is not a default**. It's a tool for the
regime where hard assignment starts committing to genuinely wrong
measurements, and on this scenario set that regime starts later than the
literature's enthusiasm suggests.

The last column matters for reading the rest of the row: it's the fraction of
clusters too large to enumerate exactly, which fall back to hard assignment.
At 2–4% the JPDA rows are genuinely JPDA. If that number were 60% the
comparison would be measuring nothing, which is why it's in the table.

### The cost

Joint events are combinatorial in the number of mutually-gating tracks, so
clustering is what makes JPDA tractable at all — twenty isolated tracks is
twenty trivial problems, twenty mutually-gating tracks is not. Past a cluster
size limit it falls back to hard assignment rather than truncating the sum:

| targets | largest cluster | events/cluster | fallbacks | seconds |
|---|---|---|---|---|
| 2 | 23 | 14.1 | 48 | 0.59 |
| 4 | 24 | 13.5 | 62 | 0.70 |
| 6 | 28 | 9.1 | 61 | 0.82 |
| 8 | 31 | 12.6 | 80 | 1.04 |
| 10 | 30 | 4.6 | 87 | 1.18 |

A pure event cap would be worse than useless: it truncates the joint sum and
then renormalises over whichever events happened to be enumerated first, which
is a biased answer wearing an exact answer's clothes. Falling back to a
different algorithm and counting how often is the version you can defend.

## Track management

This is where trackers actually die, and it's the part that isn't in the
papers. Two bugs found here during development, both of which produced
plausible-looking output:

**M-of-N confirmation builds tracks out of pure clutter.** With no targets at
all and 6 false alarms per scan, M-of-N confirmed 165 tracks over 40 scans. The
mechanism: a newly initiated track has an unknown velocity, so its covariance
is enormous, so its gate is enormous, so it nearly always contains *something*
— and M-of-N counts that as a detection. Counting hits cannot distinguish a
target from a track that is eating clutter.

The fix is to score quality instead of counting hits. Each scan adds a
log-likelihood ratio of target-present against everything-in-the-gate-is-clutter:

```
dS = ln[ (1 - Pd·Pg) + (Pd/λ) · Σ_j L_j ]
```

A real track has a tight covariance, so its predicted density is concentrated
and each hit is strong evidence. A clutter track's density is spread thin over
a huge gate, so the same hit is weak evidence and the misses charge against it.
Confirmation and deletion are then a Wald sequential test with bounds set from
the two error rates you actually care about — how often a clutter track gets
confirmed, and how often a real one gets discarded.

**A miss counter that meant the wrong thing silently destroyed JPDA.** Deleting
a track after 4 consecutive scans with `beta0 > 0.5` looks reasonable and is
fine for GNN. Under JPDA in dense clutter, association weight spreads across
every gated measurement, the true measurement's share falls below a half, every
scan scores a miss, and correctly-updated tracks get deleted. Coverage went to
**0.000 at 80 false alarms per scan** — JPDA looked catastrophically worse than
GNN, and the association code was fine. The miss counter now means "the gate
was empty", which is the only thing it should ever have meant; track quality is
the score's job.

Both of these are why the metric set includes coverage and false-track rate and
not just position error. Neither bug moves RMSE much. Both destroy the tracker.

## Numerical details worth the trouble

**The coordinated-turn matrix goes through `sinc`.** The position terms are
`sin(wT)/w` and `(1-cos(wT))/w`, both 0/0 as `w → 0`, and the naive forms are
bad well before the limit — at `wT = 1e-7`, `1 - cos(wT)` cancels down to about
two significant digits. Written as

```
sin(wT)/w      = T · sinc(wT/π)
(1-cos(wT))/w  = (w T²/2) · sinc(wT/2π)²
```

both are exact at `w = 0` and there is no threshold to tune. The usual fix is a
small-angle branch, which works but puts a tunable constant in the middle of
the state transition and a discontinuity at its cutoff.

**Cholesky, not inverses.** Gating, the Gaussian likelihood and the Kalman gain
all want `S⁻¹`, and none of them should form it. `cho_factor` once per
innovation, `cho_solve` for each use, and the log-determinant comes off the
same factor.

**The predictive density of a manoeuvring track is a mixture, not a Gaussian.**
Association scores measurements against `Σ_j μ_j N(z; ẑ_j, S_j)`, not against
the moment-matched combination. Using the latter is a common shortcut that
under-weights measurements sitting out on the turning modes — exactly when
association needs to work. Gating still uses the moment-matched covariance,
which is provably conservative, so it can't reject a measurement the mixture
would have accepted.

## Tests

67 tests, `pytest`. The ones that earn their keep:

- **An IMM whose modes are all identical must reproduce a plain Kalman filter,
  step for step.** Mixing and combination have to be the identity in that case,
  so anything wrong in either shows up immediately.
- **The mixture likelihood integrates to 1** over the plane, on a 601×601 grid.
- **JPDA marginals against a brute-force re-derivation** — every assignment
  enumerated by hand in the test and marginalised independently.
- **Exclusivity**: two tracks gating one measurement must have betas summing to
  at most 1. Independent PDA filters fail this and nothing else catches it.
- **The turn model closes a circle**: 36 steps of 10° returns to the initial
  state exactly.
- **OSPA charges a spurious track exactly what it charges a dropped one**, at
  equal cardinality.

Three tests in here assert the *negative* result — that GNN produces fewer
false tracks than JPDA, and that JPDA doesn't win on the crossing. Tests that
only encode the flattering direction are how a benchmark quietly stops being a
measurement.

## What this doesn't do

- **Truth is generated from the same models the filter uses.** So these numbers
  measure estimation and association quality, not model mismatch, which is the
  thing that hurts most on real data. Sensor-frame measurements
  (range/bearing with an EKF or UKF), real clutter statistics, and recorded
  ADS-B or radar tracks are all missing and all matter.
- **Measurements are Cartesian position.** Real radar gives range, bearing and
  usually Doppler, in a frame where the noise is not remotely isotropic.
- **No MHT.** JPDA is a single-hypothesis-per-scan method; multiple hypothesis
  tracking defers the decision across scans and is the standard answer where
  JPDA's coalescence hurts.
- **Single sensor.** No registration, no bias estimation, no track-to-track
  fusion.

MIT licensed.
