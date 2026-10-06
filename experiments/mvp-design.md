# Standalone MVP design

This document records the first experiment's assumptions before training.
They define a synthetic benchmark, not a validated production controller.

## Stack

Python 3.12 with uv and a committed lockfile. NumPy generates data; PyTorch
implements the MLP and calibration; scikit-learn supplies classification
metrics; Matplotlib renders reliability diagrams; pytest tests behavior.
JSON and CSV store configuration, data and results. No service or database is required.

## State (issue 03)

One snapshot of a replicated service. CPU and memory represent average
utilization across replicas; request rate is total incoming traffic; latency
is observed p95 request latency; error rate is the fraction of failed requests.
All observations refer to the same measurement interval.

| Feature | Type | Unit | Valid range |
| --- | --- | --- | --- |
| cpu | real | percent | 0–100 |
| memory | real | percent | 0–100 |
| request_rate | real | requests/second, total | 0–10,000 |
| latency | real | milliseconds, p95 | 0–10,000 |
| error_rate | real | fraction | 0–1 |
| replicas | integer | instances | 1–20 |

Reject missing, nonnumeric, nonfinite or out-of-range values and booleans.
Training support will be narrower than the valid input range so OOD states
can be valid inputs. Input validation is not OOD detection.

## Actions (issue 04)

The stable class order is `NO_ACTION`, `SCALE_IN`, `SCALE_OUT`, `RESTART`, `ESCALATE`.

- `NO_ACTION`: keep the current deployment when the policy finds no actionable condition.
- `SCALE_IN`: recommend removing one replica when load is low and at least two exist.
- `SCALE_OUT`: recommend adding one replica when resource and traffic pressure align.
- `RESTART`: recommend restarting one replica when errors are high without CPU overload.
- `ESCALATE`: request external review when observations conflict or model confidence is insufficient.

These are recommendations; the experiment does not simulate their effects.

## Ground truth (issue 05)

Policy `server-policy-v1` applies the following rules in order:

1. Error rate ≥ 0.10: `ESCALATE` if CPU ≥ 85%, otherwise `RESTART`.
2. CPU ≥ 75% or memory ≥ 85%: `SCALE_OUT` if latency ≥ 200 ms and
   requests/second/replica ≥ 300; otherwise `ESCALATE` for conflicting signals.
3. Replicas > 1, CPU ≤ 25%, memory ≤ 40%, latency ≤ 100 ms,
   requests/second/replica ≤ 100 and error rate ≤ 0.01: `SCALE_IN`.
4. Otherwise: `NO_ACTION`.

Thresholds are explicit synthetic assumptions, not operational recommendations.
They remain fixed throughout this experiment. Replicas above training support
are valid OOD inputs, not a request to implement deployment capacity limits.

## Data (issues 06–09)

Default: 20,000 states, seed 42, five equally sampled scenario families:
normal, low load, overload, high errors and mixed/conflicting signals.
Families are sampling regimes, not labels; the policy assigns every label.
Replicas range from 1 to 10; clean latency ≤ 1,000 ms and total rate ≤ 6,000/s.

Independent seeded Gaussian measurement noise has standard deviations of
3 CPU percentage points, 12 ms latency, 30 requests/s and 0.002 error fraction.
Observations are clipped to valid ranges. Memory and replica count are unchanged.
Labels come from the clean latent state; the oracle applied to noisy observations
is therefore an imperfect baseline. CSV retains both states, scenario and label.
Metadata records policy thresholds, generator version, seed and noise configuration.

Fixed stratified splits: 70% train, 15% validation, 15% test.
Validation is divided equally into model-selection and calibration subsets.
Normalization uses training data only; early stopping uses selection data;
temperature and escalation thresholds use calibration data. The test set
is not used to choose any of these. Row IDs are saved in a split manifest.

## Model, calibration and uncertainty (issues 11–17)

CPU MLP: 6 → 256 → 128 → 5 logits, with ReLU hidden activations.
Training standardizes features using training statistics, uses cross-entropy
and Adam, and selects the lowest validation loss with early stopping.
Defaults: batch size 256, learning rate 0.001, weight decay 0.0001,
120 maximum epochs, 12-epoch patience, minimum improvement 0.0001, one CPU thread.

Temperature scaling minimizes calibration-subset NLL with the base weights
frozen. Temperature is bounded to [0.1, 10]; an unsuccessful fit retains T=1.
Report before/after test metrics even if they worsen. Technique:
[Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html).

Sweep confidence thresholds 0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99 and 1.
For the deployed experiment wrapper, require a minimum threshold of 0.8 and
choose the highest non-escalated coverage with calibration-subset accuracy
≥ 0.98 and at least 30 accepted cases. The floor keeps uncertainty handling
enabled even when threshold zero meets the empirical target. If no candidate
meets the target, use the highest threshold and explicitly record target failure.
This validation selection target is not an integration success criterion.

The public call accepts a `State` or a dictionary:

```python
from system_one.decision import DecisionModel

model = DecisionModel.load("artifacts/mvp-v1/model.pt")
decision = model.decide(state)
```

`action` is the final recommendation. `confidence` is the largest calibrated
model probability, even when the threshold overrides the action to `ESCALATE`.
`probabilities` retains all five model probabilities. `escalation_reason`
distinguishes `low_confidence` from `predicted_escalate` (otherwise null).
An uncertainty override does not make the model's `P(ESCALATE)` equal to its
top probability or establish a calibrated probability that escalation is correct.

## Evaluation (issues 14–19)

Save raw model classification and final thresholded classification separately:
accuracy, per-action precision/recall/F1 and confusion matrices in stable class order.
Probability metrics use the full model probability vector before any override:
multiclass Brier = mean sum of squared class errors, NLL = mean negative log
probability of the true class, and top-label ECE using 15 equal-width bins.
Empty bins are omitted from ECE; exact confidence 1 belongs to the last bin.
Reliability plots show confidence versus empirical correctness; marker size
reflects bin population. ECE alone can conceal poorly calibrated small bins.

Coverage means the fraction receiving a non-`ESCALATE` action; accepted
accuracy is correctness within that fraction. Empty coverage has undefined
accuracy (JSON null). Save every wrong raw prediction with confidence ≥ 0.9,
including clean/observed state, row ID, true class, predicted class, final action,
confidence and the policy applied to the observation. Crossing a policy boundary
under measurement noise is recorded, not used to excuse a model error.

Use 100 seeded valid OOD states per scenario, excluded from all fitting:

- Extreme latency: 5,000–10,000 ms, otherwise moderate utilization.
- Unusual CPU/traffic: 98–100% CPU, 0–10 total requests/s and 0–20 ms latency.
- Near range limits: resource use ≥ 98%, traffic ≥ 9,000/s, latency ≥ 9,000 ms,
  error fraction ≥ 0.9 and 11–20 replicas.

OOD policy agreement measures imitation of the synthetic oracle; it does not
establish operational safety. In particular, this policy ignores extreme latency
alone when resource use is moderate. Softmax confidence is not an OOD detector.

Time 1,000 sequential single-state CPU decisions after 100 warmup calls.
Include validation, normalization, network forward pass, probabilities and
decision selection; exclude model loading and file/JSON I/O. Save mean/p50/p95/p99.
This is initial latency evidence, not the complete resource benchmark of issue 23.

## Reproduction and boundaries

Each new output directory stores configuration, data/splits, training history,
model/preprocessing/calibration, metrics, failure cases, OOD probes, reliability
plot and environment metadata including dependency versions, Git status and
source hashes. Refuse to overwrite an existing run directory.

Seeds and deterministic PyTorch algorithms target reproducibility in the locked
CPU environment; identical results across releases or hardware are not promised.
[PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

History, temporal inputs, multiple heads, full resource benchmarking and formal
integration success criteria remain later issues. The MVP's existence does not
authorize 5G integration, and an empirical issue outcome may remain unmet even
when its experiment code exists.
