# system-one-5g

Experimental System One decision model.

The first goal is **not 5G integration**. The project starts as a standalone decision-model experiment and moves into the 5G/Open5GS testbed only after the model is useful.

## Goal

Build a small model that:

- receives structured state
- chooses from predefined actions
- returns a probability for every action
- produces calibrated confidence
- knows when to escalate instead of guessing
- runs with low inference latency

## First experiment

The first decision problem ([issue 02](https://github.com/khvedela/system-one-5g/issues/2)) is:

- Observe one snapshot of a synthetic replicated server service.
- The snapshot contains CPU use, memory use, request rate, latency, error rate and replica count.
- Choose exactly one action: `NO_ACTION`, `SCALE_IN`, `SCALE_OUT`, `RESTART` or `ESCALATE`.
- Recommend an action without executing changes to a real service.
- A fixed, documented ground-truth policy defines the correct action for each generated state.
- A good decision matches that policy while supporting healthy service operation and avoiding unnecessary changes.
- Explicitly represent noisy and ambiguous states, and allow escalation when confidence is insufficient.
- Evaluate held-out decision quality, probability calibration, uncertainty handling and inference latency reproducibly.

Example state:

```text
cpu
memory
request_rate
latency
error_rate
replicas
```

Example actions:

```text
NO_ACTION
SCALE_IN
SCALE_OUT
RESTART
ESCALATE
```

Target interface:

```python
decision = model.decide(state)
```

Illustrative result (not a measured prediction):

```json
{
  "action": "SCALE_OUT",
  "confidence": 0.91,
  "probabilities": {
    "NO_ACTION": 0.03,
    "SCALE_IN": 0.01,
    "SCALE_OUT": 0.91,
    "RESTART": 0.03,
    "ESCALATE": 0.02
  },
  "escalation_reason": null
}
```

## Roadmap

1. Define the decision problem
2. Generate synthetic training data
3. Build a simple decision model
4. Measure accuracy and calibration
5. Add uncertainty handling
6. Test unseen states
7. Add temporal and multi-head decisions
8. Integrate with the 5G testbed only after the standalone model is proven

## Run the standalone MVP

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --locked
uv run pytest
uv run system-one run --config experiments/mvp.json --output artifacts/my-first-run
uv run system-one decide --model artifacts/my-first-run/model.pt --state experiments/example-state.json
```

Use a new output directory for every run. Training runs on CPU. The default
experiment generates 20,000 seeded states, trains a 6 → 256 → 128 → 5 MLP,
fits temperature scaling and selects an escalation threshold on validation data.
The held-out test set is used only for evaluation.

```python
from system_one.decision import DecisionModel
from system_one.state import State

model = DecisionModel.load("artifacts/my-first-run/model.pt")
decision = model.decide(State(cpu=92, memory=80, request_rate=1800,
                             latency=350, error_rate=0.01, replicas=3))
print(decision.as_dict())
```

`confidence` is the highest calibrated model probability. A low-confidence
override returns `ESCALATE` without rewriting the probabilities. The additional
`escalation_reason` field identifies a threshold override or a predicted escalation.

Each run saves CSV data with clean/noisy states, split row IDs, configuration,
weights and preprocessing, training history, classification/calibration metrics,
a reliability plot, OOD responses, confident errors and environment metadata.
Generated artifacts stay local under the ignored `artifacts/` directory.

See [experiment design](experiments/mvp-design.md) for feature units, fixed policy,
stack, noise assumptions and metric definitions, and [measured results](experiments/mvp-results.md)
for the first run and its limitations.

## Broader telemetry and usability work

The metric layer discovers every exposed series from configured NF Prometheus
endpoints, container exporters, kube-state-metrics and Docker stats. The pinned
[metric catalog](experiments/metrics-catalog.json) contains 127 Open5GS families,
100 cAdvisor metrics and 336 kube-state-metrics entries, with source provenance.
Models can train on an arbitrary-width numeric metric schema, with training-only
imputation/normalization, separate incident-group holdouts and explicit escalation
for missing metrics, schema drift and values outside training support.

See [telemetry workflow](experiments/telemetry.md) for collection, feature discovery,
labeled dataset assembly, training and read-only shadow logging. `prometheus-client`
provides the standard exposition parsers; the Docker SDK retrieves full resource
stats rather than only CLI summaries. Both are pinned in `uv.lock`.

```sh
uv run system-one run --config experiments/expanded.json --seed 142 --output artifacts/expanded-run
uv run system-one robustness --reference artifacts/expanded-run --output artifacts/guarded-run
uv run system-one decide --model artifacts/guarded-run/model.pt --state experiments/example-state.json
```

The robustness experiment evaluates support guards, broader combinations,
action-effect simulation, latency/resource cost and the [readiness criteria](experiments/readiness.json).
See [usability results](experiments/usability-results.md) for cross-seed findings.

## Current status

The standalone experiments and read-only telemetry workflow are implemented.
Support checks catch the tested extreme OOD cases; broader training improves
generalization. All three expanded runs still fail provisional readiness criteria,
and NF training awaits real observations with reviewed action labels. Network
actions remain disabled. History, temporal modeling and multiple decision heads
remain later experiments; GitHub issue statuses have not been changed.
