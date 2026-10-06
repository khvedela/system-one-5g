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

Use a synthetic server-management environment.

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

Expected result:

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
  }
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

## Current status

Planning / research.
