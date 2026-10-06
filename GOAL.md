# GOAL

## Mission

Build a small, local **System One decision model** that maps structured state to a predefined action with calibrated probabilities.

The project must prove the decision model works in a standalone environment **before** integrating it with the 5G/Open5GS testbed.

## Core contract

Input:

```text
structured state
```

Output:

```text
action
confidence
probability for every allowed action
```

Target interface:

```python
decision = model.decide(state)
```

Initial actions:

```text
NO_ACTION
SCALE_IN
SCALE_OUT
RESTART
ESCALATE
```

## First environment

Start with a synthetic server-management problem using features such as:

- cpu
- memory
- request_rate
- latency
- error_rate
- replicas

The first dataset must have known ground truth so model behavior can be measured objectively.

## What success means

The standalone prototype should demonstrate:

1. correct decisions on held-out data
2. calibrated probabilities
3. explicit uncertainty handling
4. low inference latency
5. reproducible training and evaluation
6. useful behavior on noisy and unseen states

Accuracy alone is not enough. A highly confident wrong decision is an important failure.

## Development path

### Phase 1 — Define the problem
Define state, actions and ground-truth decision rules.

### Phase 2 — Build the dataset
Generate synthetic states, labels, realistic noise and fixed train/validation/test splits.

### Phase 3 — Build the baseline model
Start with a small MLP. Do not begin with an LLM or transformer.

### Phase 4 — Evaluate decisions
Measure classification quality, calibration and failure cases.

### Phase 5 — Improve uncertainty handling
Add calibration, ESCALATE behavior and out-of-distribution testing.

### Phase 6 — Add context only when justified
Test action history, temporal state and multiple decision heads.

### Phase 7 — Decide if the model is useful
Define explicit success criteria and compare results against them.

### Phase 8 — 5G integration
Only after the standalone model passes the success criteria, map the same decision interface to Open5GS/OAI/Kubernetes state.

## Non-goals for the first prototype

Do not:

- integrate with the 5G testbed yet
- build an LLM
- add text generation
- use a transformer without evidence it is needed
- optimize for model size before the baseline works
- add actions beyond the defined action space without a concrete reason
- hide uncertainty by always forcing a decision

## Research question

Can a small decision-focused model produce fast, useful and calibrated action probabilities from structured system state?

The later 5G question is:

Can the same decision architecture be useful for autonomous network management and security?
