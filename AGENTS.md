# AGENTS.md

## Purpose

This repository is an experimental System One decision-model project.

Read `GOAL.md` and the relevant GitHub issue before making changes.

GitHub issues are the execution roadmap. Work on **one issue at a time** unless the user explicitly asks otherwise.

## Main rule

Do not jump ahead.

The current goal is to prove a standalone decision model first. 5G/Open5GS/OAI/Kubernetes integration comes only after the standalone experiment passes its success criteria.

## Engineering approach

Prefer the simplest implementation that can answer the current research question.

For the first model:

- use structured numeric inputs
- use a small MLP
- output logits for predefined actions
- convert logits to probabilities
- keep `ESCALATE` available for uncertainty
- keep training and evaluation reproducible

Do not introduce an LLM, transformer, agent framework, vector database or distributed system unless a later experiment demonstrates that it is necessary.

## Decision model contract

Preserve a clean boundary similar to:

```python
decision = model.decide(state)
```

The result should expose:

```text
action
confidence
probabilities
```

Model internals may change later, but callers should not need to know which model architecture is being used.

## Data rules

- Keep synthetic data generation deterministic when given a seed.
- Keep train, validation and test splits separate.
- Never train on the test set.
- Do not silently change the ground-truth policy to improve model scores.
- Store enough metadata to reproduce a dataset.
- Make noisy and ambiguous states explicit rather than hiding them.

## Evaluation rules

Do not report only accuracy.

Track the metrics required by the active issue, including when relevant:

- accuracy
- precision
- recall
- F1
- confusion matrix
- Brier score
- Negative Log-Likelihood
- Expected Calibration Error
- inference latency
- high-confidence error cases

Treat confident wrong decisions as first-class failures.

## Calibration

Probability quality matters.

A probability such as `0.8` should have empirical meaning. Prefer established calibration techniques before inventing new training methods.

Start with temperature scaling unless the experiment specifically requires another method.

## Uncertainty

The model must be allowed to say, effectively, "I am not confident."

Do not force every uncertain state into a normal action.

Use `ESCALATE` or the uncertainty mechanism defined by the active issue.

## Scope control

When implementing an issue:

1. read the issue
2. identify the smallest change that satisfies it
3. implement only that scope
4. add or update tests
5. run the relevant checks
6. summarize what changed and any important result

Do not bundle unrelated refactors with issue work.

If you notice future improvements, mention them instead of implementing them unless they are necessary for the current task.

## Code quality

Favor:

- small modules
- explicit types where useful
- simple interfaces
- deterministic behavior
- readable names
- reproducible scripts
- configuration over hidden constants

Avoid premature abstractions.

Comment only where the reason is not obvious from the code.

## Repository structure

Do not invent a large directory hierarchy before it is needed.

A reasonable structure may emerge as:

```text
src/
  data/
  model/
  calibration/
  evaluation/

tests/
experiments/
```

Create directories only when code actually needs them.

## Dependencies

Keep dependencies minimal.

Before adding a package:

- check whether the standard library or an existing dependency is enough
- prefer established ML/scientific packages
- avoid framework dependencies for small utilities

Document any new dependency and why it is needed.

## Experiments

Experiments must be reproducible.

Record where relevant:

- random seed
- dataset version/configuration
- model configuration
- training configuration
- metrics
- hardware/device
- result artifacts

Do not overwrite useful experiment results without a reason.

## Testing

Add tests for behavior introduced by the current issue.

At minimum, test important deterministic logic such as:

- state validation
- ground-truth policy
- dataset generation
- probability normalization
- decision selection
- uncertainty thresholds

Run the relevant test suite before considering work complete.

## GitHub issues

The numbered GitHub issues define the intended sequence.

Important boundaries:

- Issues 01–10: problem and dataset foundation
- Issues 11–19: first model, evaluation, calibration and uncertainty
- Issues 20–23: history, temporal modeling, multi-head decisions and performance
- Issues 24–26: success criteria and standalone decision
- Issue 27+: 5G integration planning

Do not start 5G integration before the decision in issue 26 unless the user explicitly changes the plan.

## Documentation

Keep documentation concise and factual.

When results exist, record measured numbers rather than claims such as "fast", "accurate" or "well calibrated".

Update `README.md` or experiment documentation when project behavior materially changes.

## When requirements are unclear

Prefer the interpretation that:

- keeps the experiment measurable
- preserves reproducibility
- changes the least code
- avoids adding architectural complexity

If a choice would materially alter the research design, surface the tradeoff instead of silently choosing a complex direction.
