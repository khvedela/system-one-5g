# Usability iteration

This iteration implements source discovery, arbitrary-width metric training,
support guards, broader synthetic coverage, a toy action simulator, audit logging
and a machine-readable readiness gate. It does not establish production readiness.

## Reference preservation and policy review

The original dataset, model and [MVP report](mvp-results.md) remain unchanged.
`server-policy-v1` remains the oracle; its thresholds were not adjusted to improve scores.
Policy limitations remain explicit: isolated extreme latency can return `NO_ACTION`,
and high CPU with low latency can trigger review even when service latency is healthy.
The policy also recommends actions without consulting actual deployment capacity.
These require domain review and an explicitly versioned policy before network control.
The user's broader telemetry request is recorded as a read-only scope expansion in GOAL.md.

## Support checks and broader training

The guarded reference artifact is `artifacts/robustness-v1/model.pt`.
Bounds use only reference training minima/maxima, padded by 0.1 training standard
deviations. This is a simple range-support guard, not a general OOD detector.
It escalated all 300 probes across the three OOD families with zero additional
guard escalations among the original 3,000 test states. Extreme-latency policy
agreement remains zero because review differs from the oracle's `NO_ACTION`.
The raw network probabilities still contain the original confident mistakes.

The reference model's accepted accuracy was only 85.48% on 1,000 independent
broad combinations/policy-boundary probes, with 48.9% coverage. The expanded
dataset is explicitly versioned `server-data-v2`: 30,000 samples with 40% original
scenario families, 40% broad combinations and 20% near-policy-boundary cases.
It keeps clean latent labels, explicit measurement noise and the unchanged v1 policy.
Seeds 142–144 are independent of the original seed 42. The fixed broader probe
stream remains separate from training generation and was not used for fitting,
temperature selection or confidence-threshold selection.

| Training seed | Raw test accuracy | Accepted test accuracy | Test coverage | Broader accepted accuracy | Broader coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| 142 | 93.96% | 98.50% | 74.02% | 98.79% | 74.60% |
| 143 | 94.73% | 98.89% | 74.38% | 98.79% | 74.60% |
| 144 | 95.00% | 98.95% | 74.13% | 99.33% | 74.50% |

Raw accuracy is lower than the original easier sampling distribution; it is not a
like-for-like regression. Broader accepted accuracy improves substantially.
All three guarded models escalated all 300 OOD probes. In-range novel combinations
can still defeat a range guard. Test ECE after calibration ranges from 0.00806
to 0.00937; calibration does not improve every seed's metrics. Raw errors with
confidence ≥ 0.9 remain 0.54–0.80% of confident predictions.

## Action-effect simulation and performance

`service-effects-v1` compares a learned controller, the fixed rule controller and
no action over load bursts, low load, a restart-recoverable fault and background CPU.
Three workload seeds (42–44), 60 steps per scenario, identical exogenous traffic
for each controller. Capacity is 500 requests/s/replica; replicas are bounded to 1–10.
Restart immediately clears the modeled fault; escalation incurs review cost but
does not invent human remediation. Cost = latency_ms/200 + 20*error_fraction +
0.1*replicas + action cost (scale 0.2, restart 0.5, review 1).

The expanded models match the rule controller's mean toy cost and achieve 30.27%
of the no-action controller's cost across these scenarios. They have not demonstrated
a benefit over the rules. The simulator's assumptions, particularly immediate restart
recovery and review cost, must be validated; these numbers are not real service results.
Artifacts also measure CPU mean/p95/p99 latency, parameter/file size and whole-process
peak RSS. RSS includes libraries and data, not just model allocation. Timing is local
and varies under concurrent load.

## Readiness decision

[readiness.json](readiness.json) defines provisional standalone thresholds for
accuracy/F1, calibration, accepted accuracy/coverage, confident mistakes, OOD
escalation, latency, simulator benefit and the need for real labeled telemetry.
These are experiment criteria, not validated operational risk tolerances.
All three expanded runs **fail readiness**: final action accuracy/F1 and accepted
accuracy miss strict limits, confident errors exceed 0.1%, and no real labeled NF
telemetry has been evaluated. The next step remains `improve_standalone_first`.
Automatic network control is disabled regardless of the current gate result.

Telemetry collection and schema-based training are available as described in
[telemetry.md](telemetry.md). Definitions alone cannot train a meaningful NF model.
The current source inventory enables collection without inventing missing metrics;
reviewed NF observations/action labels remain necessary for operational evidence.

## Reproduce

```sh
uv sync --locked
uv run pytest
uv run system-one robustness --reference artifacts/mvp-v1-repeat --output artifacts/reference-guard-reproduction
uv run system-one run --config experiments/expanded.json --seed 142 --output artifacts/expanded-reproduction142
uv run system-one robustness --reference artifacts/expanded-reproduction142 --output artifacts/guarded-reproduction142
# Repeat with seeds 143 and 144 in fresh output directories.
```

If the local MVP artifact is absent, first reproduce it using the README commands.
[expanded-results.json](expanded-results.json) stores the cross-seed summary.
Full classifications, cases, sources, simulator outcomes, memory/latency measurements,
readiness checks and saved models remain under the ignored `artifacts/` run directories.
