# Container and network-function telemetry

## Pinned catalog

[metrics-catalog.json](metrics-catalog.json) contains the extracted definitions,
types, descriptions, label names, histogram definitions, source locations and hashes.
Sources were fetched on 2026-10-06:

| Source | Release | Catalog entries |
| --- | --- | ---: |
| Open5GS | [v2.8.0](https://github.com/open5gs/open5gs/releases/tag/v2.8.0) | 127 NF-qualified families |
| cAdvisor | [v0.60.6](https://github.com/google/cadvisor/releases/tag/v0.60.6) | 100 documented metrics |
| kube-state-metrics | [v2.20.0](https://github.com/kubernetes/kube-state-metrics/releases/tag/v2.20.0) | 336 documented entries |

Open5GS counts: AMF 24, SMF 29, UPF 11, PCF 5, MME 3, HSS 39, PCRF 16.
The other configured NFs have no `src/<nf>/metrics.c` at this release.
Source-defined families are not a promise that all label instances already exist
in a running NF. Slice/DNN/cause/QFI series can depend on activity.
The extractor preprocesses the C macros and verifies that every literal metric
declaration was extracted; it fails on unsupported patterns rather than returning
a silently incomplete inventory. Regeneration requires git and a C preprocessor (`cc`).

To reproduce the source inventory:

```sh
git clone --depth 1 --branch v2.8.0 https://github.com/open5gs/open5gs.git artifacts/sources/open5gs-v2.8.0
git clone --depth 1 --branch v0.60.6 https://github.com/google/cadvisor.git artifacts/sources/cadvisor-v0.60.6
git clone --depth 1 --branch v2.20.0 https://github.com/kubernetes/kube-state-metrics.git artifacts/sources/kube-state-metrics-v2.20.0
uv run system-one catalog --open5gs artifacts/sources/open5gs-v2.8.0 \
  --cadvisor artifacts/sources/cadvisor-v0.60.6 \
  --kube-state-metrics artifacts/sources/kube-state-metrics-v2.20.0 \
  --output artifacts/regenerated-catalog.json
```

Release tags may have aliases (cAdvisor's recorded exact tag is `lib/v0.60.6`);
the commit hash in the catalog is the authoritative source identity.
Upstream source licenses remain applicable; catalog entries link to their origins.

## Collection coverage

Configure the actual URLs in a copy of [telemetry.example.json](telemetry.example.json).
The sample DNS names are placeholders, not discovered endpoints. They cover the
seven NFs with metric definitions, cAdvisor and kube-state-metrics. Remove targets
for NFs/exporters that are not deployed and add every available target; there is
no metric-name allowlist. Docker collection is optional and disabled in the example.

- NF `/metrics` on the configured port (commonly 9090): all returned samples,
  including counters, gauges, histogram buckets/sums/counts, summaries and untyped metrics.
- Container resource exporter: CPU usage/throttling/scheduling, memory RSS/cache/
  working set/swap, block and filesystem I/O, network bytes/packets/errors/drops,
  processes, threads, file descriptors, sockets, pressure and available hardware metrics.
  Exact coverage depends on runtime, kernel, exporter flags and build features.
- kube-state-metrics: all configured object-state metrics, including resource
  requests/limits, readiness, restarts, OOM reasons, pod conditions and deployment state.
- Docker Engine: all returned nested numeric stats plus lifecycle/resource-limit
  fields; original stats are retained. This is an alternative to container exporters,
  not a Kubernetes-specific dependency. Container environment variables are not saved.
- Add node exporters or other Prometheus endpoints to collect their full exposed metrics too.

Kubernetes resource usage and Kubernetes object state are different sources;
`kubectl top`/the resource Metrics API alone does not provide the complete set.
See [resource metrics pipeline](https://kubernetes.io/docs/tasks/debug/debug-cluster/resource-metrics-pipeline/),
[object-state metrics](https://kubernetes.io/docs/concepts/cluster-administration/kube-state-metrics/)
and [CRI container metrics](https://kubernetes.io/docs/reference/instrumentation/cri-pod-container-metrics/).
For kubelet/CRI/cAdvisor endpoints, use the endpoints supported by the cluster version;
an authenticated API-server node proxy URL can be configured like any other target.
Supply a token via `bearer_token_env` and an optional CA path via `ca_file`.
TLS verification stays enabled. URLs must not contain embedded credentials.

```sh
uv run system-one collect --config my-telemetry.json --output artifacts/capture-001
# Capture again later to obtain counter rates; keep both snapshots.
uv run system-one collect --config my-telemetry.json --output artifacts/capture-002
uv run system-one features --snapshot artifacts/capture-002/snapshot.json \
  --previous artifacts/capture-001/snapshot.json --output artifacts/features-002.json
```

Capture saves raw exposition, parsed samples, inventory, source timestamps,
types/labels/HELP/unit metadata, raw checksums, failures and full Docker stats.
Counter names are preserved as exposed even when the parser internally normalizes them.
The collector negotiates text/OpenMetrics; unsupported encodings fail explicitly.
It cannot obtain metrics that an NF/kernel/exporter does not expose; enable the
appropriate instrumentation rather than fabricate zero values.

Unavailable sources cause exit status 1 while preserving successful captures and
failure metadata. No containers are started, restarted or scaled by these commands.
The local Docker daemon was unavailable during development; no live cluster/NF
capture is claimed. HTTP collection and Docker extraction were verified with fixtures.

## Every observed metric can become a model feature

Feature identity includes source, metric name and every label. All series are retained.
Counters additionally have rate/second and reset indicators. A first sample, reset,
missing or nonfinite value has an unavailable rate/value, not an invented zero.
Histograms retain all buckets/sums/counts and their rates; quantile estimation and
domain-specific aggregation are not silently performed. Prometheus summary quantiles
must not be treated as aggregatable histogram buckets.

Freeze the schema using only training captures:

```sh
uv run system-one schema --snapshots artifacts/capture-001/snapshot.json \
  artifacts/capture-002/snapshot.json --output artifacts/schema.json
```

The model input size is determined by that schema, not fixed to six. Each metric
feature receives a missing indicator; median imputation and normalization use
training observations only. The feature order/hash and preprocessing are saved
with the weights. Missing metrics, new metrics or values outside training support
cause an explicit escalation. Unknown metrics remain in the raw capture and require
schema review/retraining, rather than silently changing the current network weights.

Per-series identity means changing pod/container IDs or label cardinality can cause
schema drift. Before operational deployment, define stable per-NF/entity aggregation
and action targets using actual traces; the raw archive preserves the information
needed to do this. A global metric catalog is not an action-target mapping.

## Labeled dataset and grouped training

Source code provides definitions, not observations or correct actions. Labels must
come from reviewed incidents, controlled tests or an explicitly versioned policy.
Do not auto-label every NF using the synthetic server oracle.

Create an annotations JSON with `label_origin` and `records`. Each record has
`snapshot`, optional `previous`, `action` and `group_id`. Snapshot paths are relative
to the annotations file. Example structure only (these are not real labels):

```json
{
  "label_origin": "reviewed incident annotations, version 1",
  "records": [{"snapshot": "capture-002/snapshot.json",
               "previous": "capture-001/snapshot.json",
               "action": "ESCALATE", "group_id": "incident-001"}]
}
```

```sh
uv run system-one assemble-metrics --annotations artifacts/annotations.json \
  --schema artifacts/schema.json --output artifacts/labeled-metrics.csv
uv run system-one train-metrics --data artifacts/labeled-metrics.csv \
  --config experiments/mvp.json --output artifacts/nf-model
```

Assembly preserves labels and source hashes and refuses failed captures or unknown
features. Training consumes every numeric metric column, plus `action` and `group_id`.
At least 20 independent incident/collection-run groups are required. Complete groups
stay in exactly one split, including separate selection and calibration groups.
Training saves data, group/row manifests, configuration, schema, preprocessing,
weights, calibration, classification metrics and reliability plots.
The arbitrary-width path is tested with 24 metrics (48 inputs including masks);
that test is not evidence of an NF-trained model. Existing synthetic reference
models still use six real input fields.

## Shadow recommendations

```sh
uv run system-one shadow --model artifacts/nf-model/model.pt \
  --snapshot artifacts/capture-002/snapshot.json \
  --previous artifacts/capture-001/snapshot.json --log artifacts/shadow-audit.jsonl
```

Shadow mode records model hash, source status, timestamps, recommendation,
probabilities and escalation reason in an append-only log. It executes no action.
Stale/failed telemetry escalates before inference; confidence and probabilities
are null when no model inference was possible. For imputed or unsupported states,
model probabilities are retained for audit but do not imply trustworthy confidence
in the escalation action.
