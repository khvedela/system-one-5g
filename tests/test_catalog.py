import json

from system_one.catalog import extract_open5gs, markdown_metrics


def test_c_macro_entries_and_histogram_labels_are_extracted():
    text = '''const char *labels_slice[] = {"plmnid", "snssai"};
#define ENTRY(id, name_, desc_) [id] = {.type=OGS_METRICS_METRIC_TYPE_COUNTER, .name=name_, .description=desc_, .labels=labels_slice},
int entries[] = {
ENTRY(NF_COUNTER, "requests", "Requests")
[NF_HIST] = {.type=OGS_METRICS_METRIC_TYPE_HISTOGRAM, .name="latency", .description="Latency", .histogram_params={.count=8}},
};
'''
    entries = extract_open5gs(text)
    assert len(entries) == 2
    assert entries[0]["labels"] == ["plmnid", "snssai"]
    assert entries[1]["histogram_definition"] == ".count=8"


def test_pinned_catalog_has_every_metrics_file_and_no_invented_nf_metrics():
    from pathlib import Path
    catalog = json.loads(Path("experiments/metrics-catalog.json").read_text())
    assert catalog["origins"]["open5gs"]["tag"] == "v2.8.0"
    nfs = {e["nf"] for e in catalog["metrics"] if e["source"] == "open5gs"}
    assert nfs == {"amf", "smf", "upf", "pcf", "mme", "hss", "pcrf"}
    assert catalog["open5gs_nfs"]["ausf"]["defined_families"] == 0
    assert any(e["name"] == "container_memory_working_set_bytes" for e in catalog["metrics"])
    assert any(e["name"] == "kube_pod_container_status_restarts_total" for e in catalog["metrics"])


def test_markdown_table_preserves_escaped_label_pipes(tmp_path):
    path = tmp_path / "metrics.md"
    path.write_text('| kube_status | Gauge | Status | `condition`=<true\\|false> | STABLE |\n')
    assert markdown_metrics(path)[0]["documentation_columns"] == ["`condition`=<true\\|false>", "STABLE"]
