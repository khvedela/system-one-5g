from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from system_one.features import discover_schema, encode_features, snapshot_features
from system_one.telemetry import capture, parse_metrics, flatten_numeric


TEXT = '''# HELP nf_requests_total Requests
# TYPE nf_requests_total counter
nf_requests_total{slice="a",cause="ok"} 20
nf_requests_total{slice="a",cause="fail"} 2
# TYPE nf_sessions gauge
nf_sessions{slice="a"} 5
# TYPE nf_delay_seconds histogram
nf_delay_seconds_bucket{le="0.1"} 10
nf_delay_seconds_bucket{le="+Inf"} 20
nf_delay_seconds_sum 3
nf_delay_seconds_count 20
unknown_metric NaN
'''


def test_all_series_labels_types_and_nonfinite_values_are_preserved():
    samples = parse_metrics(TEXT, "amf")
    assert len(samples) == 8
    assert samples[0]["labels"] == {"slice": "a", "cause": "ok"}
    assert samples[0]["cumulative"]
    assert not samples[2]["cumulative"]
    assert samples[-1]["value"] is None
    assert samples[-1]["raw_value"] == "nan"
    assert len({s["key"] for s in samples}) == 8


def test_counter_rates_missing_values_and_reset_are_explicit():
    first = {"collected_at": 1, "samples": parse_metrics(TEXT, "amf"), "sources": [{"ok": True}]}
    second = {"collected_at": 3, "samples": parse_metrics(TEXT.replace(" 20", " 24"), "amf"),
              "sources": [{"ok": True}]}
    key = first["samples"][0]["key"]
    values = snapshot_features(second, first)
    assert values[key + "/rate_per_second"] == 2
    reset = {**second, "samples": parse_metrics(TEXT.replace(" 20", " 1"), "amf")}
    assert snapshot_features(reset, first)[key + "/rate_per_second"] is None
    assert snapshot_features(reset, first)[key + "/counter_reset"] == 1
    schema = discover_schema([first, second])
    assert len(schema["feature_names"]) > 8
    encoded, missing, extra = encode_features(values, schema, [0.] * len(schema["feature_names"]))
    assert len(encoded) == 2 * len(schema["feature_names"])
    assert missing and not extra
    with pytest.raises(ValueError):
        discover_schema([{**first, "sources": [{"ok": False}]}])


def test_capture_real_http_preserves_body_inventory_and_failure_status(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(TEXT.encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        config = {"targets": [{"id": "amf", "url": f"http://127.0.0.1:{server.server_port}/metrics"},
                              {"id": "missing", "url": "http://127.0.0.1:1/metrics"}]}
        snapshot = capture(config, tmp_path / "capture")
        assert snapshot["sources"][0]["ok"]
        assert not snapshot["sources"][1]["ok"]
        assert len(snapshot["samples"]) == 8
        assert (tmp_path / "capture/amf.metrics").read_text() == TEXT
        assert len(json.loads((tmp_path / "capture/inventory.json").read_text())) == 8
        with pytest.raises(FileExistsError):
            capture(config, tmp_path / "capture")
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_docker_nested_numeric_fields_are_not_reduced_to_summary():
    values = flatten_numeric({"cpu": {"percpu": [2, 3]}, "network": {"drops": 4}, "status": "running"})
    assert values == {"/cpu/percpu/0": 2., "/cpu/percpu/1": 3., "/network/drops": 4.}


def test_openmetrics_and_duplicate_series():
    metrics = parse_metrics("# TYPE sessions gauge\nsessions 2\n# EOF\n", "smf", openmetrics=True)
    assert metrics[0]["value"] == 2
    with pytest.raises(ValueError):
        parse_metrics("sessions 2\nsessions 3\n", "smf")
    counter = parse_metrics("# TYPE open5gs_counter counter\nopen5gs_counter 2\n", "amf")[0]
    assert counter["name"] == "open5gs_counter"
    assert counter["type"] == "counter"


def test_full_docker_stats_lifecycle_and_no_container_secrets(monkeypatch):
    import docker
    from system_one.telemetry import docker_snapshot
    class Container:
        id = "123"
        name = "open5gs-upf"
        attrs = {"State": {"Running": True, "OOMKilled": False, "Pid": 4}, "RestartCount": 2,
                 "Config": {"Env": ["PASSWORD=secret"]}, "HostConfig": {"Memory": 1000}}
        def stats(self, stream):
            assert not stream
            return {"cpu_stats": {"cpu_usage": {"total_usage": 200}, "system_cpu_usage": 1000, "online_cpus": 2},
                    "precpu_stats": {"cpu_usage": {"total_usage": 100}, "system_cpu_usage": 800},
                    "memory_stats": {"stats": {"rss": 30, "cache": 10}},
                    "networks": {"eth0": {"rx_bytes": 42, "rx_dropped": 1}},
                    "pids_stats": {"current": 5}}
    class Containers:
        def list(self, all, filters):
            assert all
            return [Container()]
    class Client:
        containers = Containers()
        def close(self):
            pass
    monkeypatch.setattr(docker, "DockerClient", lambda **kwargs: Client())
    samples, raw = docker_snapshot({"base_url": "unix://fixture"}, 5)
    assert raw["123"]["stats"]["networks"]["eth0"]["rx_bytes"] == 42
    assert "secret" not in json.dumps(raw)
    assert any(s["name"].endswith("/rss") for s in samples)
    assert any(s["name"].endswith("/rx_bytes") and s["cumulative"] for s in samples)
    assert next(s["value"] for s in samples if s["name"] == "derived/cpu_percent_of_one_core") == 100


def test_block_device_metric_identity_does_not_depend_on_list_order():
    one = {"major": 8, "minor": 0, "op": "Read", "value": 10}
    two = {"major": 8, "minor": 0, "op": "Write", "value": 20}
    assert flatten_numeric([one, two]) == flatten_numeric([two, one])
