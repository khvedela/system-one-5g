import csv
import json

import pytest

from system_one.features import discover_schema
from system_one.metric_dataset import assemble_table
from system_one.telemetry import parse_metrics


def test_external_labels_preserved_and_no_missing_value_becomes_zero(tmp_path):
    snapshot = {"collected_at": 1, "samples": parse_metrics("load 5\nmissing NaN\n", "nf"),
                "sources": [{"ok": True}]}
    (tmp_path / "snapshot.json").write_text(json.dumps(snapshot))
    schema = discover_schema([snapshot])
    (tmp_path / "schema.json").write_text(json.dumps(schema))
    annotations = {"label_origin": "synthetic test only", "records": [
        {"snapshot": "snapshot.json", "action": "ESCALATE", "group_id": "episode-1"}]}
    (tmp_path / "labels.json").write_text(json.dumps(annotations))
    result = assemble_table(tmp_path / "labels.json", tmp_path / "schema.json", tmp_path / "table.csv")
    assert result["metric_features"] == 2
    with (tmp_path / "table.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["action"] == "ESCALATE"
    assert sorted(row[name] for name in schema["feature_names"]) == ["", "5.0"]
    assert (tmp_path / "table.csv.metadata.json").is_file()
    with pytest.raises(FileExistsError):
        assemble_table(tmp_path / "labels.json", tmp_path / "schema.json", tmp_path / "table.csv")
