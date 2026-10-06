import json
import subprocess

import pytest

from system_one.decision import DecisionModel
from system_one.experiment import run_experiment
from system_one.state import State


def test_end_to_end_artifacts_inference_and_no_overwrite(tmp_path):
    output = tmp_path / "run"
    config = {"dataset": {"samples": 1000, "seed": 7},
              "training": {"epochs": 2, "hidden_sizes": [8, 4]},
              "evaluation": {"latency_iterations": 5}}
    result = run_experiment(output, config)
    assert result["split_sizes"]["test"] == 150
    assert result["uncertainty"]["threshold"] >= .8
    assert len(result["ood"]) == 3
    assert result["latency"]["iterations"] == 5
    for name in ("model.pt", "config.json", "environment.json", "metrics.json", "training.json",
                 "reliability.png", "ood.json", "high-confidence-errors.json", "dataset/splits.json"):
        assert (output / name).is_file()
    decision = DecisionModel.load(output / "model.pt").decide(State(50, 50, 500, 100, .01, 2))
    assert sum(decision.probabilities.values()) == pytest.approx(1)
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(State(50, 50, 500, 100, .01, 2).as_dict()))
    cli = subprocess.run(["system-one", "decide", "--model", str(output / "model.pt"),
                          "--state", str(state_file)], capture_output=True, text=True, check=True)
    assert json.loads(cli.stdout) == decision.as_dict()
    before = (output / "model.pt").read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(output, config)
    assert (output / "model.pt").read_bytes() == before
