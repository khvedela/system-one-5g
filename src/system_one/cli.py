import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone System One experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Generate, train, calibrate and evaluate")
    run.add_argument("--config", type=Path, default=Path("experiments/mvp.json"))
    run.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten")
    run.add_argument("--seed", type=int, help="Override both dataset and training seeds")
    decide = commands.add_parser("decide", help="Return a decision for one JSON state")
    decide.add_argument("--model", type=Path, required=True)
    decide.add_argument("--state", type=Path, required=True)
    collect = commands.add_parser("collect", help="Discover and preserve all exposed telemetry")
    collect.add_argument("--config", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    schema = commands.add_parser("schema", help="Freeze features from training captures")
    schema.add_argument("--snapshots", type=Path, nargs="+", required=True)
    schema.add_argument("--output", type=Path, required=True)
    catalog = commands.add_parser("catalog", help="Extract pinned Open5GS and Kubernetes metric definitions")
    catalog.add_argument("--open5gs", type=Path, required=True)
    catalog.add_argument("--cadvisor", type=Path, required=True)
    catalog.add_argument("--kube-state-metrics", type=Path, required=True)
    catalog.add_argument("--output", type=Path, required=True)
    table = commands.add_parser("train-metrics", help="Train on all numeric CSV metrics with independent group splits")
    table.add_argument("--data", type=Path, required=True)
    table.add_argument("--output", type=Path, required=True)
    table.add_argument("--config", type=Path, default=Path("experiments/mvp.json"))
    robust = commands.add_parser("robustness", help="Evaluate support guards, broader cases, simulator and readiness")
    robust.add_argument("--reference", type=Path, required=True)
    robust.add_argument("--output", type=Path, required=True)
    robust.add_argument("--criteria", type=Path, default=Path("experiments/readiness.json"))
    shadow = commands.add_parser("shadow", help="Record telemetry decisions without executing actions")
    shadow.add_argument("--model", type=Path, required=True)
    shadow.add_argument("--snapshot", type=Path, required=True)
    shadow.add_argument("--previous", type=Path)
    shadow.add_argument("--log", type=Path, required=True)
    shadow.add_argument("--max-age-seconds", type=float, default=30)
    export = commands.add_parser("features", help="Export every captured numeric feature and counter rate")
    export.add_argument("--snapshot", type=Path, required=True)
    export.add_argument("--previous", type=Path)
    export.add_argument("--output", type=Path, required=True)
    dataset = commands.add_parser("assemble-metrics", help="Join captures with explicit labels and a frozen schema")
    dataset.add_argument("--annotations", type=Path, required=True)
    dataset.add_argument("--schema", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            from .experiment import run_experiment
            config = json.loads(args.config.read_text())
            if args.seed is not None:
                config.setdefault("dataset", {})["seed"] = args.seed
                config.setdefault("training", {})["seed"] = args.seed
            run_experiment(args.output, config)
        elif args.command == "collect":
            from .telemetry import capture
            result = capture(json.loads(args.config.read_text()), args.output)
            print(json.dumps({"samples": len(result["samples"]), "sources": result["sources"]}, indent=2))
            if any(not source["ok"] for source in result["sources"]):
                raise SystemExit(1)
        elif args.command == "assemble-metrics":
            from .metric_dataset import assemble_table
            print(json.dumps(assemble_table(args.annotations, args.schema, args.output), indent=2))
        elif args.command == "features":
            from .features import snapshot_features
            previous = json.loads(args.previous.read_text()) if args.previous else None
            values = snapshot_features(json.loads(args.snapshot.read_text()), previous)
            with args.output.open("x") as stream:
                json.dump(values, stream, indent=2, allow_nan=False)
            print(f"Exported {len(values)} metric features")
        elif args.command == "shadow":
            from .shadow import shadow_decision
            previous = json.loads(args.previous.read_text()) if args.previous else None
            result = shadow_decision(args.model, json.loads(args.snapshot.read_text()), previous,
                                     args.log, args.max_age_seconds)
            print(json.dumps(result, indent=2))
        elif args.command == "robustness":
            from .robustness import run_robustness
            result = run_robustness(args.reference, args.output, json.loads(args.criteria.read_text()))
            print(json.dumps({"test": result["test"], "ood": result["ood"],
                              "broader_distribution": result["broader_distribution"],
                              "readiness": result["readiness"]}, indent=2))
        elif args.command == "train-metrics":
            from .table_training import train_table
            from .training import TrainingConfig
            settings = json.loads(args.config.read_text()).get("training", {})
            if "hidden_sizes" in settings:
                settings["hidden_sizes"] = tuple(settings["hidden_sizes"])
            result = train_table(args.data, args.output, TrainingConfig(**settings))
            print(json.dumps({"metric_count": result["metric_count"],
                              "input_width": result["input_width_including_missing_indicators"],
                              "final_accuracy": result["final_classification"]["accuracy"]}, indent=2))
        elif args.command == "catalog":
            from .catalog import build_catalog
            result = build_catalog(args.open5gs, args.cadvisor, args.kube_state_metrics, args.output)
            print(json.dumps({"origins": result["origins"], "counts": result["counts"],
                              "open5gs_nfs": result["open5gs_nfs"]}, indent=2))
        elif args.command == "schema":
            from .features import discover_schema
            result = discover_schema([json.loads(path.read_text()) for path in args.snapshots])
            with args.output.open("x") as stream:
                json.dump(result, stream, indent=2)
            print(f"Discovered {len(result['feature_names'])} features")
        else:
            from .decision import DecisionModel
            model = DecisionModel.load(args.model)
            print(json.dumps(model.decide(json.loads(args.state.read_text())).as_dict(), indent=2))
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
