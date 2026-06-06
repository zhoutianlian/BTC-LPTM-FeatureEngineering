from __future__ import annotations

import argparse

from .config import DEFAULT_CONFIG_PATH, load_config
from .inference import run_batch_inference
from .train import train_pipeline
from .visualization import generate_reports
from .scheduler import monthly_retrain_if_due, run_monitoring
from .io import read_input_frame, write_json
from .features import build_feature_frame
from .validation import NoFutureLeakageChecker, summarize_checks


COMMANDS = {"train", "infer", "report", "validate", "monitor", "scheduled-retrain", "all"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PLIE-PIC project CLI")
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted(COMMANDS),
        help="Pipeline command. If omitted, runtime.command in the config is used.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    command = args.command or str(cfg.get("runtime", "command", default="all"))
    if command not in COMMANDS:
        raise ValueError(f"Unsupported runtime.command: {command}. Expected one of {sorted(COMMANDS)}")
    generate_html = bool(cfg.get("runtime", "generate_html", default=True))
    force_retrain = bool(cfg.get("runtime", "force_retrain", default=False))
    if command in {"train", "all"}:
        artifacts = train_pipeline(cfg)
        print(f"Training complete. Model: {artifacts.model_path}")
    if command in {"infer"}:
        paths = run_batch_inference(cfg)
        for name, p in paths.items():
            print(f"{name}: {p}")
    if command in {"report", "all"} and generate_html:
        paths = generate_reports(cfg)
        for name, p in paths.items():
            print(f"{name}: {p}")
    if command == "monitor":
        paths = run_monitoring(cfg, generate_html=generate_html)
        for name, p in paths.items():
            print(f"{name}: {p}")
    if command == "scheduled-retrain":
        result = monthly_retrain_if_due(cfg, force=force_retrain, generate_html=generate_html)
        print(result)
    if command == "validate":
        raw = read_input_frame(cfg.path("paths", "input_csv"))
        features = build_feature_frame(raw, cfg)
        checker = NoFutureLeakageChecker(cfg)
        checks = checker.run_all(raw, features, features, cfg.get("model", "feature_names"))
        out = summarize_checks(checks)
        path = write_json(out, cfg.path("paths", "manual_validation_checks"))
        print(f"Validation checks written to {path}")


if __name__ == "__main__":
    main()
