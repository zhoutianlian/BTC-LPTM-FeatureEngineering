from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from stable_history import overlay_csv_history


ROOT_DIR = Path(__file__).resolve().parent
MODEL_INPUT_DIR = ROOT_DIR / "model_input"
HMM_BUNDLE_PATH = ROOT_DIR / "liq_pressure_hmm/model/liq_hmm_bundle.joblib"
INPUT_ZIP_PATH = ROOT_DIR / "input.zip"


def build_pipeline_commands(*, retrain_hmm: bool) -> list[tuple[str, list[str]]]:
    hmm_command = [sys.executable, "-m", "liq_pressure_hmm.feature_plie_HMM"]
    if retrain_hmm:
        hmm_command.append("--retrain")
    else:
        hmm_command.append("--skip-train")

    return [
        ("download", [sys.executable, "-m", "liq_data_download.scripts.run_data_download"]),
        ("liq_dataflow", [sys.executable, "-m", "liq_dataflow.feature_liq_dataflow"]),
        ("hmm_pressure", hmm_command),
        ("plie", [sys.executable, "-m", "plie_pic_project.feature_plie_pic"]),
        ("price context", [sys.executable, "-m", "price_context.feature_price_context"]),
        ("abs", [sys.executable, "-m", "qd_mar_project.feature_qd_mar"]),
        ("liqprice", [sys.executable, "-m", "btc_liqprice_features_artifact.feature_liqprice"]),
    ]


def validate_hmm_mode(*, retrain_hmm: bool) -> None:
    if retrain_hmm:
        return
    if not HMM_BUNDLE_PATH.is_file():
        raise SystemExit(
            "Missing HMM bundle for new-data inference mode:\n"
            f"- {HMM_BUNDLE_PATH.relative_to(ROOT_DIR)}\n"
            "Default full-pipeline runs reuse the accepted HMM bundle to keep historical states stable. "
            "Provide the bundle, or rerun with --retrain-hmm only when you intentionally want to fit a new HMM."
        )

RESULT_FILES = [
    Path("plie_pic_project/outputs/predictions/plie_predictions_source.csv"),
    Path("qd_mar_project/output/features/absorption_memory.csv"),
    Path("qd_mar_project/output/features/base_context.csv"),
    Path("qd_mar_project/output/features/path_absorption_multiscale.csv"),
    Path("qd_mar_project/output/features/path_absorption.csv"),
    Path("price_context/output/price_context_features.csv"),
    Path("btc_liqprice_features_artifact/output/liqprice_features.csv"),
    Path("liq_dataflow/data/features/features_liq_dataflow.csv"),
]

STABLE_HISTORY_REFERENCES: dict[Path, tuple[str, dict[str, str]]] = {
    Path("liq_dataflow/data/features/features_liq_dataflow.csv"): (
        "input.zip::input/fhmv_liq_features.csv",
        {"RPN": "risk_priority_number"},
    ),
    Path("plie_pic_project/outputs/predictions/plie_predictions_source.csv"): (
        "input.zip::input/plie_predictions_source.csv",
        {},
    ),
    Path("qd_mar_project/output/features/absorption_memory.csv"): (
        "input.zip::input/absorption_memory.csv",
        {},
    ),
    Path("qd_mar_project/output/features/base_context.csv"): (
        "input.zip::input/base_context.csv",
        {},
    ),
    Path("qd_mar_project/output/features/path_absorption_multiscale.csv"): (
        "input.zip::input/path_absorption_multiscale.csv",
        {},
    ),
    Path("qd_mar_project/output/features/path_absorption.csv"): (
        "input.zip::input/path_absorption.csv",
        {},
    ),
    Path("price_context/output/price_context_features.csv"): (
        "input.zip::input/price_context_features.csv",
        {},
    ),
    Path("btc_liqprice_features_artifact/output/liqprice_features.csv"): (
        "input.zip::input/liqprice_features.csv",
        {},
    ),
}

STEP_STABLE_OUTPUTS: dict[str, list[Path]] = {
    "liq_dataflow": [Path("liq_dataflow/data/features/features_liq_dataflow.csv")],
    "plie": [Path("plie_pic_project/outputs/predictions/plie_predictions_source.csv")],
    "price context": [Path("price_context/output/price_context_features.csv")],
    "abs": [
        Path("qd_mar_project/output/features/absorption_memory.csv"),
        Path("qd_mar_project/output/features/base_context.csv"),
        Path("qd_mar_project/output/features/path_absorption_multiscale.csv"),
        Path("qd_mar_project/output/features/path_absorption.csv"),
    ],
    "liqprice": [Path("btc_liqprice_features_artifact/output/liqprice_features.csv")],
}


def run_pipeline_steps(commands: list[tuple[str, list[str]]]) -> None:
    with tempfile.TemporaryDirectory(prefix="stable_history_", dir=ROOT_DIR) as tmp_dir:
        tmp_root = Path(tmp_dir)
        for name, command in commands:
            snapshots = snapshot_step_outputs(name, tmp_root)
            print(f"\n==> Running {name}: {' '.join(command)}", flush=True)
            completed = subprocess.run(command, cwd=ROOT_DIR)
            if completed.returncode != 0:
                raise SystemExit(f"{name} failed with exit code {completed.returncode}")
            stabilize_step_outputs(name, snapshots)


def snapshot_step_outputs(step_name: str, tmp_root: Path) -> dict[Path, Path]:
    snapshots: dict[Path, Path] = {}
    for path in STEP_STABLE_OUTPUTS.get(step_name, []):
        source = ROOT_DIR / path
        if not source.is_file():
            continue
        target = tmp_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshots[path] = target
    return snapshots


def stabilize_output(path: Path, previous_reference: Path | None = None) -> None:
    source = ROOT_DIR / path
    if not source.is_file():
        return

    if previous_reference is not None and previous_reference.is_file():
        stats = overlay_csv_history(source, str(previous_reference), output_path=source, root=ROOT_DIR)
        print(
            "Stable history overlay "
            f"{path}: restored {stats['preserved_rows']} previously accepted rows",
            flush=True,
        )

    if INPUT_ZIP_PATH.is_file():
        spec = STABLE_HISTORY_REFERENCES.get(path)
        if spec is None:
            return
        reference, rename = spec
        stats = overlay_csv_history(
            source,
            reference,
            output_path=source,
            root=ROOT_DIR,
            reference_rename=rename,
        )
        print(
            "Stable history overlay "
            f"{path}: preserved {stats['preserved_rows']} baseline rows, "
            f"kept {stats['current_rows'] - stats['preserved_rows']} rows beyond baseline",
            flush=True,
        )


def stabilize_step_outputs(step_name: str, snapshots: dict[Path, Path] | None = None) -> None:
    snapshots = snapshots or {}
    for path in STEP_STABLE_OUTPUTS.get(step_name, []):
        stabilize_output(path, previous_reference=snapshots.get(path))


def recreate_model_input_dir() -> None:
    if MODEL_INPUT_DIR.is_symlink() or MODEL_INPUT_DIR.is_file():
        MODEL_INPUT_DIR.unlink()
    elif MODEL_INPUT_DIR.is_dir():
        shutil.rmtree(MODEL_INPUT_DIR)
    MODEL_INPUT_DIR.mkdir()


def copy_results() -> None:
    missing_files = [path for path in RESULT_FILES if not (ROOT_DIR / path).is_file()]
    if missing_files:
        missing_text = "\n".join(f"- {path}" for path in missing_files)
        raise SystemExit(f"Missing result files:\n{missing_text}")

    recreate_model_input_dir()
    for path in RESULT_FILES:
        source = ROOT_DIR / path
        stabilize_output(path)
        target = MODEL_INPUT_DIR / source.name
        shutil.copy2(source, target)
        print(f"Copied {path} -> {target.relative_to(ROOT_DIR)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run README pipeline and collect model input files.")
    parser.add_argument(
        "--retrain-hmm",
        action="store_true",
        help="Retrain liq_pressure_hmm instead of reusing the existing bundle.",
    )
    args = parser.parse_args()

    validate_hmm_mode(retrain_hmm=args.retrain_hmm)
    run_pipeline_steps(build_pipeline_commands(retrain_hmm=args.retrain_hmm))
    copy_results()
    print(f"\nDone. Model input files are in {MODEL_INPUT_DIR.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
