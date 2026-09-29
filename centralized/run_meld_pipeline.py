import argparse
import os
import shutil
import subprocess
import sys

import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.model_factory import MODEL_CHOICES, normalize_model_name

DEFAULT_METADATA_SAVE_DIR = (
    "/home/tri.pm/polyp/fptu/MinhNhut/FedalSER_document/metadata/MELD_centralized"
)


def _run(cmd):
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_yaml_config(path):
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _with_run_id(path, run_id):
    if not path:
        return path
    path = os.path.normpath(str(path))
    if not run_id:
        return path

    # Support templated config paths like ".../{run_id}" or ".../${run_id}".
    path = (
        path.replace("{run_id}", run_id)
        .replace("${run_id}", run_id)
        .replace("$run_id", run_id)
        .replace("$RUN_ID", run_id)
    )
    # Also tolerate accidentally hardcoded "$<run_id>" fragments.
    path = path.replace(f"${run_id}", run_id)
    path = os.path.normpath(path)

    if os.path.basename(path) == run_id:
        return path
    if run_id in path.split(os.sep):
        return path
    return os.path.join(path, run_id)


def _ensure_files_exist(root_dir, filenames, label):
    missing = [name for name in filenames if not os.path.isfile(os.path.join(root_dir, name))]
    if missing:
        raise FileNotFoundError(
            f"Missing {label} files under {root_dir}: {', '.join(missing)}"
        )


def _find_dir_with_required_files(candidates, filenames):
    for root_dir in candidates:
        if not root_dir:
            continue
        if all(os.path.isfile(os.path.join(root_dir, name)) for name in filenames):
            return root_dir
    return None


def _resolve_results_path(results_root, logs_root, results_suffix):
    candidates = [
        os.path.join(results_root or "", f"{results_suffix}.csv"),
        os.path.join("results", f"{results_suffix}.csv"),
        os.path.join(logs_root or "", f"{results_suffix}.csv"),
        os.path.join(os.getcwd(), f"{results_suffix}.csv"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _preprocess_supports_out_dir(preprocess_script):
    if not os.path.isfile(preprocess_script):
        return False
    with open(preprocess_script, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return "--out_dir" in content


def _sync_meld_metadata(src_dir, dst_dir):
    required = ["train.pkl", "val.pkl", "test.pkl"]
    _ensure_files_exist(src_dir, required, label="legacy metadata")
    os.makedirs(dst_dir, exist_ok=True)
    for name in required:
        shutil.copy2(os.path.join(src_dir, name), os.path.join(dst_dir, name))


def _save_metadata_snapshot(src_dir, dst_dir):
    if not dst_dir:
        return
    if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
        return
    required = ["train.pkl", "val.pkl", "test.pkl"]
    _ensure_files_exist(src_dir, required, label="metadata")
    os.makedirs(dst_dir, exist_ok=True)
    for name in required:
        shutil.copy2(os.path.join(src_dir, name), os.path.join(dst_dir, name))


def _set_defaults_if_present(parser, values):
    defaults = {k: v for k, v in values.items() if v is not None}
    if defaults:
        parser.set_defaults(**defaults)


def parse_args():
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", type=str, default=None)
    known, _ = base.parse_known_args()
    cfg = _load_yaml_config(known.config)

    parser = argparse.ArgumentParser(
        description="Run MELD centralized pipeline (preprocess + extract features + train)."
    )
    parser.add_argument("--config", type=str, default=known.config)

    parser.add_argument("--dataset", type=str, default="MELD", choices=["MELD"])
    parser.add_argument("--num_classes", type=int, default=7, choices=[7])
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)

    parser.add_argument("--data_root", type=str, default=None,
                        help="Root path containing MELD CSV metadata and audio files.")
    parser.add_argument("--audio_root", type=str, default=None,
                        help="Optional audio root for feature extraction (defaults to --data_root).")
    parser.add_argument("--metadata_root", type=str, default="metadata/MELD_preprocessed")
    parser.add_argument("--metadata_save_dir", type=str, default=DEFAULT_METADATA_SAVE_DIR,
                        help="Base directory to save MELD metadata snapshots (train/val/test.pkl).")
    parser.add_argument("--features_root", type=str, default="features/MELD/centralized")
    parser.add_argument("--logs_root", type=str, default="logs/centralized")
    parser.add_argument("--results_root", type=str, default="results")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore_length", type=int, default=0)

    parser.add_argument(
        "--model_name",
        type=str,
        default="fedalmer",
        help=f"Model name. Supported: {', '.join(MODEL_CHOICES)}",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--modality", type=str, default="both", choices=["both", "text", "audio"])

    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_features", action="store_true")
    parser.add_argument("--skip_train", action="store_true")

    if cfg:
        paths = cfg.get("paths", {}) or {}
        model_cfg = cfg.get("model", {}) or {}
        _set_defaults_if_present(parser, {
            "dataset": cfg.get("dataset"),
            "num_classes": cfg.get("num_classes"),
            "run_id": cfg.get("run_id"),
            "data_root": paths.get("data_root"),
            "audio_root": paths.get("audio_root"),
            "metadata_root": paths.get("metadata_root"),
            "metadata_save_dir": paths.get("metadata_save_dir"),
            "features_root": paths.get("features_root"),
            "logs_root": paths.get("logs_root"),
            "results_root": paths.get("results_root"),
            "model_name": model_cfg.get("name") or cfg.get("model_name"),
        })

        preprocess_cfg = cfg.get("preprocess", {}) or {}
        _set_defaults_if_present(parser, {
            "seed": preprocess_cfg.get("seed"),
            "ignore_length": preprocess_cfg.get("ignore_length"),
        })

        train_cfg = cfg.get("train", {}) or {}
        _set_defaults_if_present(parser, {
            "model_name": train_cfg.get("model_name") or parser.get_default("model_name"),
            "epochs": train_cfg.get("epochs"),
            "batch_size": train_cfg.get("batch_size"),
            "modality": train_cfg.get("modality") or parser.get_default("modality"),
        })

    args = parser.parse_args()
    args.model_name = normalize_model_name(args.model_name)

    if not args.data_root:
        parser.error("Missing required args: --data_root (or set paths.data_root in config)")
    if not args.audio_root:
        args.audio_root = args.data_root

    return args


def main():
    args = parse_args()

    metadata_dir = _with_run_id(args.metadata_root, args.run_id)
    metadata_save_dir = _with_run_id(args.metadata_save_dir, args.run_id)
    features_dir = _with_run_id(args.features_root, args.run_id)
    required_metadata = ["train.pkl", "val.pkl", "test.pkl"]
    legacy_metadata_dir = os.path.join("metadata", "MELD_preprocessed")
    preprocess_script = os.path.join("centralized", "preprocess.py")
    preprocess_has_out_dir = _preprocess_supports_out_dir(preprocess_script)

    if not args.skip_preprocess:
        cmd = [
            sys.executable,
            preprocess_script,
            "--dataset", "MELD",
            "--data_root", args.data_root,
            "--seed", str(args.seed),
            "--ignore_length", str(args.ignore_length),
        ]
        if preprocess_has_out_dir:
            cmd += ["--out_dir", metadata_dir]
        _run(cmd)
        if not preprocess_has_out_dir:
            if os.path.abspath(legacy_metadata_dir) != os.path.abspath(metadata_dir):
                print(
                    "[WARN] centralized/preprocess.py does not support --out_dir; "
                    "copying metadata from legacy output directory."
                )
                _sync_meld_metadata(legacy_metadata_dir, metadata_dir)
            else:
                _ensure_files_exist(metadata_dir, required_metadata, label="metadata")
    else:
        resolved_metadata_dir = _find_dir_with_required_files(
            [metadata_dir, metadata_save_dir, legacy_metadata_dir],
            required_metadata,
        )
        if not resolved_metadata_dir:
            raise FileNotFoundError(
                "Missing metadata files for --skip_preprocess. "
                f"Checked: {metadata_dir}, {metadata_save_dir}, {legacy_metadata_dir}"
            )
        if os.path.abspath(resolved_metadata_dir) != os.path.abspath(metadata_dir):
            print(
                "[WARN] Requested metadata path does not contain train/val/test; "
                f"using fallback path: {resolved_metadata_dir}"
            )
        metadata_dir = resolved_metadata_dir

    _save_metadata_snapshot(metadata_dir, metadata_save_dir)

    if not args.skip_features:
        cmd = [
            sys.executable,
            "feature_extract/extract_feature.py",
            "--dataset", "MELD",
            "--wav_base", args.audio_root,
            "--pkl_dir", metadata_dir,
            "--output_dir", features_dir,
        ]
        _run(cmd)
    else:
        _ensure_files_exist(
            features_dir,
            [
                "MELD_BERT_Wav2Vec2_train.pkl",
                "MELD_BERT_Wav2Vec2_val.pkl",
                "MELD_BERT_Wav2Vec2_test.pkl",
            ],
            label="feature",
        )

    base_exp = args.exp_name or args.run_id or "MELD_centralized"
    if args.model_name != "fedalmer" and args.exp_name is None:
        base_exp = f"{base_exp}_{args.model_name}"

    results_suffix = f"{base_exp}_results"

    if args.skip_train:
        print("Skip training enabled.")
        print(f"Metadata dir: {os.path.abspath(metadata_dir)}")
        print(f"Features dir: {os.path.abspath(features_dir)}")
        return

    cmd = [
        sys.executable,
        "centralized/train.py",
        "--data_dir", features_dir,
        "--dataset", "MELD",
        "--num_classes", "7",
        "--model_name", args.model_name,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--modality", args.modality,
        "--logs_root", args.logs_root,
        "--exp_name", base_exp,
        "--reset_logs",
        "--results_suffix", results_suffix,
    ]
    _run(cmd)

    results_path = _resolve_results_path(args.results_root, args.logs_root, results_suffix)
    if not results_path:
        raise FileNotFoundError(
            f"Results not found for suffix {results_suffix}. "
            f"Checked roots: {args.results_root}, {args.logs_root}, {os.getcwd()}"
        )

    print("MELD centralized pipeline completed.")
    print(f"Metadata dir: {os.path.abspath(metadata_dir)}")
    if metadata_save_dir:
        print(f"Metadata snapshot dir: {os.path.abspath(metadata_save_dir)}")
    print(f"Features dir: {os.path.abspath(features_dir)}")
    print(f"Results CSV: {os.path.abspath(results_path)}")


if __name__ == "__main__":
    main()
