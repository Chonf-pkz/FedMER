import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.model_factory import MODEL_CHOICES, normalize_model_name


def _run(cmd, cwd=None):
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def _load_yaml_config(path):
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _parse_list(value):
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _ensure_feature_names(client_feat_dir):
    expected = {
        "train": "MELD_BERT_Wav2Vec2_train.pkl",
        "val": "MELD_BERT_Wav2Vec2_val.pkl",
        "test": "MELD_BERT_Wav2Vec2_test.pkl",
    }
    if all(os.path.isfile(os.path.join(client_feat_dir, name)) for name in expected.values()):
        return True

    mapping = {
        "train": "train_labeled_features.pkl",
        "val": "val_features.pkl",
        "test": "test_features.pkl",
    }
    ok = True
    for split, expected_name in expected.items():
        expected_path = os.path.join(client_feat_dir, expected_name)
        if os.path.isfile(expected_path):
            continue
        src_path = os.path.join(client_feat_dir, mapping[split])
        if not os.path.isfile(src_path):
            ok = False
            continue
        try:
            os.symlink(src_path, expected_path)
        except OSError:
            import shutil
            shutil.copy2(src_path, expected_path)
    return ok


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _count_test_samples(client_feat_dir):
    cand = os.path.join(client_feat_dir, "MELD_BERT_Wav2Vec2_test.pkl")
    if not os.path.isfile(cand):
        cand = os.path.join(client_feat_dir, "test_features.pkl")
    if not os.path.isfile(cand):
        return 0
    try:
        data = _load_pickle(cand)
        return len(data)
    except Exception:
        return 0


def _read_results_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    metrics = {}
    for row in rows:
        metric = row.get("Metric")
        mean = row.get("Mean")
        std = row.get("Std")
        if metric and mean is not None:
            metrics[metric] = {
                "mean": float(mean),
                "std": float(std) if std is not None else 0.0,
            }
    return metrics


def _resolve_results_path(results_root, logs_root, results_suffix):
    candidates = [
        os.path.join(results_root or "", f"{results_suffix}.csv"),
        os.path.join("results", f"{results_suffix}.csv"),
        os.path.join(logs_root or "", f"{results_suffix}.csv"),
        os.path.join(os.getcwd(), f"{results_suffix}.csv"),
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return None, candidates


def _mean_std(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5


def _weighted_mean(values, weights):
    total = float(sum(weights))
    if total <= 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total


def parse_args():
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", type=str, default=None)
    known, _ = base.parse_known_args()
    cfg = _load_yaml_config(known.config)

    parser = argparse.ArgumentParser(description="Local-only centralized training on MELD clients.")
    parser.add_argument("--config", type=str, default=known.config)
    parser.add_argument("--dataset", type=str, default="MELD", choices=["MELD"])
    parser.add_argument("--num_classes", type=int, default=7, choices=[7])
    parser.add_argument("--clients_root", type=str, default=None)
    parser.add_argument("--features_root", type=str, default=None)
    parser.add_argument("--logs_root", type=str, default="logs/centralized")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--num_clients", type=int, default=None)
    parser.add_argument("--client_filter", type=str, default=None,
                        help="Comma-separated list of client ids to run.")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--model_name",
        type=str,
        default="fedalmer",
        help=f"Model name. Supported: {', '.join(MODEL_CHOICES)}",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--modality", type=str, default="both", choices=["both", "text", "audio"])

    parser.add_argument("--skip_train", action="store_true")

    if cfg:
        paths = cfg.get("paths", {}) or {}
        model_cfg = cfg.get("model", {}) or {}
        parser.set_defaults(
            clients_root=paths.get("clients_root") or paths.get("out_base"),
            features_root=paths.get("features_root") or paths.get("features_base"),
            logs_root=paths.get("logs_root"),
            results_root=paths.get("results_root"),
            run_id=cfg.get("run_id"),
            model_name=model_cfg.get("name") or cfg.get("model_name"),
        )
        train = cfg.get("train", {}) or {}
        parser.set_defaults(
            model_name=train.get("model_name") or parser.get_default("model_name"),
            epochs=train.get("epochs"),
            batch_size=train.get("batch_size"),
            modality=train.get("modality") or "both",
        )
    args = parser.parse_args()
    args.model_name = normalize_model_name(args.model_name)

    if not args.features_root:
        parser.error("Missing required args: --features_root (or set in config)")
    return args


def main():
    args = parse_args()

    run_id = args.run_id
    features_root = args.features_root
    if run_id:
        features_root = os.path.join(features_root, run_id)
    if not os.path.isdir(features_root):
        raise FileNotFoundError(f"features_root not found: {features_root}")

    clients = sorted(p.name for p in Path(features_root).glob("client_*") if p.is_dir())
    if args.client_filter:
        allow = set(_parse_list(args.client_filter))
        clients = [c for c in clients if c in allow]
    if args.num_clients:
        clients = clients[: int(args.num_clients)]
    if not clients:
        raise ValueError(f"No clients found under {features_root}")

    base_exp = args.exp_name or run_id or "MELD_local_only"
    if args.model_name != "fedalmer" and args.exp_name is None:
        base_exp = f"{base_exp}_{args.model_name}"

    per_client = []
    for client_id in clients:
        client_feat_dir = os.path.join(features_root, client_id)
        if not _ensure_feature_names(client_feat_dir):
            raise FileNotFoundError(f"Missing required feature files under {client_feat_dir}")

        client_exp = f"{base_exp}_{client_id}"
        results_suffix = f"{client_exp}_results"
        results_path = os.path.join(args.results_root, f"{results_suffix}.csv")

        if not args.skip_train:
            cmd = [
                os.path.basename(os.environ.get("PYTHON", "")) or "python",
                "centralized/train.py",
                "--data_dir", client_feat_dir,
                "--dataset", args.dataset,
                "--num_classes", str(args.num_classes),
                "--model_name", args.model_name,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--modality", args.modality,
                "--logs_root", args.logs_root,
                "--exp_name", client_exp,
                "--reset_logs",
                "--results_suffix", results_suffix,
            ]
            _run(cmd)

        if not os.path.isfile(results_path):
            resolved = _resolve_results_path(args.results_root, args.logs_root, results_suffix)
            if isinstance(resolved, tuple):
                found, candidates = resolved
            else:
                found, candidates = resolved, []
            if found:
                results_path = found
            else:
                raise FileNotFoundError(
                    f"Missing results file: {results_path}. "
                    f"Searched: {', '.join(candidates)}"
                )

        metrics = _read_results_csv(results_path)
        test_samples = _count_test_samples(client_feat_dir)
        per_client.append({
            "client_id": client_id,
            "test_samples": test_samples,
            "WA": metrics.get("WA", {}).get("mean", 0.0),
            "WA_std": metrics.get("WA", {}).get("std", 0.0),
            "UA": metrics.get("UA", {}).get("mean", 0.0),
            "UA_std": metrics.get("UA", {}).get("std", 0.0),
            "WF1": metrics.get("WF1", {}).get("mean", 0.0),
            "WF1_std": metrics.get("WF1", {}).get("std", 0.0),
            "UF1": metrics.get("UF1", {}).get("mean", 0.0),
            "UF1_std": metrics.get("UF1", {}).get("std", 0.0),
            "results_path": os.path.abspath(results_path),
        })

    os.makedirs(args.logs_root, exist_ok=True)
    out_dir = os.path.join(args.logs_root, "meld_local_only", base_exp)
    os.makedirs(out_dir, exist_ok=True)

    per_client_path = os.path.join(out_dir, "per_client_metrics.csv")
    with open(per_client_path, "w", newline="") as f:
        fieldnames = [
            "client_id", "test_samples",
            "WA", "WA_std",
            "UA", "UA_std",
            "WF1", "WF1_std",
            "UF1", "UF1_std",
            "results_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_client:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary = {}
    for key in ("WA", "UA", "WF1", "UF1"):
        values = [float(r.get(key, 0.0)) for r in per_client]
        mean, std = _mean_std(values)
        weights = [int(r.get("test_samples", 0)) for r in per_client]
        summary[f"macro_{key}"] = mean
        summary[f"macro_{key}_std"] = std
        summary[f"weighted_{key}"] = _weighted_mean(values, weights)

    summary["num_clients"] = len(per_client)
    summary["total_test_samples"] = sum(int(r.get("test_samples", 0)) for r in per_client)
    summary["model_name"] = args.model_name

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("MELD local-only summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
