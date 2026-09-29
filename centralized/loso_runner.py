import argparse
import csv
import json
import logging
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.model_factory import MODEL_CHOICES, normalize_model_name
from centralized.preprocess import collect_loso_samples


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _run(cmd, cwd=None):
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def _parse_list(value):
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_int_list(value):
    items = _parse_list(value)
    parsed = []
    for item in items:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            continue
    return parsed


def _dedupe_keep_order(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _load_yaml_config(path):
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    defaults = {}
    defaults["dataset"] = cfg.get("dataset")
    defaults["num_classes"] = cfg.get("num_classes")
    model_cfg = cfg.get("model", {}) or {}
    defaults["model_name"] = model_cfg.get("name") or cfg.get("model_name")

    paths = cfg.get("paths", {}) or {}
    defaults["data_root"] = paths.get("data_root")
    defaults["metadata_base"] = paths.get("metadata_base")
    defaults["features_base"] = paths.get("features_base")
    defaults["logs_root"] = paths.get("logs_root")
    defaults["results_root"] = paths.get("results_root")

    loso = cfg.get("loso", {}) or {}
    defaults["split_by"] = loso.get("split_by")
    defaults["val_speaker"] = loso.get("val_speaker")
    defaults["test_speakers"] = loso.get("test_speakers")
    defaults["val_session"] = loso.get("val_session")
    defaults["test_sessions"] = loso.get("test_sessions")
    defaults["seed"] = loso.get("seed")
    defaults["ignore_length"] = loso.get("ignore_length")
    defaults["exp_prefix"] = loso.get("exp_prefix")
    defaults["run_id"] = loso.get("run_id")
    defaults["clean_metadata"] = loso.get("clean_metadata")
    defaults["clean_features"] = loso.get("clean_features")
    defaults["local_only"] = loso.get("local_only")

    train = cfg.get("train", {}) or {}
    defaults["model_name"] = train.get("model_name") or defaults["model_name"]
    defaults["epochs"] = train.get("epochs")
    defaults["batch_size"] = train.get("batch_size")
    defaults["modality"] = train.get("modality")

    return defaults


def _read_results_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    metric_map = {}
    for row in rows:
        metric = row.get("Metric")
        mean = row.get("Mean")
        std = row.get("Std")
        if metric and mean is not None:
            metric_map[metric] = {
                "mean": float(mean),
                "std": float(std) if std is not None else 0.0,
            }
    return metric_map


def _mean_std(values):
    count = len(values)
    if count == 0:
        return 0.0, 0.0
    mean = sum(values) / count
    if count < 2:
        return mean, 0.0
    var = sum((val - mean) ** 2 for val in values) / (count - 1)
    return mean, math.sqrt(var)


def _expected_feature_files(dataset, num_classes):
    if dataset == "IEMOCAP":
        prefix = "IEMOCAP_BERT_Wav2Vec2"
    elif dataset == "MSP-IMPROV":
        prefix = "MSPIMPROV_BERT_Wav2Vec2"
    elif dataset == "ESD":
        prefix = "ESD_BERT_Wav2Vec2"
    elif dataset == "MELD":
        prefix = "MELD_BERT_Wav2Vec2"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return {
        "train": f"{prefix}_train.pkl",
        "val": f"{prefix}_val.pkl",
        "test": f"{prefix}_test.pkl",
    }


def _ensure_centralized_feature_names(client_feat_dir, dataset, num_classes):
    expected = _expected_feature_files(dataset, num_classes)
    if all(os.path.isfile(os.path.join(client_feat_dir, fname)) for fname in expected.values()):
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
        src_name = mapping.get(split)
        if not src_name:
            ok = False
            continue
        src_path = os.path.join(client_feat_dir, src_name)
        if not os.path.isfile(src_path):
            ok = False
            continue
        try:
            os.symlink(src_path, expected_path)
        except OSError:
            shutil.copy2(src_path, expected_path)
    return ok


def _resolve_clients_root(base_dir):
    if not base_dir:
        return base_dir
    if list(Path(base_dir).glob("client_*")):
        return base_dir
    nested = os.path.join(base_dir, "clients")
    if list(Path(nested).glob("client_*")):
        return nested
    return base_dir


def _weighted_mean_std(values, weights):
    if not values:
        return 0.0, 0.0
    total = float(sum(weights))
    if total <= 0:
        return 0.0, 0.0
    mean = sum(val * weight for val, weight in zip(values, weights)) / total
    if len(values) < 2:
        return mean, 0.0
    var = sum(weight * (val - mean) ** 2 for val, weight in zip(values, weights)) / total
    return mean, math.sqrt(var)


def _aggregate_fold_results(rows):
    keys = ["WA", "UA", "WF1", "UF1"]
    totals = [int(row["test_samples"]) for row in rows]
    total_samples = sum(totals)
    if total_samples <= 0:
        macro = {key: 0.0 for key in keys}
        weighted = {key: 0.0 for key in keys}
        macro_std = {key: 0.0 for key in keys}
        weighted_std = {key: 0.0 for key in keys}
        return macro, macro_std, weighted, weighted_std, total_samples

    macro = {}
    macro_std = {}
    weighted = {}
    weighted_std = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        mean, std = _mean_std(values)
        macro[key] = mean
        macro_std[key] = std

        w_mean, w_std = _weighted_mean_std(values, totals)
        weighted[key] = w_mean
        weighted_std[key] = w_std

    return macro, macro_std, weighted, weighted_std, total_samples


def parse_args():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=None)
    known_args, _ = base_parser.parse_known_args()
    cfg_defaults = _load_yaml_config(known_args.config)

    parser = argparse.ArgumentParser(description="Centralized LOSO runner for IEMOCAP/MSP-IMPROV.")
    parser.add_argument("--config", type=str, default=known_args.config)
    parser.add_argument("--dataset", type=str, default="IEMOCAP", choices=["IEMOCAP", "MSP-IMPROV"])
    parser.add_argument("--num_classes", type=int, default=4, choices=[4])
    parser.add_argument(
        "--model_name",
        type=str,
        default="fedalmer",
        help=f"Model name. Supported: {', '.join(MODEL_CHOICES)}",
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--metadata_base", type=str, default="metadata/IEMOCAP_loso_centralized")
    parser.add_argument("--features_base", type=str, default="features/IEMOCAP_loso_centralized")
    parser.add_argument("--logs_root", type=str, default="logs/centralized")
    parser.add_argument("--results_root", type=str, default="results")

    parser.add_argument("--split_by", type=str, default="session", choices=["speaker", "session"])
    parser.add_argument("--val_speaker", type=str, default=None)
    parser.add_argument("--test_speakers", type=str, default=None)
    parser.add_argument("--val_session", type=int, default=None)
    parser.add_argument("--test_sessions", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore_length", type=int, default=0)
    parser.add_argument("--exp_prefix", type=str, default="IEMOCAP_loso_centralized")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--clean_metadata", action="store_true")
    parser.add_argument("--clean_features", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--modality", type=str, default="both", choices=["both", "text", "audio"])

    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_features", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--local_only", action="store_true",
                        help="Train a separate local model per client and evaluate on the shared test set.")

    for key, value in cfg_defaults.items():
        if value is not None:
            parser.set_defaults(**{key: value})
    args = parser.parse_args()
    args.model_name = normalize_model_name(args.model_name)
    if not args.data_root:
        parser.error("Missing required args: data_root")
    return args


def main():
    args = parse_args()

    if isinstance(args.val_speaker, (list, tuple)):
        args.val_speaker = args.val_speaker[0] if args.val_speaker else None

    speaker_map, session_map = collect_loso_samples(
        dataset=args.dataset,
        data_root=args.data_root,
        ignore_length=args.ignore_length,
        seed=args.seed,
    )

    speakers = sorted(speaker_map.keys())
    sessions = sorted(session_map.keys())
    if args.split_by == "speaker" and not speakers:
        raise ValueError(
            f"No speakers found for {args.dataset} under data_root={args.data_root}."
        )
    if args.split_by == "session" and not sessions:
        raise ValueError(
            f"No sessions found for {args.dataset} under data_root={args.data_root}."
        )

    if (
        args.dataset == "IEMOCAP"
        and args.split_by == "speaker"
        and not args.val_speaker
        and not _parse_list(args.test_speakers)
    ):
        args.val_speaker = "Ses01F"

    if args.split_by == "speaker":
        if args.val_speaker and args.val_speaker not in speakers:
            raise ValueError(f"Unknown val speaker: {args.val_speaker}")
        test_speakers = _parse_list(args.test_speakers)
        if not test_speakers:
            if args.val_speaker:
                test_speakers = [spk for spk in speakers if spk != args.val_speaker]
            else:
                test_speakers = list(speakers)
        missing_speakers = [spk for spk in test_speakers if spk not in speakers]
        if missing_speakers:
            raise ValueError(f"Unknown test speakers: {missing_speakers}")
        if args.val_speaker:
            adjusted = []
            for test_spk in test_speakers:
                if test_spk == args.val_speaker:
                    idx = speakers.index(test_spk)
                    test_spk = speakers[(idx + 1) % len(speakers)]
                adjusted.append(test_spk)
            test_speakers = _dedupe_keep_order(adjusted)
            if not test_speakers:
                raise ValueError("No valid test speakers after applying val/test conflict rule.")
    else:
        test_sessions = _parse_int_list(args.test_sessions)
        if not test_sessions:
            test_sessions = list(sessions)
        missing_sessions = [sess for sess in test_sessions if sess not in sessions]
        if missing_sessions:
            raise ValueError(f"Unknown test sessions: {missing_sessions}")
        if args.val_session is not None and args.val_session not in sessions:
            raise ValueError(f"Unknown val session: {args.val_session}")

    run_tag = str(args.run_id).strip() if args.run_id else ""
    results = []
    client_results = []

    if args.split_by == "speaker":
        fold_items = [{"test_speaker": spk} for spk in test_speakers]
    else:
        fold_items = [{"test_session": sess} for sess in test_sessions]

    for fold in fold_items:
        test_spk = fold.get("test_speaker")
        test_session = fold.get("test_session")
        val_spk = None
        train_speakers = []
        train_sessions = []
        if args.split_by == "session":
            if args.val_session is not None:
                val_session = args.val_session
                if val_session == test_session:
                    raise ValueError("val_session must be different from test_session.")
            else:
                idx = sessions.index(test_session)
                val_session = sessions[(idx + 1) % len(sessions)]
        else:
            val_session = None
            if args.val_speaker:
                val_spk = args.val_speaker
                if test_spk == val_spk:
                    idx = speakers.index(test_spk)
                    test_spk = speakers[(idx + 1) % len(speakers)]
            else:
                idx = speakers.index(test_spk)
                val_spk = speakers[(idx + 1) % len(speakers)]

        if args.split_by == "speaker":
            fold_name = f"test_{test_spk}_val_{val_spk}"
        else:
            fold_name = f"test_session{test_session}_val_session{val_session}"

        if args.model_name == "fedalmer":
            exp_name = f"{args.exp_prefix}_{fold_name}"
        else:
            exp_name = f"{args.exp_prefix}_{args.model_name}_{fold_name}"
        if run_tag:
            if args.model_name == "fedalmer":
                exp_name = f"{args.exp_prefix}_{run_tag}_{fold_name}"
            else:
                exp_name = f"{args.exp_prefix}_{args.model_name}_{run_tag}_{fold_name}"
            metadata_dir = os.path.join(args.metadata_base, run_tag, fold_name)
            features_dir = os.path.join(args.features_base, run_tag, fold_name)
        else:
            metadata_dir = os.path.join(args.metadata_base, fold_name)
            features_dir = os.path.join(args.features_base, fold_name)

        if args.clean_metadata and os.path.isdir(metadata_dir):
            shutil.rmtree(metadata_dir, ignore_errors=True)
        if args.clean_features and os.path.isdir(features_dir):
            shutil.rmtree(features_dir, ignore_errors=True)

        if args.split_by == "speaker":
            train_speakers = [spk for spk in speakers if spk not in (val_spk, test_spk)]
            train_samples = []
            for spk in train_speakers:
                train_samples.extend(speaker_map[spk])
            val_samples = list(speaker_map[val_spk])
            test_samples = list(speaker_map[test_spk])
        else:
            train_sessions = [sess for sess in sessions if sess not in (val_session, test_session)]
            train_samples = []
            for sess in train_sessions:
                train_samples.extend(session_map[sess])
            val_samples = list(session_map[val_session])
            test_samples = list(session_map[test_session])

        rng = random.Random(args.seed)
        rng.shuffle(train_samples)
        rng.shuffle(val_samples)
        rng.shuffle(test_samples)

        if not args.local_only:
            os.makedirs(metadata_dir, exist_ok=True)
            if not args.skip_preprocess:
                with open(os.path.join(metadata_dir, "train.pkl"), "wb") as f:
                    pickle.dump(train_samples, f)
                with open(os.path.join(metadata_dir, "val.pkl"), "wb") as f:
                    pickle.dump(val_samples, f)
                with open(os.path.join(metadata_dir, "test.pkl"), "wb") as f:
                    pickle.dump(test_samples, f)

                if args.split_by == "speaker":
                    logging.info(
                        "Fold %s - Train speakers: %s | Val: %s | Test: %s",
                        fold_name, train_speakers, val_spk, test_spk
                    )
                else:
                    logging.info(
                        "Fold %s - Train sessions: %s | Val session: %s | Test session: %s",
                        fold_name, train_sessions, val_session, test_session
                    )
                logging.info(
                    "Fold %s - Train: %d | Val: %d | Test: %d",
                    fold_name, len(train_samples), len(val_samples), len(test_samples)
                )

            if not args.skip_features:
                cmd = [
                    sys.executable, "feature_extract/extract_feature.py",
                    "--dataset", args.dataset,
                    "--wav_base", args.data_root,
                    "--pkl_dir", metadata_dir,
                    "--output_dir", features_dir,
                ]
                _run(cmd)

            results_suffix = f"{exp_name}_results"
            results_path = os.path.join(args.results_root, f"{results_suffix}.csv")

            if not args.skip_train:
                cmd = [
                    sys.executable, "centralized/train.py",
                    "--data_dir", features_dir,
                    "--dataset", args.dataset,
                    "--num_classes", str(args.num_classes),
                    "--model_name", args.model_name,
                    "--epochs", str(args.epochs),
                    "--batch_size", str(args.batch_size),
                    "--modality", args.modality,
                    "--logs_root", args.logs_root,
                    "--exp_name", exp_name,
                    "--reset_logs",
                    "--results_suffix", results_suffix,
                ]
                _run(cmd)

            if not os.path.isfile(results_path):
                raise FileNotFoundError(f"Missing results file: {results_path}")

            metrics = _read_results_csv(results_path)
            row = {
                "exp_name": exp_name,
                "split_by": args.split_by,
                "test_speaker": test_spk,
                "val_speaker": val_spk,
                "test_session": test_session,
                "val_session": val_session,
                "test_samples": len(test_samples),
                "WA": metrics.get("WA", {}).get("mean", 0.0),
                "WA_std": metrics.get("WA", {}).get("std", 0.0),
                "UA": metrics.get("UA", {}).get("mean", 0.0),
                "UA_std": metrics.get("UA", {}).get("std", 0.0),
                "WF1": metrics.get("WF1", {}).get("mean", 0.0),
                "WF1_std": metrics.get("WF1", {}).get("std", 0.0),
                "UF1": metrics.get("UF1", {}).get("mean", 0.0),
                "UF1_std": metrics.get("UF1", {}).get("std", 0.0),
                "results_path": os.path.abspath(results_path),
            }
            results.append(row)
        else:
            if args.split_by == "speaker":
                client_ids = list(train_speakers)
                client_key = "speaker"
            else:
                client_ids = list(train_sessions)
                client_key = "session"

            clients_root = _resolve_clients_root(metadata_dir)
            features_root = _resolve_clients_root(features_dir)
            existing_meta_clients = []
            existing_feat_clients = []
            if os.path.isdir(clients_root):
                existing_meta_clients = sorted(Path(clients_root).glob("client_*"))
            if os.path.isdir(features_root):
                existing_feat_clients = sorted(Path(features_root).glob("client_*"))

            client_entries = []
            if args.skip_features:
                if not existing_feat_clients:
                    raise FileNotFoundError(
                        f"Missing client features under {features_root} "
                        f"(checked {features_dir} and {os.path.join(features_dir, 'clients')})"
                    )
                for client_dir in existing_feat_clients:
                    client_entries.append({
                        "client_name": client_dir.name,
                        "client_id": client_dir.name,
                        "client_key": "existing",
                        "meta_dir": os.path.join(clients_root, client_dir.name),
                        "feat_dir": str(client_dir),
                        "samples": None,
                    })
            elif args.skip_preprocess and existing_meta_clients:
                for client_dir in existing_meta_clients:
                    client_entries.append({
                        "client_name": client_dir.name,
                        "client_id": client_dir.name,
                        "client_key": "existing",
                        "meta_dir": str(client_dir),
                        "feat_dir": os.path.join(features_root, client_dir.name),
                        "samples": None,
                    })
            else:
                for client_id in client_ids:
                    if args.split_by == "speaker":
                        client_name = f"client_{client_id}"
                        client_samples = list(speaker_map[client_id])
                    else:
                        client_name = f"client_session{client_id}"
                        client_samples = list(session_map[client_id])
                    client_entries.append({
                        "client_name": client_name,
                        "client_id": client_id,
                        "client_key": client_key,
                        "meta_dir": os.path.join(clients_root, client_name),
                        "feat_dir": os.path.join(features_root, client_name),
                        "samples": client_samples,
                    })

            fold_client_rows = []
            for entry in client_entries:
                client_name = entry["client_name"]
                client_id = entry["client_id"]
                client_key = entry["client_key"]
                client_meta_dir = entry["meta_dir"]
                client_feat_dir = entry["feat_dir"]
                client_samples = entry["samples"]

                if not args.skip_preprocess:
                    os.makedirs(client_meta_dir, exist_ok=True)
                    with open(os.path.join(client_meta_dir, "train.pkl"), "wb") as f:
                        pickle.dump(client_samples, f)
                    with open(os.path.join(client_meta_dir, "val.pkl"), "wb") as f:
                        pickle.dump(val_samples, f)
                    with open(os.path.join(client_meta_dir, "test.pkl"), "wb") as f:
                        pickle.dump(test_samples, f)
                elif not args.skip_features and not os.path.isdir(client_meta_dir):
                    raise FileNotFoundError(f"Missing client metadata: {client_meta_dir}")

                if not args.skip_features:
                    cmd = [
                        sys.executable, "feature_extract/extract_feature.py",
                        "--dataset", args.dataset,
                        "--wav_base", args.data_root,
                        "--pkl_dir", client_meta_dir,
                        "--output_dir", client_feat_dir,
                    ]
                    _run(cmd)
                elif not os.path.isdir(client_feat_dir):
                    raise FileNotFoundError(f"Missing client features: {client_feat_dir}")
                else:
                    if not _ensure_centralized_feature_names(client_feat_dir, args.dataset, args.num_classes):
                        raise FileNotFoundError(
                            f"Missing centralized feature files under {client_feat_dir}. "
                            "Expected train/val/test pkl or federated feature names."
                        )

                client_exp_name = f"{exp_name}_{client_name}"
                results_suffix = f"{client_exp_name}_results"
                results_path = os.path.join(args.results_root, f"{results_suffix}.csv")

                if not args.skip_train:
                    cmd = [
                        sys.executable, "centralized/train.py",
                        "--data_dir", client_feat_dir,
                        "--dataset", args.dataset,
                        "--num_classes", str(args.num_classes),
                        "--model_name", args.model_name,
                        "--epochs", str(args.epochs),
                        "--batch_size", str(args.batch_size),
                        "--modality", args.modality,
                        "--logs_root", args.logs_root,
                        "--exp_name", client_exp_name,
                        "--reset_logs",
                        "--results_suffix", results_suffix,
                    ]
                    _run(cmd)

                if not os.path.isfile(results_path):
                    raise FileNotFoundError(f"Missing results file: {results_path}")

                metrics = _read_results_csv(results_path)
                row = {
                    "exp_name": client_exp_name,
                    "split_by": args.split_by,
                    "test_speaker": test_spk,
                    "val_speaker": val_spk,
                    "test_session": test_session,
                    "val_session": val_session,
                    "client_id": client_id,
                    "client_type": client_key,
                    "test_samples": len(test_samples),
                    "WA": metrics.get("WA", {}).get("mean", 0.0),
                    "WA_std": metrics.get("WA", {}).get("std", 0.0),
                    "UA": metrics.get("UA", {}).get("mean", 0.0),
                    "UA_std": metrics.get("UA", {}).get("std", 0.0),
                    "WF1": metrics.get("WF1", {}).get("mean", 0.0),
                    "WF1_std": metrics.get("WF1", {}).get("std", 0.0),
                    "UF1": metrics.get("UF1", {}).get("mean", 0.0),
                    "UF1_std": metrics.get("UF1", {}).get("std", 0.0),
                    "results_path": os.path.abspath(results_path),
                }
                fold_client_rows.append(row)
                client_results.append(row)

            metrics_keys = ["WA", "UA", "WF1", "UF1"]
            fold_summary = {}
            for key in metrics_keys:
                values = [float(r.get(key, 0.0)) for r in fold_client_rows]
                mean, std = _mean_std(values)
                fold_summary[key] = mean
                fold_summary[f"{key}_std"] = std

            results.append({
                "exp_name": exp_name,
                "split_by": args.split_by,
                "test_speaker": test_spk,
                "val_speaker": val_spk,
                "test_session": test_session,
                "val_session": val_session,
                "test_samples": len(test_samples),
                "WA": fold_summary["WA"],
                "WA_std": fold_summary["WA_std"],
                "UA": fold_summary["UA"],
                "UA_std": fold_summary["UA_std"],
                "WF1": fold_summary["WF1"],
                "WF1_std": fold_summary["WF1_std"],
                "UF1": fold_summary["UF1"],
                "UF1_std": fold_summary["UF1_std"],
                "results_path": "",
            })

    macro, macro_std, weighted, weighted_std, total_samples = _aggregate_fold_results(results)

    exp_prefix_for_logs = args.exp_prefix
    if args.model_name != "fedalmer":
        exp_prefix_for_logs = f"{args.exp_prefix}_{args.model_name}"

    loso_dir = os.path.join(args.logs_root, "loso", exp_prefix_for_logs)
    if run_tag:
        loso_dir = os.path.join(loso_dir, run_tag)
    os.makedirs(loso_dir, exist_ok=True)

    per_fold_path = os.path.join(loso_dir, "per_fold_metrics.csv")
    with open(per_fold_path, "w", newline="") as f:
        fieldnames = [
            "exp_name", "split_by",
            "test_speaker", "val_speaker",
            "test_session", "val_session",
            "test_samples",
            "WA", "WA_std",
            "UA", "UA_std",
            "WF1", "WF1_std",
            "UF1", "UF1_std",
            "results_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    if args.local_only and client_results:
        per_client_path = os.path.join(loso_dir, "per_client_metrics.csv")
        with open(per_client_path, "w", newline="") as f:
            fieldnames = [
                "exp_name", "split_by",
                "test_speaker", "val_speaker",
                "test_session", "val_session",
                "client_id", "client_type",
                "test_samples",
                "WA", "WA_std",
                "UA", "UA_std",
                "WF1", "WF1_std",
                "UF1", "UF1_std",
                "results_path",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in client_results:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "split_by": args.split_by,
        "val_speaker": args.val_speaker if args.split_by == "speaker" else None,
        "val_session": args.val_session if args.split_by == "session" else None,
        "num_folds": len(results),
        "local_only": bool(args.local_only),
        "macro_avg": macro,
        "macro_std": macro_std,
        "weighted_avg": weighted,
        "weighted_std": weighted_std,
        "total_test_samples": total_samples,
    }
    summary_path = os.path.join(loso_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Centralized LOSO summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
