import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from federated.evaluate import evaluate_checkpoint
from src.model_factory import normalize_model_name


def _run(cmd):
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_yaml_config(path):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _list_clients(clients_root):
    return sorted(
        p.name for p in Path(clients_root).glob("client_*")
        if p.is_dir()
    )


def _with_run_id(path, run_id):
    if not path or not run_id:
        return path
    path = os.path.normpath(path)
    if os.path.basename(path) == run_id:
        return path
    return os.path.join(path, run_id)


def _resolve_best_checkpoint(ckpt_root):
    if not os.path.isdir(ckpt_root):
        return None
    meta_path = os.path.join(ckpt_root, "best_meta.pt")
    if os.path.isfile(meta_path):
        meta = torch.load(meta_path, map_location="cpu")
        best_path = meta.get("best_path")
        if best_path and os.path.isfile(best_path):
            return best_path
        best_round = meta.get("best_round")
        if best_round is not None:
            candidate = os.path.join(ckpt_root, f"global_round_best_{best_round}.pt")
            if os.path.isfile(candidate):
                return candidate

    best_candidates = []
    for name in os.listdir(ckpt_root):
        if not (name.startswith("global_round_best_") and name.endswith(".pt")):
            continue
        parts = name.split("_")
        if len(parts) < 4:
            continue
        round_part = parts[-1].replace(".pt", "")
        try:
            round_idx = int(round_part)
        except ValueError:
            continue
        best_candidates.append((round_idx, os.path.join(ckpt_root, name)))
    if not best_candidates:
        return None
    best_candidates.sort(key=lambda x: x[0], reverse=True)
    return best_candidates[0][1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MELD preprocess + extract + train + eval (federated)."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_features", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = _load_yaml_config(args.config)

    dataset = cfg.get("dataset")
    num_classes = cfg.get("num_classes")
    if dataset != "MELD":
        raise ValueError("run_meld_pipeline only supports dataset=MELD.")
    if not num_classes:
        raise ValueError("Config must include num_classes.")

    run_id = args.run_id or cfg.get("run_id")

    paths = cfg.get("paths", {}) or {}
    data_root = paths.get("data_root") or (cfg.get("prepare", {}) or {}).get("data_root")
    clients_root = paths.get("clients_root")
    features_root = paths.get("features_root")
    checkpoints_root = paths.get("checkpoints_root", "checkpoints/federated")
    logs_root = paths.get("logs_root", "logs/federated")

    if not clients_root or not features_root:
        raise ValueError("paths.clients_root and paths.features_root are required.")

    clients_root = _with_run_id(clients_root, run_id)
    features_root = _with_run_id(features_root, run_id)

    preprocess_cfg = cfg.get("preprocess", {}) or {}
    extract_cfg = cfg.get("extract", {}) or {}
    fed_cfg = cfg.get("federated", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    model_name = normalize_model_name(
        model_cfg.get("name") or fed_cfg.get("model_name") or "fedalmer"
    )
    modality = fed_cfg.get("modality") or "both"

    num_clients = preprocess_cfg.get("num_clients") or fed_cfg.get("num_clients")
    seed = preprocess_cfg.get("seed", fed_cfg.get("seed", 42))

    meld_dir = preprocess_cfg.get("meld_dir") or paths.get("meld_dir")
    meld_out_dir = (cfg.get("prepare", {}) or {}).get("out_dir")
    preprocess_out_dir = _with_run_id(preprocess_cfg.get("out_dir") or clients_root, run_id)

    extract_clients_root = _with_run_id(extract_cfg.get("clients_root") or clients_root, run_id)
    extract_features_root = _with_run_id(extract_cfg.get("features_root") or features_root, run_id)
    audio_root = extract_cfg.get("audio_root") or data_root

    if not args.skip_preprocess:
        cmd = [
            sys.executable, "federated/preprocess.py",
            "--dataset", "MELD",
            "--data_root", data_root,
            "--out_dir", preprocess_out_dir,
            "--split_by", preprocess_cfg.get("split_by", "iid"),
            "--min_labeled_per_client", str(preprocess_cfg.get("min_labeled_per_client", 1)),
            "--seed", str(seed),
        ]
        if num_clients:
            cmd += ["--num_clients", str(num_clients)]
        if preprocess_cfg.get("dirichlet_alpha") is not None:
            cmd += ["--dirichlet_alpha", str(preprocess_cfg["dirichlet_alpha"])]
        if meld_dir:
            cmd += ["--meld_dir", meld_dir]
        if meld_out_dir:
            cmd += ["--meld_out_dir", meld_out_dir]
        _run(cmd)

    if args.skip_preprocess and not os.path.isdir(preprocess_out_dir):
        raise FileNotFoundError(f"clients_root not found: {preprocess_out_dir}")

    if not args.skip_features:
        clients = _list_clients(extract_clients_root)
        if num_clients:
            clients = clients[:num_clients]
        if not clients:
            raise ValueError(f"No clients found under {extract_clients_root}")
        for client_id in clients:
            client_dir = os.path.join(extract_clients_root, client_id)
            out_dir = os.path.join(extract_features_root, client_id)
            cmd = [
                sys.executable, "feature_extract/extract_feature.py",
                "--dataset", "MELD",
                "--client_dir", client_dir,
                "--out_dir", out_dir,
                "--wav_base", audio_root,
            ]
            _run(cmd)

    if args.exp_name:
        exp_name = args.exp_name
    elif run_id:
        exp_name = run_id if model_name == "fedalmer" else f"{run_id}_{model_name}"
    else:
        exp_name = f"{dataset}_federated" if model_name == "fedalmer" else f"{dataset}_{model_name}_federated"

    if not args.skip_train:
        cmd = [
            sys.executable, "-m", "federated.run_federated",
            "--config", args.config,
            "--dataset", dataset,
            "--num_classes", str(num_classes),
            "--model_name", model_name,
            "--modality", str(modality),
            "--clients_root", clients_root,
            "--features_root", features_root,
            "--exp_name", exp_name,
        ]
        if run_id:
            cmd += ["--run_id", run_id]
        if num_clients:
            cmd += ["--num_clients", str(num_clients)]
        _run(cmd)

    if args.skip_eval:
        print("Skip evaluation enabled.")
        return

    stage = "pretrain"
    ckpt_root = os.path.join(checkpoints_root, exp_name, stage)

    latest_ckpt = os.path.join(ckpt_root, "global_round_latest.pt")
    if os.path.isfile(latest_ckpt):
        out_dir = os.path.join(logs_root, exp_name, "eval", stage, "latest")
        metrics = evaluate_checkpoint(
            dataset=dataset,
            num_classes=num_classes,
            model_name=model_name,
            modality=modality,
            clients_root=clients_root,
            features_root=features_root,
            checkpoint=latest_ckpt,
            batch_size=fed_cfg.get("batch_size", 128),
            num_clients=num_clients,
            num_workers=fed_cfg.get("num_workers", 1),
            output_dir=out_dir,
            reset_logs=True,
        )
        print("[EVAL] Latest", metrics)
    else:
        print(f"[WARN] Latest checkpoint not found: {latest_ckpt}")

    best_ckpt = _resolve_best_checkpoint(ckpt_root)
    if best_ckpt:
        out_dir = os.path.join(logs_root, exp_name, "eval", stage, "best")
        metrics = evaluate_checkpoint(
            dataset=dataset,
            num_classes=num_classes,
            model_name=model_name,
            modality=modality,
            clients_root=clients_root,
            features_root=features_root,
            checkpoint=best_ckpt,
            batch_size=fed_cfg.get("batch_size", 128),
            num_clients=num_clients,
            num_workers=fed_cfg.get("num_workers", 1),
            output_dir=out_dir,
            reset_logs=True,
        )
        print("[EVAL] Best", metrics)
    else:
        print(f"[WARN] Best checkpoint not found under {ckpt_root}")


if __name__ == "__main__":
    main()
