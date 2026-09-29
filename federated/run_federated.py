import argparse
import csv
import json
import os
import sys

import torch
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from federated.evaluate import evaluate_checkpoint
from federated.server import run_stage
from src.model_factory import MODEL_CHOICES, normalize_model_name


def _parse_seeds(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        seeds = []
        for item in value:
            if item is None:
                continue
            if not str(item).strip():
                continue
            seeds.append(int(item))
        return seeds
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return [int(item) for item in items]


def _mean_std(values):
    count = len(values)
    if count == 0:
        return 0.0, 0.0
    mean = sum(values) / count
    if count < 2:
        return mean, 0.0
    var = sum((val - mean) ** 2 for val in values) / (count - 1)
    return mean, var ** 0.5


def _build_cfg(args, exp_name, local_epochs, seed=None):
    seed_value = args.seed if seed is None else seed
    return {
        "dataset": args.dataset,
        "num_classes": args.num_classes,
        "model_name": args.model_name,
        "modality": args.modality,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "local_epochs": local_epochs,
        "seed": seed_value,
        "weight_by": args.weight_by,
        "num_clients": args.num_clients,
        "exp_name": exp_name,
        "best_metric": args.best_metric,
        "fl_method": args.fl_method,
        "fedproc_lambda": args.fedproc_lambda,
        "fedproc_temperature": args.fedproc_temperature,
        "fedprox_mu": args.fedprox_mu,
        "feddc_correction_weight": args.feddc_correction_weight,
        "feddc_drift_momentum": args.feddc_drift_momentum,
    }


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
    defaults["run_id"] = cfg.get("run_id")
    defaults["exp_name"] = cfg.get("exp_name")

    paths = cfg.get("paths", {}) or {}
    defaults["clients_root"] = paths.get("clients_root")
    defaults["features_root"] = paths.get("features_root")
    defaults["checkpoints_root"] = paths.get("checkpoints_root")
    defaults["logs_root"] = paths.get("logs_root")

    fed = cfg.get("federated", {}) or {}
    defaults["num_clients"] = fed.get("num_clients")
    defaults["rounds_pretrain"] = fed.get("rounds_pretrain")
    defaults["local_epochs_pretrain"] = fed.get("local_epochs_pretrain")
    defaults["batch_size"] = fed.get("batch_size")
    defaults["modality"] = fed.get("modality")
    defaults["lr"] = fed.get("lr")
    defaults["weight_decay"] = fed.get("weight_decay")
    defaults["seed"] = fed.get("seed")
    defaults["seeds"] = fed.get("seeds")
    defaults["weight_by"] = fed.get("weight_by")
    defaults["best_metric"] = fed.get("best_metric")
    defaults["fl_method"] = fed.get("fl_method")
    defaults["model_name"] = fed.get("model_name") or defaults["model_name"]
    defaults["fedproc_lambda"] = fed.get("fedproc_lambda")
    defaults["fedproc_temperature"] = fed.get("fedproc_temperature")
    defaults["fedprox_mu"] = fed.get("fedprox_mu")
    defaults["feddc_correction_weight"] = fed.get("feddc_correction_weight")
    defaults["feddc_drift_momentum"] = fed.get("feddc_drift_momentum")
    defaults["num_workers"] = fed.get("num_workers")

    return defaults


def parse_args():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=None)
    known_args, _ = base_parser.parse_known_args()
    cfg_defaults = _load_yaml_config(known_args.config)

    parser = argparse.ArgumentParser(description="Federated training runner for multimodal SER models")
    parser.add_argument("--config", type=str, default=known_args.config, help="Path to YAML config file.")
    parser.add_argument("--dataset", type=str, choices=["IEMOCAP", "MSP-IMPROV", "ESD", "MELD"], default=None)
    parser.add_argument("--num_classes", type=int, choices=[4, 5, 7], default=None)
    parser.add_argument(
        "--model_name",
        type=str,
        default="fedalmer",
        help=f"Model name. Supported: {', '.join(MODEL_CHOICES)}",
    )
    parser.add_argument("--clients_root", type=str, default=None)
    parser.add_argument("--features_root", type=str, default=None)
    parser.add_argument("--num_clients", type=int, default=None)
    parser.add_argument("--stage", type=str, choices=["pretrain"], default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--rounds_pretrain", type=int, default=0)
    parser.add_argument("--local_epochs", type=int, default=None)
    parser.add_argument("--local_epochs_pretrain", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--modality", type=str, default="both", choices=["both", "text", "audio"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds for multi-run averaging.")
    parser.add_argument("--weight_by", type=str, default="total", choices=["total", "labeled"])
    parser.add_argument(
        "--best_metric",
        type=str,
        default="global_UA",
        choices=["global_WA", "global_UA", "global_WF1", "global_UF1"],
    )
    parser.add_argument("--fl_method", type=str, default="fedavg",
                        choices=["fedavg", "fedprox", "fednova", "fedproc", "fedbn", "feddc"],
                        help="Federated optimization method.")
    parser.add_argument("--fedproc_lambda", type=float, default=1.0,
                        help="Weight of the FedProc prototype contrastive term.")
    parser.add_argument("--fedproc_temperature", type=float, default=0.5,
                        help="Temperature for FedProc prototype similarities.")
    parser.add_argument("--fedprox_mu", type=float, default=0.0,
                        help="FedProx proximal term weight (mu).")
    parser.add_argument("--feddc_correction_weight", type=float, default=1.0,
                        help="FedDC server drift correction weight.")
    parser.add_argument("--feddc_drift_momentum", type=float, default=0.0,
                        help="FedDC drift-state momentum in [0, 1].")
    parser.add_argument("--test_speakers", type=str, default="", help="Optional tag for exp naming.")
    parser.add_argument("--resume_global", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None,
                        help="Optional run id used as exp_name when exp_name is not set.")
    parser.add_argument("--checkpoints_root", type=str, default="checkpoints/federated")
    parser.add_argument("--logs_root", type=str, default="logs/federated")
    parser.add_argument("--num_workers", type=int, default=1)

    for key, value in cfg_defaults.items():
        if value is not None:
            parser.set_defaults(**{key: value})

    args = parser.parse_args()
    args.model_name = normalize_model_name(args.model_name)
    missing = [name for name in ("dataset", "num_classes", "clients_root", "features_root")
               if getattr(args, name) in (None, "")]
    if missing:
        parser.error(f"Missing required args: {', '.join(missing)}")
    return args


def _infer_test_speakers_tag(clients_root):
    if not clients_root:
        return ""
    map_path = os.path.join(clients_root, "client_map.json")
    if not os.path.isfile(map_path):
        return ""
    try:
        with open(map_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    test_speakers = data.get("test_speakers")
    if not test_speakers:
        return ""
    if isinstance(test_speakers, list):
        return ",".join(str(spk) for spk in test_speakers)
    return str(test_speakers)


def _reset_logs(log_root):
    for filename in ("client_metrics.csv", "round_metrics.csv"):
        path = os.path.join(log_root, filename)
        if os.path.exists(path):
            os.remove(path)


def _resolve_best_checkpoint(ckpt_root):
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
    for name in os.listdir(ckpt_root):
        if name.startswith("global_round_best_") and name.endswith(".pt"):
            return os.path.join(ckpt_root, name)
    return None


def _evaluate_and_report(args, exp_name, ckpt_root, stage):
    latest_ckpt = os.path.join(ckpt_root, "global_round_latest.pt")
    best_ckpt = _resolve_best_checkpoint(ckpt_root)
    results = {}

    if os.path.isfile(latest_ckpt):
        out_dir = os.path.join(args.logs_root, exp_name, "eval", stage, "latest")
        metrics = evaluate_checkpoint(
            dataset=args.dataset,
            num_classes=args.num_classes,
            model_name=args.model_name,
            clients_root=args.clients_root,
            features_root=args.features_root,
            checkpoint=latest_ckpt,
            batch_size=args.batch_size,
            modality=args.modality,
            num_clients=args.num_clients,
            num_workers=args.num_workers,
            output_dir=out_dir,
            reset_logs=True,
        )
        print(f"[EVAL] Latest ({latest_ckpt}) -> {metrics}")
        results["latest"] = metrics

    if best_ckpt and os.path.isfile(best_ckpt):
        out_dir = os.path.join(args.logs_root, exp_name, "eval", stage, "best")
        metrics = evaluate_checkpoint(
            dataset=args.dataset,
            num_classes=args.num_classes,
            model_name=args.model_name,
            clients_root=args.clients_root,
            features_root=args.features_root,
            checkpoint=best_ckpt,
            batch_size=args.batch_size,
            modality=args.modality,
            num_clients=args.num_clients,
            num_workers=args.num_workers,
            output_dir=out_dir,
            reset_logs=True,
        )
        print(f"[EVAL] Best ({best_ckpt}) -> {metrics}")
        results["best"] = metrics

    return results


def _run_for_seed(args, exp_name, seed):
    if args.stage:
        if args.rounds is None:
            raise ValueError("--rounds is required when --stage is set")
        local_epochs = args.local_epochs or args.local_epochs_pretrain

        ckpt_root = os.path.join(args.checkpoints_root, exp_name, args.stage)
        log_root = os.path.join(args.logs_root, exp_name)
        _reset_logs(log_root)

        cfg = _build_cfg(args, exp_name, local_epochs, seed=seed)
        run_stage(
            stage=args.stage,
            rounds=args.rounds,
            cfg=cfg,
            clients_root=args.clients_root,
            features_root=args.features_root,
            ckpt_root=ckpt_root,
            log_root=log_root,
            num_workers=args.num_workers,
            resume_path=args.resume_global,
        )
        eval_results = _evaluate_and_report(args, exp_name, ckpt_root, args.stage)
        return args.stage, eval_results

    log_root = os.path.join(args.logs_root, exp_name)
    _reset_logs(log_root)

    ckpt_root = os.path.join(args.checkpoints_root, exp_name, "pretrain")
    cfg = _build_cfg(args, exp_name, args.local_epochs_pretrain, seed=seed)
    run_stage(
        stage="pretrain",
        rounds=args.rounds_pretrain,
        cfg=cfg,
        clients_root=args.clients_root,
        features_root=args.features_root,
        ckpt_root=ckpt_root,
        log_root=log_root,
        num_workers=args.num_workers,
        resume_path=args.resume_global,
    )
    eval_results = _evaluate_and_report(args, exp_name, ckpt_root, "pretrain")

    return "pretrain", eval_results


def main():
    args = parse_args()
    if not args.test_speakers:
        args.test_speakers = _infer_test_speakers_tag(args.clients_root)
    if args.exp_name:
        base_exp = args.exp_name
    elif args.run_id:
        base_exp = args.run_id
    elif args.dataset in ("IEMOCAP", "MSP-IMPROV") and args.test_speakers:
        if args.model_name == "fedalmer":
            base_exp = f"{args.dataset}_test_speaker{args.test_speakers}"
        else:
            base_exp = f"{args.dataset}_{args.model_name}_test_speaker{args.test_speakers}"
    else:
        if args.model_name == "fedalmer":
            base_exp = f"{args.dataset}_federated"
        else:
            base_exp = f"{args.dataset}_{args.model_name}_federated"

    seeds = _parse_seeds(args.seeds) or [args.seed]
    multi_seed = len(seeds) > 1
    seed_rows = []
    final_stage = None

    for seed in seeds:
        exp_name = base_exp if not multi_seed else f"{base_exp}_seed{seed}"
        stage, eval_results = _run_for_seed(args, exp_name, seed)
        final_stage = stage
        for checkpoint, metrics in eval_results.items():
            seed_rows.append({
                "seed": seed,
                "exp_name": exp_name,
                "stage": stage,
                "checkpoint": checkpoint,
                "test_samples": metrics.get("test_samples", 0),
                "global_WA": metrics.get("global_WA", 0.0),
                "global_UA": metrics.get("global_UA", 0.0),
                "global_WF1": metrics.get("global_WF1", 0.0),
                "global_UF1": metrics.get("global_UF1", 0.0),
            })

    if multi_seed and seed_rows:
        summary_root = os.path.join(args.logs_root, base_exp)
        os.makedirs(summary_root, exist_ok=True)
        seed_metrics_path = os.path.join(summary_root, "seed_metrics.csv")
        fieldnames = [
            "seed", "exp_name", "stage", "checkpoint",
            "test_samples", "global_WA", "global_UA", "global_WF1", "global_UF1",
        ]
        with open(seed_metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in seed_rows:
                writer.writerow(row)

        summary = {
            "dataset": args.dataset,
            "num_classes": args.num_classes,
            "model_name": args.model_name,
            "stage": final_stage,
            "num_seeds": len(seeds),
            "checkpoint_summary": {},
        }
        metrics_keys = ["global_WA", "global_UA", "global_WF1", "global_UF1"]
        for checkpoint in sorted({row["checkpoint"] for row in seed_rows}):
            rows = [row for row in seed_rows if row["checkpoint"] == checkpoint]
            checkpoint_summary = {}
            for key in metrics_keys:
                values = [float(row[key]) for row in rows]
                mean, std = _mean_std(values)
                checkpoint_summary[key] = {"mean": mean, "std": std}
            summary["checkpoint_summary"][checkpoint] = checkpoint_summary

        summary_path = os.path.join(summary_root, "seed_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
