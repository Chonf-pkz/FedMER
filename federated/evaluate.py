import argparse
import json
import os
import multiprocessing as mp
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from federated.client import evaluate_global
from federated.utils import append_csv, ensure_dir, list_clients, load_state_dict
from src.model_factory import MODEL_CHOICES, normalize_model_name


AGG_FIELDS = [
    "checkpoint",
    "num_clients",
    "test_samples",
    "global_WA",
    "global_UA",
    "global_WF1",
    "global_UF1",
]

def _aggregate_eval(eval_results):
    total = sum(r["num_samples"] for r in eval_results)
    if total <= 0:
        return {"test_samples": 0, "global_WA": 0.0, "global_UA": 0.0, "global_WF1": 0.0, "global_UF1": 0.0}

    def _weighted(metric):
        return sum(r["metrics"][metric] * r["num_samples"] for r in eval_results) / total

    return {
        "test_samples": total,
        "global_WA": _weighted("WA"),
        "global_UA": _weighted("UA"),
        "global_WF1": _weighted("WF1"),
        "global_UF1": _weighted("UF1"),
    }


def _reset_logs(output_dir):
    for filename in ("aggregate_metrics.csv", "aggregate_metrics.json"):
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            os.remove(path)


def _eval_worker(payload):
    return evaluate_global(**payload)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a federated global checkpoint")
    parser.add_argument("--dataset", type=str, required=True, choices=["IEMOCAP", "MSP-IMPROV", "ESD", "MELD"])
    parser.add_argument("--num_classes", type=int, required=True, choices=[4, 5, 7])
    parser.add_argument(
        "--model_name",
        type=str,
        default="fedalmer",
        help=f"Model name. Supported: {', '.join(MODEL_CHOICES)}",
    )
    parser.add_argument("--modality", type=str, default="both", choices=["both", "text", "audio"])
    parser.add_argument("--clients_root", type=str, required=True)
    parser.add_argument("--features_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_clients", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="logs/federated/eval")
    parser.add_argument("--num_workers", type=int, default=1)
    return parser.parse_args()


def evaluate_checkpoint(
    dataset,
    num_classes,
    clients_root,
    features_root,
    checkpoint,
    model_name="fedalmer",
    batch_size=128,
    modality="both",
    num_clients=None,
    num_workers=1,
    output_dir=None,
    reset_logs=True,
):
    model_name = normalize_model_name(model_name)
    state_dict = load_state_dict(checkpoint)
    cfg = {
        "dataset": dataset,
        "num_classes": num_classes,
        "model_name": model_name,
        "modality": modality,
        "batch_size": batch_size,
    }

    clients = list_clients(clients_root, num_clients)
    if not clients:
        raise ValueError(f"No clients found under {clients_root}")

    payloads = []
    for client_id in clients:
        features_dir = os.path.join(features_root, client_id)
        if not os.path.isdir(features_dir):
            raise FileNotFoundError(f"Features directory not found: {features_dir}")
        payloads.append({
            "client_id": client_id,
            "cfg": cfg,
            "features_dir": features_dir,
            "state_dict": state_dict,
            "split": "test",
        })

    if num_workers and num_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            eval_results = pool.map(_eval_worker, payloads)
    else:
        eval_results = [_eval_worker(payload) for payload in payloads]

    aggregate = _aggregate_eval(eval_results)

    if output_dir:
        ensure_dir(output_dir)
        if reset_logs:
            _reset_logs(output_dir)
        aggregate_path = os.path.join(output_dir, "aggregate_metrics.csv")
        append_csv(aggregate_path, AGG_FIELDS, {
            "checkpoint": os.path.abspath(checkpoint),
            "num_clients": len(clients),
            **aggregate,
        })

        aggregate_json_path = os.path.join(output_dir, "aggregate_metrics.json")
        with open(aggregate_json_path, "w") as f:
            json.dump({
                "checkpoint": os.path.abspath(checkpoint),
                "num_clients": len(clients),
                **aggregate,
            }, f, indent=2)

    return aggregate


def main():
    args = parse_args()
    metrics = evaluate_checkpoint(
        dataset=args.dataset,
        num_classes=args.num_classes,
        model_name=args.model_name,
        modality=args.modality,
        clients_root=args.clients_root,
        features_root=args.features_root,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        num_clients=args.num_clients,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        reset_logs=True,
    )

    print("Evaluation complete.")
    print(metrics)


if __name__ == "__main__":
    main()
