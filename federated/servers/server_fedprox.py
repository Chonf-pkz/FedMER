import os
import time
import multiprocessing as mp

import torch
from federated.clients.client_fedprox import local_train
from federated.clients.common import evaluate_global, build_model
from federated.utils import (
    append_csv,
    ensure_dir,
    list_clients,
    load_state_dict,
    save_global_checkpoint,
    set_seed,
)


CLIENT_FIELDS = [
    "stage",
    "round",
    "client_id",
    "num_samples",
    "train_labeled",
    "val",
    "test",
    "loss",
    "test_WA",
    "test_UA",
    "test_WF1",
    "test_UF1",
]

ROUND_FIELDS = [
    "stage",
    "round",
    "num_clients",
    "val_samples",
    "val_WA",
    "val_UA",
    "val_WF1",
    "val_UF1",
]


def _train_worker(payload):
    return local_train(**payload)


def _eval_worker(payload):
    return evaluate_global(**payload)


def _init_global_state(cfg, resume_path=None, init_state_dict=None):
    if resume_path:
        return load_state_dict(resume_path)
    if init_state_dict is not None:
        return init_state_dict
    model = build_model(cfg["num_classes"], cfg["model_name"], cfg=cfg)
    return model.state_dict()


def _aggregate_eval(eval_results, prefix):
    total = sum(r["num_samples"] for r in eval_results)
    if total <= 0:
        return {
            f"{prefix}_WA": 0.0,
            f"{prefix}_UA": 0.0,
            f"{prefix}_WF1": 0.0,
            f"{prefix}_UF1": 0.0,
            f"{prefix}_samples": 0,
        }

    def _weighted(metric):
        return sum(r["metrics"][metric] * r["num_samples"] for r in eval_results) / total

    return {
        f"{prefix}_WA": _weighted("WA"),
        f"{prefix}_UA": _weighted("UA"),
        f"{prefix}_WF1": _weighted("WF1"),
        f"{prefix}_UF1": _weighted("UF1"),
        f"{prefix}_samples": total,
    }


def _get_metric(metrics, name):
    if name in metrics:
        return metrics[name]
    return metrics.get(f"test_{name}")


def _fedavg(state_dicts, weights):
    if not state_dicts:
        return {}
    if len(weights) != len(state_dicts):
        raise ValueError("len(weights) must equal len(state_dicts)")

    weights = [float(w) for w in weights]
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0] * len(weights)
        total_weight = float(len(weights))

    keys = state_dicts[0].keys()
    averaged = {}
    for k in keys:
        t0 = state_dicts[0][k]
        if torch.is_floating_point(t0):
            acc = torch.zeros_like(t0.detach().cpu(), dtype=torch.float32)
            for sd, w in zip(state_dicts, weights):
                acc += sd[k].detach().cpu().to(torch.float32) * w
            out = acc / total_weight
            out = out.to(dtype=t0.dtype)
            averaged[k] = out
        else:
            averaged[k] = t0.detach().cpu()
    return averaged


def run_stage(
    stage,
    rounds,
    cfg,
    clients_root,
    features_root,
    ckpt_root,
    log_root,
    num_workers=1,
    resume_path=None,
    init_state_dict=None,
):
    if rounds <= 0:
        return init_state_dict

    ensure_dir(ckpt_root)
    ensure_dir(log_root)

    clients = list_clients(clients_root, cfg.get("num_clients"))
    if not clients:
        raise ValueError(f"No clients found under {clients_root}")

    global_state = _init_global_state(cfg, resume_path=resume_path, init_state_dict=init_state_dict)

    client_log_path = os.path.join(log_root, "client_metrics.csv")
    round_log_path = os.path.join(log_root, "round_metrics.csv")

    best_metric = float("-inf")
    best_round = -1
    metric_key = cfg.get("best_metric", "global_UA")
    metric_key_resolved = metric_key
    if metric_key.startswith("global_"):
        metric_key_resolved = f"val_{metric_key[len('global_'):]}"

    for round_idx in range(1, rounds + 1):
        tmp_dir = os.path.join(ckpt_root, "_tmp_clients")
        ensure_dir(tmp_dir)
        set_seed(cfg.get("seed"))

        payloads = []
        for client_id in clients:
            features_dir = os.path.join(features_root, client_id)
            save_path = os.path.join(tmp_dir, f"{client_id}_round{round_idx}.pt")
            payloads.append(
                {
                    "client_id": client_id,
                    "stage": stage,
                    "cfg": cfg,
                    "features_dir": features_dir,
                    "round_idx": round_idx,
                    "init_state_dict": global_state,
                    "save_path": save_path,
                }
            )

        if num_workers and num_workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_workers) as pool:
                results = pool.map(_train_worker, payloads)
        else:
            results = [_train_worker(payload) for payload in payloads]

        state_dicts = [result["state_dict"] for result in results]
        weights = [result["num_samples"] for result in results]

        global_state = _fedavg(state_dicts, weights)
        global_ckpt_path = os.path.join(ckpt_root, "global_round_latest.pt")
        save_global_checkpoint(
            global_ckpt_path,
            global_state,
            meta={
                "stage": stage,
                "round": round_idx,
                "fl_method": "fedprox",
                "model_name": cfg["model_name"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "clients": clients,
            },
        )

        for result in results:
            metrics = result["metrics"]
            counts = result["counts"]
            append_csv(
                client_log_path,
                CLIENT_FIELDS,
                {
                    "stage": stage,
                    "round": round_idx,
                    "client_id": result["client_id"],
                    "num_samples": result["num_samples"],
                    "train_labeled": counts["train_labeled"],
                    "val": counts["val"],
                    "test": counts["test"],
                    "loss": metrics.get("loss", None),
                    "test_WA": _get_metric(metrics, "WA"),
                    "test_UA": _get_metric(metrics, "UA"),
                    "test_WF1": _get_metric(metrics, "WF1"),
                    "test_UF1": _get_metric(metrics, "UF1"),
                },
            )

        eval_payloads = []
        for client_id in clients:
            features_dir = os.path.join(features_root, client_id)
            eval_payloads.append(
                {
                    "client_id": client_id,
                    "cfg": cfg,
                    "features_dir": features_dir,
                    "state_dict": global_state,
                    "split": "val",
                }
            )
        if num_workers and num_workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_workers) as pool:
                eval_results = pool.map(_eval_worker, eval_payloads)
        else:
            eval_results = [_eval_worker(payload) for payload in eval_payloads]

        agg = _aggregate_eval(eval_results, "val")
        append_csv(
            round_log_path,
            ROUND_FIELDS,
            {
                "stage": stage,
                "round": round_idx,
                "num_clients": len(clients),
                "val_samples": agg["val_samples"],
                "val_WA": agg["val_WA"],
                "val_UA": agg["val_UA"],
                "val_WF1": agg["val_WF1"],
                "val_UF1": agg["val_UF1"],
            },
        )

        metric_value = agg.get(metric_key_resolved, None)
        if metric_value is not None and metric_value > best_metric:
            best_metric = float(metric_value)
            best_round = round_idx
            best_path = os.path.join(ckpt_root, f"global_round_best_{best_round}.pt")
            save_global_checkpoint(
                best_path,
                global_state,
                meta={
                    "stage": stage,
                    "round": round_idx,
                    "fl_method": "fedprox",
                    "model_name": cfg["model_name"],
                    "metric": metric_key,
                    "metric_value": best_metric,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            torch.save(
                {
                    "best_round": best_round,
                    "best_metric": best_metric,
                    "best_path": best_path,
                },
                os.path.join(ckpt_root, "best_meta.pt"),
            )

    return global_state
