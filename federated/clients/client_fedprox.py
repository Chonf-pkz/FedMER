import os
import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from centralized.train import combined_loss
from src.model_factory import get_model_display_name, normalize_model_name
from src.utils.utils import train_and_evaluate, device as train_device

from federated.clients.common import (
    build_model,
    load_client_datasets,
    _compute_class_weights,
)
from federated.utils import set_seed


def local_train(
    client_id,
    stage,
    cfg,
    features_dir,
    round_idx,
    init_state_dict,
    save_path,
    **_unused,
):
    seed_value = cfg.get("seed")
    if seed_value is not None:
        try:
            client_offset = int(client_id.split("_")[-1])
        except ValueError:
            client_offset = 0
        seed_value = int(seed_value) + int(round_idx) + client_offset
        set_seed(seed_value)

    print(f"[Round {round_idx}] Client {client_id} - Stage {stage} starting training...", flush=True)

    datasets = load_client_datasets(features_dir)
    counts = datasets["counts"]

    model_name = normalize_model_name(cfg["model_name"])
    model_display_name = get_model_display_name(model_name)
    model = build_model(cfg["num_classes"], model_name=model_name, cfg=cfg)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict, strict=False)

    global_params = None
    if init_state_dict is not None:
        global_params = {
            k: v.detach().clone().to(train_device)
            for k, v in init_state_dict.items()
            if torch.is_floating_point(v)
        }

    class_weights = _compute_class_weights(datasets["train_dataset"], cfg["num_classes"]).to(train_device)
    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    def loss_fn(out, y, conf):
        base = combined_loss(out, y, ce_loss_fn, confidences=conf)
        mu = float(cfg.get("fedprox_mu", 0.0))
        if mu > 0.0 and global_params is not None:
            prox = 0.0
            for name, p in model.named_parameters():
                if (not p.requires_grad) or (not torch.is_floating_point(p)):
                    continue
                gp = global_params.get(name)
                if gp is None:
                    continue
                prox = prox + torch.sum((p - gp) ** 2)
            base = base + 0.5 * mu * prox
        return base

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    log_context = {
        "dataset": cfg["dataset"],
        "num_classes": cfg["num_classes"],
        "model_name": model_name,
        "model_display_name": model_display_name,
        "modality": cfg.get("modality", "both"),
        "batch_size": cfg["batch_size"],
        "group": f"{cfg['dataset']}_{cfg['num_classes']}class_{model_name}_federated",
        "project": f"{model_display_name}-Federated-{cfg['dataset']}",
        "experiment_name": cfg["exp_name"],
        "seed": seed_value,
        "stage": stage,
        "client_id": client_id,
        "round": round_idx,
        "labeled_samples": counts["train_labeled"],
        "val_samples": counts["val"],
        "test_samples": counts["test"],
    }

    metrics = train_and_evaluate(
        model,
        datasets["train_dataset"],
        datasets["val_dataset"],
        datasets["test_dataset"],
        optimizer,
        scheduler,
        loss_fn,
        epochs=cfg["local_epochs"],
        save_path=save_path,
        seed=cfg.get("seed"),
        batch_size=cfg["batch_size"],
        modality=cfg.get("modality", "both"),
        log_context=log_context,
    )

    state_dict = torch.load(save_path, map_location="cpu")
    if save_path:
        try:
            os.remove(save_path)
        except OSError:
            pass

    num_samples = counts["train_labeled"]
    local_batches = max(1, math.ceil(num_samples / float(cfg["batch_size"])))
    local_steps = int(cfg["local_epochs"]) * local_batches

    return {
        "client_id": client_id,
        "state_dict": state_dict,
        "num_samples": num_samples,
        "local_steps": local_steps,
        "metrics": metrics,
        "counts": counts,
    }
