import math
import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from centralized.train import combined_loss
from src.model_factory import get_model_display_name, normalize_model_name
from src.utils.utils import apply_modality_mask, train_and_evaluate, device as train_device

from federated.clients.common import (
    _compute_class_weights,
    build_model,
    load_client_datasets,
)
from federated.fedproc import (
    compute_class_prototypes,
    extract_representation,
    prototype_contrastive_loss,
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
    global_prototypes=None,
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

    print(f"[Round {round_idx}] Client {client_id} - Stage {stage} FedProc training...", flush=True)

    datasets = load_client_datasets(features_dir)
    counts = datasets["counts"]

    model_name = normalize_model_name(cfg["model_name"])
    model_display_name = get_model_display_name(model_name)
    model = build_model(cfg["num_classes"], model_name=model_name, cfg=cfg)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict, strict=False)

    class_weights = _compute_class_weights(datasets["train_dataset"], cfg["num_classes"]).to(train_device)
    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    fedproc_lambda = float(cfg.get("fedproc_lambda", 1.0))
    fedproc_temperature = float(cfg.get("fedproc_temperature", 0.5))
    if fedproc_lambda < 0:
        raise ValueError("fedproc_lambda must be non-negative")
    if fedproc_temperature <= 0:
        raise ValueError("fedproc_temperature must be greater than zero")

    def loss_fn(out, y, conf):
        classification_loss = combined_loss(out, y, ce_loss_fn, confidences=conf)
        if classification_loss.ndim > 0:
            classification_loss = classification_loss.mean()
        if not global_prototypes or fedproc_lambda == 0:
            return classification_loss
        prototype_loss = prototype_contrastive_loss(
            extract_representation(out),
            y,
            global_prototypes,
            temperature=fedproc_temperature,
            sample_weights=conf,
        )
        return classification_loss + fedproc_lambda * prototype_loss

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
        "fl_method": "fedproc",
        "fedproc_lambda": fedproc_lambda,
        "fedproc_temperature": fedproc_temperature,
        "fedproc_global_classes": len(global_prototypes or {}),
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
    model.load_state_dict(state_dict, strict=False)
    local_prototypes, prototype_counts = compute_class_prototypes(
        model=model,
        dataset=datasets["train_dataset"],
        batch_size=cfg["batch_size"],
        device=train_device,
        modality=cfg.get("modality", "both"),
        modality_mask_fn=apply_modality_mask,
    )
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
        "local_prototypes": local_prototypes,
        "prototype_counts": prototype_counts,
        "metrics": metrics,
        "counts": counts,
    }
