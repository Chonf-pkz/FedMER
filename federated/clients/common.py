import os

import torch
from torch.utils.data import DataLoader

from centralized.train import build_tensor_dataset
from src.model_factory import build_model as build_ser_model
from src.model_factory import normalize_model_name
from src.utils.utils import apply_modality_mask, compute_metrics, device as train_device

from federated.utils import load_pickle


FEATURE_FILES = {
    "train_labeled": "train_labeled_features.pkl",
    "val": "val_features.pkl",
    "test": "test_features.pkl",
}


def build_model(num_classes, model_name="fedalmer", cfg=None):
    normalized_name = normalize_model_name(model_name)
    return build_ser_model(
        model_name=normalized_name,
        num_classes=num_classes,
    )


def _load_split(features_dir, split_name):
    filename = FEATURE_FILES[split_name]
    path = os.path.join(features_dir, filename)
    if not os.path.isfile(path):
        return None, path
    data = load_pickle(path)
    return data, path


def load_client_datasets(features_dir):
    labeled_samples, labeled_path = _load_split(features_dir, "train_labeled")
    if labeled_samples is None:
        raise FileNotFoundError(f"Missing labeled features: {labeled_path}")

    val_samples, val_path = _load_split(features_dir, "val")
    test_samples, test_path = _load_split(features_dir, "test")

    if val_samples is None or len(val_samples) == 0:
        print(f"[WARN] Missing or empty validation features at {val_path}; using labeled train split for val.")
        val_samples = list(labeled_samples)
    if test_samples is None or len(test_samples) == 0:
        print(f"[WARN] Missing or empty test features at {test_path}; using labeled train split for test.")
        test_samples = list(labeled_samples)

    label_map = {}
    train_dataset, label_map = build_tensor_dataset(labeled_samples, label_map=label_map)
    val_dataset, label_map = build_tensor_dataset(val_samples, label_map=label_map)
    test_dataset, label_map = build_tensor_dataset(test_samples, label_map=label_map)

    counts = {
        "train_labeled": len(labeled_samples),
        "val": len(val_samples),
        "test": len(test_samples),
    }

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "counts": counts,
    }


def _compute_class_weights(train_dataset, num_classes):
    train_labels = train_dataset.tensors[2]
    class_counts = torch.bincount(train_labels, minlength=num_classes).float()
    class_counts[class_counts == 0] = 1.0
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes
    return class_weights


def evaluate_global(client_id, cfg, features_dir, state_dict, split="test"):
    labeled_samples, labeled_path = _load_split(features_dir, "train_labeled")
    split_samples, split_path = _load_split(features_dir, split)

    if labeled_samples is None:
        raise FileNotFoundError(f"Missing labeled features: {labeled_path}")
    if split_samples is None or len(split_samples) == 0:
        print(
            f"[WARN] Missing or empty {split} features at {split_path}; "
            "using labeled train split for evaluation."
        )
        split_samples = list(labeled_samples)

    label_map = {}
    _, label_map = build_tensor_dataset(labeled_samples, label_map=label_map)
    eval_dataset, _ = build_tensor_dataset(split_samples, label_map=label_map)

    model = build_model(
        cfg["num_classes"],
        cfg["model_name"],
        cfg=cfg,
    ).to(train_device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    loader = DataLoader(eval_dataset, batch_size=cfg["batch_size"], shuffle=False)
    preds = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            text_x = batch[0].to(train_device)
            audio_x = batch[1].to(train_device)
            text_x, audio_x = apply_modality_mask(text_x, audio_x, cfg.get("modality", "both"))
            y = batch[2].to(train_device)
            outputs = model(text_x, audio_x, return_all=True)
            preds.extend(torch.argmax(outputs["logits"], dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    wa, ua, wf1, uf1 = compute_metrics(labels, preds)
    metrics = {
        "WA": wa,
        "UA": ua,
        "WF1": wf1,
        "UF1": uf1,
    }

    return {
        "client_id": client_id,
        "metrics": metrics,
        "num_samples": len(split_samples),
        "split": split,
    }
