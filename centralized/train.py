import argparse
import csv
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_factory import (  # noqa: E402
    MODEL_CHOICES,
    build_model,
    get_model_display_name,
    normalize_model_name,
)
from src.utils.utils import set_seed, train_and_evaluate  # noqa: E402


def _ensure_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return torch.tensor(x, dtype=torch.float32)


def build_tensor_dataset(data, label_map=None):
    if label_map is None:
        label_map = {}

    next_index = len(label_map)
    text_tensors, audio_tensors = [], []
    label_tensors, confidence_tensors = [], []

    for item in data:
        text_tensors.append(_ensure_tensor(item["text_embed"]))
        audio_tensors.append(_ensure_tensor(item["audio_embed"]))

        label_value = item.get("label", -1)
        if isinstance(label_value, str):
            normalized_label = label_value.strip().lower()
            if normalized_label not in label_map:
                label_map[normalized_label] = next_index
                next_index += 1
            label_idx = label_map[normalized_label]
        else:
            label_idx = int(label_value)

        label_tensors.append(torch.tensor(label_idx, dtype=torch.long))
        confidence_value = item.get("confidence", 1.0)
        if confidence_value is None:
            confidence_value = 1.0
        confidence_tensors.append(torch.tensor(float(confidence_value), dtype=torch.float32))

    dataset = TensorDataset(
        torch.stack(text_tensors),
        torch.stack(audio_tensors),
        torch.stack(label_tensors),
        torch.stack(confidence_tensors),
    )
    return dataset, label_map


def clone_samples(samples, confidence=1.0):
    cloned = []
    for item in samples:
        new_item = dict(item)
        new_item["confidence"] = confidence
        cloned.append(new_item)
    return cloned


def combined_loss(outputs, labels, ce_loss, confidences=None):
    logits = outputs["logits"]
    losses = ce_loss(logits, labels)
    if confidences is not None:
        weights = confidences.float()
        return (losses * weights).sum() / (weights.sum() + 1e-8)
    return losses


def append_csv(path, fieldnames, row):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_filenames(data_dir, dataset, num_classes):
    if dataset == "IEMOCAP":
        assert num_classes == 4, "IEMOCAP now only supports 4 classes."
        prefix = "IEMOCAP_BERT_Wav2Vec2"
    elif dataset == "MSP-IMPROV":
        assert num_classes == 4, "MSP-IMPROV uses 4 classes."
        prefix = "MSPIMPROV_BERT_Wav2Vec2"
    elif dataset == "ESD":
        assert num_classes == 5, "ESD uses 5 classes."
        prefix = "ESD_BERT_Wav2Vec2"
    elif dataset == "MELD":
        assert num_classes == 7, "MELD uses 7 classes."
        prefix = "MELD_BERT_Wav2Vec2"
    else:
        raise ValueError("Dataset must be one of 'IEMOCAP', 'MSP-IMPROV', 'ESD', or 'MELD'.")

    return {
        "train": os.path.join(data_dir, f"{prefix}_train.pkl"),
        "val": os.path.join(data_dir, f"{prefix}_val.pkl"),
        "test": os.path.join(data_dir, f"{prefix}_test.pkl"),
    }


def _class_weights(train_dataset, num_classes, train_device):
    train_labels = train_dataset.tensors[2]
    class_counts = torch.bincount(train_labels, minlength=num_classes).float()
    class_counts[class_counts == 0] = 1.0
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes
    return weights.to(train_device)


def parse_args():
    parser = argparse.ArgumentParser(description="Train multimodal SER models with supervised learning only.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["IEMOCAP", "MSP-IMPROV", "ESD", "MELD"])
    parser.add_argument("--num_classes", type=int, required=True, choices=[4, 5, 7])
    parser.add_argument(
        "--model_name",
        type=str,
        default="fedalmer",
        help=f"Model name. Supported: {', '.join(MODEL_CHOICES)}",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--modality", type=str, default="both", choices=["both", "text", "audio"])
    parser.add_argument("--results_suffix", type=str, default=None)
    parser.add_argument("--logs_root", type=str, default="logs/centralized")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--reset_logs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = normalize_model_name(args.model_name)
    model_display_name = get_model_display_name(model_name)
    seeds = [42, 52, 103]

    result_file_name = args.results_suffix or f"{args.dataset}_{args.num_classes}class_{model_display_name}_5seeds"
    exp_name = args.exp_name or result_file_name
    log_dir = os.path.join(args.logs_root, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    seed_log_path = os.path.join(log_dir, "seed_metrics.csv")
    summary_log_path = os.path.join(log_dir, "summary_metrics.csv")
    if args.reset_logs:
        for path in (seed_log_path, summary_log_path):
            if os.path.exists(path):
                os.remove(path)

    filenames = get_filenames(args.data_dir, args.dataset, args.num_classes)
    with open(filenames["train"], "rb") as f:
        train_raw_full = pickle.load(f)
    with open(filenames["val"], "rb") as f:
        val_raw_full = pickle.load(f)
    with open(filenames["test"], "rb") as f:
        test_raw_full = pickle.load(f)

    wa_list, ua_list, wf1_list, uf1_list = [], [], [], []
    for seed in seeds:
        set_seed(seed)
        print(f"\n=== Running with seed {seed} ===")

        label_map = {}
        train_dataset, label_map = build_tensor_dataset(clone_samples(train_raw_full), label_map=label_map)
        val_dataset, label_map = build_tensor_dataset(clone_samples(val_raw_full), label_map=label_map)
        test_dataset, label_map = build_tensor_dataset(clone_samples(test_raw_full), label_map=label_map)

        train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(model_name=model_name, num_classes=args.num_classes).to(train_device)
        ce_loss_fn = torch.nn.CrossEntropyLoss(
            weight=_class_weights(train_dataset, args.num_classes, train_device),
            reduction="none",
        )

        def loss_fn(out, y, conf):
            return combined_loss(out, y, ce_loss_fn, confidences=conf)

        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
        save_path = f"saved_model/{args.dataset}_{args.num_classes}class_{model_display_name}_seed{seed}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        metrics = train_and_evaluate(
            model,
            train_dataset,
            val_dataset,
            test_dataset,
            optimizer,
            scheduler,
            loss_fn,
            epochs=args.epochs,
            save_path=save_path,
            seed=seed,
            batch_size=args.batch_size,
            modality=args.modality,
            log_context={
                "dataset": args.dataset,
                "num_classes": args.num_classes,
                "model_name": model_name,
                "model_display_name": model_display_name,
                "modality": args.modality,
                "batch_size": args.batch_size,
                "group": f"{args.dataset}_{args.num_classes}class_{model_name}",
                "project": f"{model_display_name}-EmotionRecognition-{args.dataset}",
                "experiment_name": exp_name,
                "seed": seed,
                "stage": "supervised",
                "tags": [args.dataset, f"{args.num_classes}class", model_name, "supervised"],
                "labeled_samples": len(train_raw_full),
            },
        )

        print(
            f"Seed {seed} - Final WA: {metrics['test_WA']:.4f}, UA: {metrics['test_UA']:.4f}, "
            f"WF1: {metrics['test_WF1']:.4f}, UF1: {metrics['test_UF1']:.4f}"
        )

        wa_list.append(metrics["test_WA"])
        ua_list.append(metrics["test_UA"])
        wf1_list.append(metrics["test_WF1"])
        uf1_list.append(metrics["test_UF1"])

        append_csv(seed_log_path, [
            "seed", "stage", "test_WA", "test_UA", "test_WF1", "test_UF1",
        ], {
            "seed": seed,
            "stage": "supervised",
            "test_WA": metrics["test_WA"],
            "test_UA": metrics["test_UA"],
            "test_WF1": metrics["test_WF1"],
            "test_UF1": metrics["test_UF1"],
        })

    print("\n=== Average Results over 5 seeds ===")
    print(f"Avg WA:  {np.mean(wa_list):.4f}, {np.std(wa_list, ddof=1):.4f}")
    print(f"Avg UA:  {np.mean(ua_list):.4f}, {np.std(ua_list, ddof=1):.4f}")
    print(f"Avg WF1: {np.mean(wf1_list):.4f}, {np.std(wf1_list, ddof=1):.4f}")
    print(f"Avg UF1: {np.mean(uf1_list):.4f}, {np.std(uf1_list, ddof=1):.4f}")

    results_df = pd.DataFrame({
        "Metric": ["WA", "UA", "WF1", "UF1"],
        "Mean": [np.mean(wa_list), np.mean(ua_list), np.mean(wf1_list), np.mean(uf1_list)],
        "Std": [
            np.std(wa_list, ddof=1),
            np.std(ua_list, ddof=1),
            np.std(wf1_list, ddof=1),
            np.std(uf1_list, ddof=1),
        ],
    })

    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", f"{result_file_name}.csv")
    results_df.to_csv(results_path, index=False)
    results_df.to_csv(summary_log_path, index=False)
    print(f"Results saved to {results_path}")
    print(f"Centralized logs saved to {log_dir}")


if __name__ == "__main__":
    main()
