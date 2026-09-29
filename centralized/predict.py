import csv
import numbers
import os
import sys
import time
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.special import softmax
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from thop import profile
from torch_geometric.data import Data

import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.utils import set_seed
from src.architecture.FedalMER import FedalMER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_label(label_value, label_map):
    if isinstance(label_value, torch.Tensor):
        label_value = label_value.item()

    if isinstance(label_value, numbers.Integral):
        return int(label_value)

    if isinstance(label_value, str):
        normalized = label_value.strip().lower()
        if label_map is not None:
            if normalized in label_map:
                return label_map[normalized]
            # Some datasets might use numeric strings; attempt conversion before failing
            if normalized.isdigit():
                return int(normalized)
            raise ValueError(f"Label '{label_value}' not found in the provided label map.")
        else:
            return normalized  # Will be mapped later when label_map is built

    raise TypeError(f"Unsupported label type: {type(label_value)}")


def load_data(pkl_path, label_map=None):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data[0], tuple):
        raise ValueError(f"Loaded data is raw. Please provide feature-extracted .pkl: {pkl_path}")
    text = torch.stack([torch.tensor(item['text_embed']) for item in data])
    audio = torch.stack([torch.tensor(item['audio_embed']) for item in data])
    standard_map = {}
    labels = []
    for item in data:
        normalized = _normalize_label(item.get('label'), label_map)
        if isinstance(normalized, str):
            if normalized not in standard_map:
                standard_map[normalized] = len(standard_map)
            labels.append(standard_map[normalized])
        else:
            labels.append(normalized)
    labels = torch.tensor(labels, dtype=torch.long)
    return Data(text_x=text, audio_x=audio, y=labels)


def compute_metrics(y_true, y_pred):
    wa = accuracy_score(y_true, y_pred)
    ua = balanced_accuracy_score(y_true, y_pred)
    wf1 = f1_score(y_true, y_pred, average="weighted")
    uf1 = f1_score(y_true, y_pred, average="macro")
    return wa, ua, wf1, uf1


def plot_confusion_matrix(y_true, y_pred, label_names, save_path, fontsize=14):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_percent, annot=True, fmt=".2f", cmap="Blues", cbar=True,
        xticklabels=label_names, yticklabels=label_names,
        annot_kws={"size": 14, "weight": "bold"} 
    )
    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=14, fontweight="bold")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", dpi=500, format="pdf")
    plt.close()
    print(f"Saved confusion matrix to {save_path}.pdf")



def plot_tsne(features, labels, label_names, save_path):
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(7, 6))
    for i, label_name in enumerate(label_names):
        idx = labels == i
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label_name, s=10, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    plt.close()
    print(f"Saved t-SNE plot to {save_path}.pdf")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict CMCL model on IEMOCAP or ESD")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing test.pkl")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["IEMOCAP", "ESD"],
        help="Dataset",
    )
    parser.add_argument("--num_classes", type=int, required=True, help="Number of emotion classes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="logs", help="Directory to save outputs")
    parser.add_argument("--modality", type=str, choices=["both", "text", "audio"], default="both", help="Modality to predict with")
    parser.add_argument(
        "--noise_dir",
        type=str,
        default=None,
        help="Optional directory containing noisy test feature PKLs to evaluate in a single run.",
    )
    parser.add_argument(
        "--noise_pattern",
        type=str,
        default="_test_",
        help="Substring used to filter noisy PKL filenames inside --noise_dir.",
    )
    return parser.parse_args()


def get_label_mapping(dataset, label_names):
    base = {name.strip().lower(): idx for idx, name in enumerate(label_names)}
    # Also keep original cased keys for direct lookups.
    for idx, name in enumerate(label_names):
        base[name.strip()] = idx

    if dataset == "IEMOCAP":
        # Map original emotion abbreviations to the 4-class scheme
        angry_idx = base.get("angry")
        happy_idx = base.get("happy")
        sad_idx = base.get("sad")
        neutral_idx = base.get("neutral")
        alias_map = {
            "ang": angry_idx,
            "angry": angry_idx,
            "hap": happy_idx,
            "happy": happy_idx,
            "exc": happy_idx,
            "sad": sad_idx,
            "neu": neutral_idx,
            "neutral": neutral_idx,
        }
        alias_map = {k: v for k, v in alias_map.items() if v is not None}
        base.update(alias_map)
    elif dataset == "ESD":
        # Ensure lowercase labels map correctly
        lowercase_map = {k.lower(): v for k, v in base.items()}
        base.update(lowercase_map)

    return base


def get_filenames(data_dir, dataset, num_classes, split="test"):
    if dataset == "IEMOCAP":
        assert num_classes == 4, "IEMOCAP supports only 4 classes."
        prefix = "IEMOCAP_BERT_Wav2Vec2"
    elif dataset == "ESD":
        assert num_classes == 5, "ESD uses 5 classes."
        prefix = "ESD_BERT_Wav2Vec2" 
    else:
        raise ValueError("Dataset must be either 'IEMOCAP' or 'ESD'.")
    return os.path.join(data_dir, f"{prefix}_{split}.pkl")


def collect_noise_files(noise_dir, include_pattern):
    if noise_dir is None:
        return []
    directory = Path(noise_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"--noise_dir '{noise_dir}' does not exist or is not a directory.")
    files = sorted(p for p in directory.glob("*.pkl") if include_pattern in p.name)
    if not files:
        print(f"[WARN] No noisy test PKL files found in {noise_dir} matching pattern '{include_pattern}'.")
    return files


def evaluate_dataset(model, data, modality):
    model_inputs = {
        "text": data.text_x,
        "audio": data.audio_x,
    }

    with torch.no_grad():
        if modality == "text":
            model_inputs["audio"] = torch.zeros_like(model_inputs["audio"])
        elif modality == "audio":
            model_inputs["text"] = torch.zeros_like(model_inputs["text"])

        outputs = model(
            model_inputs["text"],
            model_inputs["audio"],
            return_all=True,
        )

    logits = outputs["logits"].cpu().numpy()
    probs = softmax(logits, axis=1)
    preds = np.argmax(probs, axis=1)
    labels = data.y.cpu().numpy()

    wa, ua, wf1, uf1 = compute_metrics(labels, preds)
    metrics = {"WA": wa, "UA": ua, "WF1": wf1, "UF1": uf1}
    return metrics, preds, probs, labels


def save_metrics_summary(rows, path):
    if not rows:
        return
    fieldnames = ["split", "WA", "UA", "WF1", "UF1"]
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Saved metrics summary to {path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.dataset == "IEMOCAP":
        label_names = ["Angry", "Happy", "Sad", "Neutral"]
    else:
        label_names = ["Angry", "Happy", "Sad", "Neutral", "Surprise"]

    print("\nLoading test data...")
    test_pkl = get_filenames(args.data_dir, args.dataset, args.num_classes, split="test")
    label_map = get_label_mapping(args.dataset, label_names)
    test_data = load_data(test_pkl, label_map=label_map).to(device)

    feature_dims = {
        "text": test_data.text_x.size(-1),
        "audio": test_data.audio_x.size(-1),
    }

    evaluation_targets = [("clean_test", Path(test_pkl), test_data)]

    noise_files = collect_noise_files(args.noise_dir, args.noise_pattern)
    for noise_path in noise_files:
        noise_data = load_data(noise_path, label_map=label_map).to(device)
        evaluation_targets.append((noise_path.stem, noise_path, noise_data))

    print("Loading model...")
    model = FedalMER(
        text_input_dim=feature_dims["text"],
        audio_input_dim=feature_dims["audio"],
        fusion_dim=512,
        projection_dim=256,
        num_heads=4,
        dropout=0.5,
        linear_layer_dims=[512, 256],
        num_classes=args.num_classes
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.model_path))[0]
    suffix = f"_{args.modality}only" if args.modality != "both" else ""
    cm_path = os.path.join(args.save_dir, f"{base_name}_confusion_matrix{suffix}.png")
    tsne_path = os.path.join(args.save_dir, f"{base_name}_tsne{suffix}.pdf")

    print("\nCalculating model parameters and FLOPs...")
    dummy_text = torch.randn(1, feature_dims["text"], device=device)
    dummy_audio = torch.randn(1, feature_dims["audio"], device=device)
    flops, params = profile(model, inputs=(dummy_text, dummy_audio), verbose=False)
    print(f"Model Parameters: {params/1e6:.3f} M")
    print(f"Model FLOPs: {flops/1e9:.3f} GFLOPs")

    repeats = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy_text, dummy_audio)
    end_time = time.time()
    avg_time = (end_time - start_time) / repeats * 1000
    print(f"Average inference time per sample: {avg_time:.2f} ms")

    results_rows = []

    for idx, (split_name, path_obj, data_obj) in enumerate(evaluation_targets):
        tag_suffix = "" if split_name == "clean_test" else f"_{split_name}"
        print(f"\nRunning prediction on '{path_obj.name}' (modality = {args.modality}) ...")
        metrics, preds, probs, labels = evaluate_dataset(model, data_obj, args.modality)

        print(
            f"Results [{split_name}]: "
            f"WA={metrics['WA']:.4f}, UA={metrics['UA']:.4f}, "
            f"WF1={metrics['WF1']:.4f}, UF1={metrics['UF1']:.4f}"
        )
        results_rows.append({"split": split_name, **metrics})

        report = classification_report(labels, preds, target_names=label_names, digits=4)
        split_report_path = os.path.join(args.save_dir, f"{base_name}{tag_suffix}_classification_report{suffix}.txt")
        with open(split_report_path, "w") as f:
            f.write(report)
        print(f"Saved classification report to {split_report_path}")

        # if idx == 0:
        #     plot_confusion_matrix(labels, preds, label_names, cm_path)
        #     plot_tsne(probs, labels, label_names, tsne_path)

    print("\nSummary metrics across evaluated splits:")
    for row in results_rows:
        print(
            f"  {row['split']}: "
            f"WA={row['WA']:.4f}, UA={row['UA']:.4f}, "
            f"WF1={row['WF1']:.4f}, UF1={row['UF1']:.4f}"
        )

    summary_path = os.path.join(args.save_dir, f"{base_name}_metrics_summary{suffix}.csv")
    save_metrics_summary(results_rows, summary_path)

    print("\nPrediction and evaluation complete.")


if __name__ == "__main__":
    main()
