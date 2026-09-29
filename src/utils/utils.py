
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import os
import wandb
from scipy.special import softmax
from torch.utils.data import DataLoader
import time
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def apply_word_dropout(tensor, dropout_prob):
    """
    Drop entire token representations by zeroing them out.
    Falls back to feature dropout when sequence length is 1.
    """
    if tensor is None:
        return None
    dropout_prob = float(max(0.0, min(1.0, dropout_prob)))
    if dropout_prob <= 0.0:
        return tensor.clone()

    if tensor.dim() >= 3:
        # Treat dimension-1 as token axis.
        mask_shape = tensor.shape[:-1]
        dropout_mask = torch.rand(mask_shape, device=tensor.device) < dropout_prob
        if dropout_mask.any():
            return tensor.clone().masked_fill(dropout_mask.unsqueeze(-1), 0.0)
        return tensor.clone()

    # For 2D tensors fall back to standard feature dropout.
    return F.dropout(tensor.clone(), p=dropout_prob, training=True)


def add_gaussian_noise(tensor, std):
    if tensor is None:
        return None
    std = float(max(0.0, std))
    if std <= 0.0:
        return tensor.clone()
    noise = torch.randn_like(tensor) * std
    return tensor.clone() + noise


def augment_modalities(text_tensor, audio_tensor, cfg):
    """
    Apply word dropout and additive Gaussian noise according to cfg.
    """
    cfg = cfg or {}
    word_dropout = cfg.get("word_dropout", 0.0)
    audio_noise_std = cfg.get("audio_noise_std", 0.0)

    text_aug = apply_word_dropout(text_tensor, word_dropout)
    audio_aug = add_gaussian_noise(audio_tensor, audio_noise_std)

    return text_aug, audio_aug


def normalize_modality(modality):
    if modality is None:
        return "both"
    value = str(modality).strip().lower()
    if value not in {"both", "text", "audio"}:
        raise ValueError(f"Unsupported modality '{modality}'. Expected one of: both, text, audio.")
    return value


def apply_modality_mask(text_tensor, audio_tensor, modality):
    mode = normalize_modality(modality)
    if mode == "text":
        audio_tensor = torch.zeros_like(audio_tensor)
    elif mode == "audio":
        text_tensor = torch.zeros_like(text_tensor)
    return text_tensor, audio_tensor


def compute_metrics(y_true, y_pred):
    wa = accuracy_score(y_true, y_pred)
    ua = balanced_accuracy_score(y_true, y_pred)
    wf1 = f1_score(y_true, y_pred, average="weighted")
    uf1 = f1_score(y_true, y_pred, average="macro")
    return wa, ua, wf1, uf1

def plot_and_save_roc(labels, probs, num_classes, save_path):
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))
    plt.figure(figsize=(8, 6))

    if num_classes == 4:
        class_names = ["Angry", "Happy", "Sad", "Neutral"]
    elif num_classes == 5:
        class_names = ["Angry", "Happy", "Sad", "Neutral", "Surprise"]
    elif num_classes == 7:
        class_names = ["Neutral", "Joy", "Anger", "Sadness", "Disgust", "Fear", "Surprise"]
    else:
        class_names = [f"Class {i}" for i in range(num_classes)]

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve per Class")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to {save_path}")

def train_and_evaluate(model, train_dataset, valid_dataset, test_dataset,
                       optimizer, scheduler, criterion_fn,
                       epochs=100, save_path="best_model.pt", seed=None,
                       batch_size=64, log_context=None,
                       modality="both", use_wandb=False):

    num_classes = getattr(model, "num_classes", None)
    if num_classes is None:
        all_labels = torch.cat([train_dataset.tensors[2],
                                valid_dataset.tensors[2],
                                test_dataset.tensors[2]])
        num_classes = len(torch.unique(all_labels))

    log_context = dict(log_context) if log_context is not None else {}
    stage = log_context.get("stage", "train")
    seed_value = log_context.get("seed", seed)
    log_context["seed"] = seed_value
    modality = normalize_modality(log_context.get("modality", modality))
    log_context["modality"] = modality
    use_wandb = bool(log_context.get("use_wandb", use_wandb))
    log_context["use_wandb"] = use_wandb

    def _format_float(value):
        return f"{float(value):.3f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)

    base_run_name = log_context.get("experiment_name")
    if not base_run_name:
        base_run_name = os.path.basename(save_path).replace(".pt", "") if save_path else "train"
    dataset_name = log_context.get("dataset")
    name_parts = [base_run_name]
    if dataset_name:
        name_parts.append(dataset_name)
    if stage:
        name_parts.append(stage)
    if seed_value is not None:
        name_parts.append(f"seed{seed_value}")
    run_display_name = " | ".join(name_parts)

    base_config = {
        "epochs": epochs,
        "lr": optimizer.param_groups[0]['lr'],
        "weight_decay": optimizer.param_groups[0]['weight_decay'],
        "scheduler": "ReduceLROnPlateau",
        "seed": seed_value,
        "batch_size": batch_size,
        "num_classes": num_classes
    }

    meta_keys = {
        "project", "group", "entity", "tags", "stage",
        "experiment_name", "run_name", "summary", "job_type"
    }
    for key, value in log_context.items():
        if key in meta_keys:
            continue
        if value is None:
            continue
        base_config[key] = value

    tags = log_context.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    auto_tags = []
    if dataset_name:
        auto_tags.append(dataset_name)
    config_num_classes = log_context.get("num_classes") or num_classes
    if config_num_classes:
        auto_tags.append(f"{int(config_num_classes)}class")
    if stage:
        auto_tags.append(stage)
    tags = list(dict.fromkeys(auto_tags + list(tags)))

    init_kwargs = {
        "project": log_context.get("project", "FedalMER-EmotionRecognition-{dataset_name}"),
        "name": run_display_name,
        "group": log_context.get("group"),
        "job_type": stage,
        "tags": tags,
        "config": base_config,
        "reinit": True
    }
    entity = log_context.get("entity")
    if entity:
        init_kwargs["entity"] = entity

    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(**init_kwargs)
        wandb.define_metric("epoch")
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_WA", summary="max")
        wandb.define_metric("val_UA", summary="max")
        wandb.define_metric("val_WF1", summary="max")
        wandb.define_metric("val_UF1", summary="max")

    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_val_ua = 0
    best_val_loss = float("inf")
    best_state_dict = None

    try:
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            model.train()
            total_train_loss = 0.0
            total_train_weight = 0.0
            total_train_classification = 0.0

            for batch in train_loader:
                text_x = batch[0].to(device)
                audio_x = batch[1].to(device)
                text_x, audio_x = apply_modality_mask(text_x, audio_x, modality)
                y = batch[2].to(device)
                confidences = batch[3].to(device) if len(batch) > 3 else None
                batch_size_value = float(y.size(0))
                batch_weight = confidences.sum().item() if confidences is not None else batch_size_value

                optimizer.zero_grad()
                outputs = model(text_x, audio_x, return_all=True)
                classification_loss = criterion_fn(outputs, y, confidences)

                loss = classification_loss
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * batch_size_value
                total_train_classification += classification_loss.item() * batch_size_value
                total_train_weight += batch_size_value

            avg_train_loss = total_train_loss / max(total_train_weight, 1e-8)
            avg_train_cls = total_train_classification / max(total_train_weight, 1e-8)
            model.eval()
            val_losses, val_preds, val_labels = [], [], []
            total_val_weight = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    text_x = batch[0].to(device)
                    audio_x = batch[1].to(device)
                    text_x, audio_x = apply_modality_mask(text_x, audio_x, modality)
                    y = batch[2].to(device)
                    confidences = batch[3].to(device) if len(batch) > 3 else None

                    val_outputs = model(text_x, audio_x, return_all=True)
                    val_loss = criterion_fn(val_outputs, y, confidences)
                    batch_weight = confidences.sum().item() if confidences is not None else float(y.size(0))
                    val_losses.append(val_loss.item() * batch_weight)
                    total_val_weight += batch_weight
                    val_preds.extend(torch.argmax(val_outputs["logits"], dim=1).cpu().numpy())
                    val_labels.extend(y.cpu().numpy())

            avg_val_loss = np.sum(val_losses) / max(total_val_weight, 1e-8)
            wa, ua, wf1, uf1 = compute_metrics(val_labels, val_preds)

            scheduler.step(avg_val_loss)

            log_payload = {
                "train_loss": avg_train_loss,
                "train_classification_loss": avg_train_cls,
                "val_loss": avg_val_loss,
                "val_WA": wa,
                "val_UA": ua,
                "val_WF1": wf1,
                "val_UF1": uf1,
                "epoch": epoch + 1,
                "stage": stage,
                "seed": seed_value
            }

            if use_wandb:
                wandb.log(log_payload)

            wa_for_save, ua_for_save = wa, ua

            if ua_for_save > best_val_ua or avg_val_loss < best_val_loss:
                best_val_ua = ua_for_save
                best_val_loss = avg_val_loss
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                if save_path:
                    save_dir = os.path.dirname(save_path)
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                    torch.save(best_state_dict, save_path)
                print(
                    f"\nSaved best model at epoch {epoch + 1} with "
                    f"WA = {wa_for_save:.4f}, UA = {ua_for_save:.4f}, Val Loss = {avg_val_loss:.4f}"
                )
                if use_wandb and wandb_run is not None:
                    wandb_run.summary["best_val_ua"] = best_val_ua
                    wandb_run.summary["best_val_ua"] = ua_for_save
                    wandb_run.summary["best_val_loss"] = best_val_loss

            print(
                f"[Epoch {epoch + 1}/{epochs}] "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"WA: {wa:.4f}, UA: {ua:.4f}, WF1: {wf1:.4f}, UF1: {uf1:.4f}"
            )

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        elif save_path and os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))

        model.eval()
        test_preds, test_labels, test_logits_list = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                text_x = batch[0].to(device)
                audio_x = batch[1].to(device)
                text_x, audio_x = apply_modality_mask(text_x, audio_x, modality)
                y = batch[2].to(device)

                test_outputs = model(text_x, audio_x, return_all=True)
                test_logits = test_outputs["logits"].cpu().numpy()
                test_preds.extend(np.argmax(test_logits, axis=1))
                test_labels.extend(y.cpu().numpy())
                test_logits_list.append(test_logits)

        test_logits_all = np.vstack(test_logits_list)
        test_probs = softmax(test_logits_all, axis=1)

        test_wa, test_ua, test_wf1, test_uf1 = compute_metrics(test_labels, test_preds)

        if use_wandb:
            wandb.log({
                "test_WA": test_wa,
                "test_UA": test_ua,
                "test_WF1": test_wf1,
                "test_UF1": test_uf1,
                "stage": stage,
                "seed": seed_value
            })

        summary_update = {
            "final_test_WA": test_wa,
            "final_test_UA": test_ua,
            "final_test_WF1": test_wf1,
            "final_test_UF1": test_uf1,
            "stage": stage,
            "seed": seed_value
        }
        for key in ("labeled_samples", "combined_train_samples"):
            value = log_context.get(key)
            if value is not None:
                summary_update[key] = value

        if use_wandb and wandb_run is not None:
            wandb_run.summary.update(summary_update)

        print("\nFinal Test Metrics:")
        print(f"WA: {test_wa:.4f}, UA: {test_ua:.4f}, WF1: {test_wf1:.4f}, UF1: {test_uf1:.4f}")

        save_roc = bool(log_context.get("save_roc")) if log_context is not None else False
        if save_roc and save_path:
            roc_save_path = save_path.replace(".pt", "_roc_curve.png")
            plot_and_save_roc(np.array(test_labels), test_probs, num_classes, roc_save_path)

        final_metrics = {
            "test_WA": test_wa,
            "test_UA": test_ua,
            "test_WF1": test_wf1,
            "test_UF1": test_uf1
        }
        return final_metrics
    finally:
        if use_wandb:
            wandb.finish()

def get_model_stats(model, sample_input, device="cuda"):
    model.eval().to(device)

    flops = FlopCountAnalysis(model, sample_input).total()
    params = sum(p.numel() for p in model.parameters())

    n_runs = 50
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(*sample_input) if isinstance(sample_input, tuple) else model(sample_input)
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / n_runs

    return {
        "FLOPs": flops,
        "Parameters": params,
        "Inference time (s)": avg_time
    }
