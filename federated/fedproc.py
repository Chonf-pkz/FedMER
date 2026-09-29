import torch
import torch.nn.functional as F


def extract_representation(outputs):
    """Return the representation used by FedProc from a model output dict."""
    if not isinstance(outputs, dict):
        raise TypeError("FedProc requires model(..., return_all=True) to return a dict")

    representation = outputs.get("fusion")
    if representation is None:
        text = outputs.get("text_pool", outputs.get("text_proj"))
        audio = outputs.get("audio_pool", outputs.get("audio_proj"))
        if text is None or audio is None:
            raise KeyError(
                "FedProc requires a 'fusion' representation or both text/audio pooled representations"
            )
        text = text.reshape(text.size(0), -1)
        audio = audio.reshape(audio.size(0), -1)
        representation = torch.cat([text, audio], dim=1)

    if representation.ndim < 2:
        raise ValueError("FedProc representation must include a batch dimension")
    if representation.ndim > 2:
        representation = representation.reshape(representation.size(0), -1)
    return representation


def prototype_contrastive_loss(
    representations,
    labels,
    global_prototypes,
    temperature=0.5,
    sample_weights=None,
):
    """Contrast local representations against the available global class prototypes.

    Each sample's own class prototype is the positive and every other available
    class prototype is a negative. Samples whose class has no global prototype
    are ignored until that class has been observed by the server.
    """
    if not global_prototypes or len(global_prototypes) < 2:
        return representations.sum() * 0.0

    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("fedproc_temperature must be greater than zero")

    class_ids = sorted(int(class_id) for class_id in global_prototypes)
    prototypes = torch.stack(
        [
            global_prototypes[class_id]
            .detach()
            .to(device=representations.device, dtype=representations.dtype)
            .reshape(-1)
            for class_id in class_ids
        ],
        dim=0,
    )
    representations = representations.reshape(representations.size(0), -1)
    if representations.size(1) != prototypes.size(1):
        raise ValueError(
            "FedProc representation/prototype dimension mismatch: "
            f"{representations.size(1)} != {prototypes.size(1)}"
        )

    targets = torch.full_like(labels, -1, dtype=torch.long)
    for prototype_index, class_id in enumerate(class_ids):
        targets[labels == class_id] = prototype_index
    valid = targets >= 0
    if not torch.any(valid):
        return representations.sum() * 0.0

    normalized_features = F.normalize(representations[valid], p=2, dim=1, eps=1e-12)
    normalized_prototypes = F.normalize(prototypes, p=2, dim=1, eps=1e-12)
    logits = normalized_features.matmul(normalized_prototypes.transpose(0, 1)) / temperature
    losses = F.cross_entropy(logits, targets[valid], reduction="none")

    if sample_weights is None:
        return losses.mean()
    weights = sample_weights[valid].to(device=losses.device, dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


@torch.no_grad()
def compute_class_prototypes(
    model,
    dataset,
    batch_size,
    device,
    modality,
    modality_mask_fn,
):
    """Compute normalized per-class representation means for one client."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_sums = {}
    class_counts = {}

    model = model.to(device)
    model.eval()
    for batch in loader:
        text_x = batch[0].to(device)
        audio_x = batch[1].to(device)
        text_x, audio_x = modality_mask_fn(text_x, audio_x, modality)
        labels = batch[2].to(device)
        outputs = model(text_x, audio_x, return_all=True)
        representations = F.normalize(
            extract_representation(outputs).reshape(labels.size(0), -1),
            p=2,
            dim=1,
            eps=1e-12,
        )

        for class_id_tensor in torch.unique(labels):
            class_id = int(class_id_tensor.item())
            selected = representations[labels == class_id_tensor]
            selected_sum = selected.sum(dim=0).detach().cpu()
            selected_count = int(selected.size(0))
            if class_id in class_sums:
                class_sums[class_id] += selected_sum
                class_counts[class_id] += selected_count
            else:
                class_sums[class_id] = selected_sum
                class_counts[class_id] = selected_count

    prototypes = {
        class_id: class_sums[class_id] / float(class_counts[class_id])
        for class_id in class_sums
        if class_counts[class_id] > 0
    }
    return prototypes, class_counts
