import torch


def _weighted_average(state_dicts, weights, cast_back=True):
    if len(state_dicts) == 0:
        return {}

    if len(weights) != len(state_dicts):
        raise ValueError("len(weights) must equal len(state_dicts)")

    weights = [float(w) for w in weights]
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0] * len(weights)
        total_weight = float(len(weights))

    keys = state_dicts[0].keys()
    for sd in state_dicts[1:]:
        if sd.keys() != keys:
            raise ValueError("State dict keys mismatch across clients")

    averaged = {}
    for k in keys:
        t0 = state_dicts[0][k]

        if torch.is_floating_point(t0):
            acc = torch.zeros_like(t0.detach().cpu(), dtype=torch.float32)
            for sd, w in zip(state_dicts, weights):
                acc += sd[k].detach().cpu().to(torch.float32) * w
            out = acc / total_weight
            if cast_back:
                out = out.to(dtype=t0.dtype)
            averaged[k] = out
        else:
            averaged[k] = t0.detach().cpu()

    return averaged


@torch.no_grad()
def fedavg(state_dicts, weights, cast_back=True):
    return _weighted_average(state_dicts, weights, cast_back=cast_back)


@torch.no_grad()
def fedproc(
    state_dicts,
    weights,
    local_prototypes,
    prototype_counts,
    previous_global_prototypes=None,
    cast_back=True,
):
    """Aggregate FedProc model parameters and sample-weighted class prototypes."""
    if len(local_prototypes) != len(state_dicts):
        raise ValueError("len(local_prototypes) must equal len(state_dicts)")
    if len(prototype_counts) != len(state_dicts):
        raise ValueError("len(prototype_counts) must equal len(state_dicts)")

    averaged_state = _weighted_average(state_dicts, weights, cast_back=cast_back)
    global_prototypes = {
        int(class_id): prototype.detach().cpu().clone()
        for class_id, prototype in (previous_global_prototypes or {}).items()
    }

    observed_classes = set()
    for client_prototypes in local_prototypes:
        observed_classes.update(int(class_id) for class_id in client_prototypes)

    for class_id in sorted(observed_classes):
        weighted_sum = None
        total_count = 0
        expected_shape = None
        for client_prototypes, client_counts in zip(local_prototypes, prototype_counts):
            if class_id not in client_prototypes:
                continue
            count = int(client_counts.get(class_id, 0))
            if count <= 0:
                continue
            prototype = client_prototypes[class_id].detach().cpu().to(torch.float32)
            if expected_shape is None:
                expected_shape = prototype.shape
                weighted_sum = torch.zeros_like(prototype)
            elif prototype.shape != expected_shape:
                raise ValueError(
                    f"Prototype shape mismatch for class {class_id}: "
                    f"{prototype.shape} != {expected_shape}"
                )
            weighted_sum += prototype * count
            total_count += count

        if total_count > 0:
            global_prototypes[class_id] = weighted_sum / float(total_count)

    return averaged_state, global_prototypes


@torch.no_grad()
def feddc(
    global_state,
    state_dicts,
    weights,
    drift_state=None,
    correction_weight=1.0,
    drift_momentum=0.0,
    cast_back=True,
):
    """FedDC-style server aggregation with a persistent drift correction state."""
    if len(state_dicts) == 0:
        return {}, {}
    if len(weights) != len(state_dicts):
        raise ValueError("len(weights) must equal len(state_dicts)")

    weights = [float(w) for w in weights]
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0] * len(weights)
        total_weight = float(len(weights))
    ratios = [w / total_weight for w in weights]

    keys = global_state.keys()
    for sd in state_dicts:
        if sd.keys() != keys:
            raise ValueError("State dict keys mismatch across clients")

    correction_weight = float(correction_weight)
    drift_momentum = float(drift_momentum)
    drift_momentum = min(max(drift_momentum, 0.0), 1.0)

    aggregated = {}
    new_drift = {}
    for key in keys:
        global_tensor = global_state[key].detach().cpu()
        if torch.is_floating_point(global_tensor):
            global_float = global_tensor.to(torch.float32)
            avg_local = torch.zeros_like(global_float)
            avg_delta = torch.zeros_like(global_float)
            for sd, ratio in zip(state_dicts, ratios):
                local_float = sd[key].detach().cpu().to(torch.float32)
                avg_local += ratio * local_float
                avg_delta += ratio * (local_float - global_float)

            previous_drift = None
            if drift_state is not None and key in drift_state:
                previous_drift = drift_state[key].detach().cpu().to(torch.float32)
            if previous_drift is None:
                previous_drift = torch.zeros_like(global_float)

            drift = drift_momentum * previous_drift + (1.0 - drift_momentum) * avg_delta
            out = avg_local + correction_weight * drift
            if cast_back:
                out = out.to(dtype=global_tensor.dtype)
            aggregated[key] = out
            new_drift[key] = drift
        else:
            aggregated[key] = global_tensor
            if drift_state is not None and key in drift_state:
                new_drift[key] = drift_state[key].detach().cpu()

    return aggregated, new_drift


@torch.no_grad()
def fednova(global_state, state_dicts, weights, local_steps, cast_back=True):
    """Federated normalized averaging.

    This ports the server-side normalization rule from FedNova: each client
    update is divided by its local normalizing vector (tau_i for plain local
    optimization), then scaled by tau_eff = sum_i p_i * tau_i.
    """
    if len(state_dicts) == 0:
        return {}
    if len(weights) != len(state_dicts):
        raise ValueError("len(weights) must equal len(state_dicts)")
    if len(local_steps) != len(state_dicts):
        raise ValueError("len(local_steps) must equal len(state_dicts)")

    weights = [float(w) for w in weights]
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0] * len(weights)
        total_weight = float(len(weights))
    ratios = [w / total_weight for w in weights]
    local_steps = [max(float(step), 1.0) for step in local_steps]
    tau_eff = sum(ratio * step for ratio, step in zip(ratios, local_steps))

    keys = global_state.keys()
    for sd in state_dicts:
        if sd.keys() != keys:
            raise ValueError("State dict keys mismatch across clients")

    aggregated = {}
    for key in keys:
        global_tensor = global_state[key].detach().cpu()
        if torch.is_floating_point(global_tensor):
            update = torch.zeros_like(global_tensor, dtype=torch.float32)
            global_float = global_tensor.to(torch.float32)
            for sd, ratio, step in zip(state_dicts, ratios, local_steps):
                local_tensor = sd[key].detach().cpu().to(torch.float32)
                normalized_delta = (global_float - local_tensor) / step
                update += ratio * normalized_delta
            out = global_float - tau_eff * update
            if cast_back:
                out = out.to(dtype=global_tensor.dtype)
            aggregated[key] = out
        else:
            aggregated[key] = global_tensor

    return aggregated

def batchnorm_state_keys(model):
    """Return all BatchNorm parameters and buffers in a model state dict."""
    keys = set()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            prefix = f"{module_name}." if module_name else ""
            keys.update(prefix + name for name, _ in module.named_parameters(recurse=False))
            keys.update(prefix + name for name, _ in module.named_buffers(recurse=False))
    return keys


@torch.no_grad()
def fedbn(global_state, state_dicts, weights, bn_keys, cast_back=True):
    """Average non-BatchNorm tensors while preserving client-local BN state."""
    if len(state_dicts) == 0:
        return {}

    if len(weights) != len(state_dicts):
        raise ValueError("len(weights) must equal len(state_dicts)")

    weights = [float(w) for w in weights]
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0] * len(weights)
        total_weight = float(len(weights))

    keys = global_state.keys()
    for sd in state_dicts:
        if sd.keys() != keys:
            raise ValueError("State dict keys mismatch across clients")

    bn_keys = set(bn_keys)
    averaged = {}
    for k in keys:
        global_tensor = global_state[k].detach().cpu()

        if k in bn_keys or not torch.is_floating_point(global_tensor):
            averaged[k] = global_tensor.clone()
        else:
            acc = torch.zeros_like(global_tensor, dtype=torch.float32)
            for sd, w in zip(state_dicts, weights):
                acc += sd[k].detach().cpu().to(torch.float32) * w
            out = acc / total_weight
            if cast_back:
                out = out.to(dtype=global_tensor.dtype)
            averaged[k] = out

    return averaged
