import csv
import os
import pickle
import random
from typing import List

import numpy as np
import torch


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def client_checkpoint_path(tmp_dir, client_id, round_idx):
    """Keep temporary checkpoint paths below the Windows MAX_PATH boundary."""
    client_suffix = str(client_id).rsplit("_", 1)[-1]
    return os.path.join(tmp_dir, f"c{client_suffix}_r{round_idx}.pt")


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    return data


def append_csv(path, fieldnames, row):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def list_clients(clients_root, num_clients=None) -> List[str]:
    if not os.path.isdir(clients_root):
        raise FileNotFoundError(f"clients_root not found: {clients_root}")
    candidates = sorted(
        [d for d in os.listdir(clients_root) if d.startswith("client_") and os.path.isdir(os.path.join(clients_root, d))]
    )
    if num_clients is not None:
        candidates = candidates[: int(num_clients)]
    return candidates


def load_state_dict(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def save_global_checkpoint(path, state_dict, meta=None, extra=None):
    payload = {
        "state_dict": state_dict,
        "meta": meta or {},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
