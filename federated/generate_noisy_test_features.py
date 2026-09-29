import argparse
import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal


def _parse_int_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            text = str(item).strip()
            if not text:
                continue
            out.append(int(text))
        return out
    text = str(value).strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _list_clients(features_root, num_clients=None):
    root = Path(features_root)
    clients = sorted([p.name for p in root.glob("client_*") if p.is_dir()])
    if num_clients is not None and num_clients > 0:
        clients = clients[:num_clients]
    return clients


def _load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    return data


def _save_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _load_noise(noise_path, target_sr=16000):
    wav, sr = sf.read(noise_path, always_2d=False)
    if isinstance(wav, np.ndarray):
        x = wav.astype(np.float32)
    else:
        x = np.array(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        g = np.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        x = signal.resample_poly(x, up=up, down=down).astype(np.float32)
    x = np.ascontiguousarray(x)
    if x.size == 0:
        raise ValueError(f"Noise file has no samples: {noise_path}")
    return x


def _stable_int(text):
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _vector_from_noise(noise_wave, dim, key):
    if dim <= 0:
        return np.zeros((0,), dtype=np.float32)
    start = _stable_int(key) % max(len(noise_wave), 1)
    if len(noise_wave) >= dim:
        end = start + dim
        if end <= len(noise_wave):
            vec = noise_wave[start:end]
        else:
            first = noise_wave[start:]
            remain = dim - len(first)
            second = noise_wave[:remain]
            vec = np.concatenate([first, second], axis=0)
    else:
        reps = int(np.ceil(dim / len(noise_wave)))
        vec = np.tile(noise_wave, reps)[:dim]
    return vec.astype(np.float32)


def _add_noise_to_embedding(audio_embed, noise_wave, snr_db, sample_key):
    if isinstance(audio_embed, torch.Tensor):
        emb = audio_embed.detach().cpu().float()
        original_dtype = audio_embed.dtype
    else:
        emb = torch.tensor(np.asarray(audio_embed), dtype=torch.float32)
        original_dtype = torch.float32

    shape = emb.shape
    flat = emb.reshape(-1)
    dim = flat.numel()
    if dim == 0:
        return emb.to(dtype=original_dtype)

    noise_vec = _vector_from_noise(noise_wave, dim, sample_key)
    noise_t = torch.from_numpy(noise_vec)
    noise_power = torch.mean(noise_t.pow(2)).item()
    signal_power = torch.mean(flat.pow(2)).item()

    if noise_power <= 1e-12 or signal_power <= 1e-12:
        out = flat
    else:
        target_noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
        scale = float(np.sqrt(max(target_noise_power, 1e-12) / max(noise_power, 1e-12)))
        out = flat + scale * noise_t

    return out.reshape(shape).to(dtype=original_dtype)


def _link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _build_noisy_test(clean_test_samples, noise_wave, snr_db, noise_name):
    noisy = []
    for idx, item in enumerate(clean_test_samples):
        if not isinstance(item, dict):
            noisy.append(item)
            continue
        audio_embed = item.get("audio_embed")
        if audio_embed is None:
            noisy.append(item)
            continue
        sample_id = item.get("sample_id") or item.get("audio_path") or f"idx_{idx}"
        sample_key = f"{noise_name}|{snr_db}|{sample_id}|{idx}"
        new_item = dict(item)
        new_item["audio_embed"] = _add_noise_to_embedding(audio_embed, noise_wave, snr_db, sample_key)
        new_item["noise_name"] = noise_name
        new_item["snr_db"] = float(snr_db)
        noisy.append(new_item)
    return noisy


def parse_args():
    parser = argparse.ArgumentParser(description="Generate noisy test feature roots from NoiseX-92.")
    parser.add_argument("--clean_features_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--noise_dir", type=str, required=True)
    parser.add_argument("--snrs", type=str, default="0,5,10")
    parser.add_argument("--num_clients", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    clean_root = os.path.abspath(args.clean_features_root)
    out_root = os.path.abspath(args.out_root)
    noise_dir = os.path.abspath(args.noise_dir)
    snrs = _parse_int_list(args.snrs)
    if not snrs:
        snrs = [10]

    if not os.path.isdir(clean_root):
        raise FileNotFoundError(f"clean_features_root not found: {clean_root}")
    if not os.path.isdir(noise_dir):
        raise FileNotFoundError(f"noise_dir not found: {noise_dir}")

    noise_files = sorted(
        [p for p in Path(noise_dir).glob("*.wav") if p.is_file()]
    )
    if not noise_files:
        raise ValueError(f"No .wav files found in noise_dir: {noise_dir}")

    clients = _list_clients(clean_root, num_clients=args.num_clients)
    if not clients:
        raise ValueError(f"No client_* directories under {clean_root}")

    os.makedirs(out_root, exist_ok=True)

    manifest = {
        "clean_features_root": clean_root,
        "noise_dir": noise_dir,
        "snrs": snrs,
        "num_clients": len(clients),
        "splits": [],
    }

    shared_feature_files = [
        "train_labeled_features.pkl",
        "val_features.pkl",
    ]

    cached_noises = {}
    for noise_path in noise_files:
        noise_name = noise_path.stem
        cached_noises[noise_name] = _load_noise(str(noise_path), target_sr=16000)

    for noise_path in noise_files:
        noise_name = noise_path.stem
        noise_wave = cached_noises[noise_name]

        for snr_db in snrs:
            split_name = f"noisex92_{noise_name}_snr{snr_db}"
            split_root = os.path.join(out_root, split_name)
            os.makedirs(split_root, exist_ok=True)

            for client_id in clients:
                clean_client_dir = os.path.join(clean_root, client_id)
                out_client_dir = os.path.join(split_root, client_id)
                os.makedirs(out_client_dir, exist_ok=True)

                clean_test_path = os.path.join(clean_client_dir, "test_features.pkl")
                if not os.path.isfile(clean_test_path):
                    raise FileNotFoundError(f"Missing clean test features: {clean_test_path}")

                out_test_path = os.path.join(out_client_dir, "test_features.pkl")
                if os.path.isfile(out_test_path) and not args.overwrite:
                    continue

                for fname in shared_feature_files:
                    src = os.path.join(clean_client_dir, fname)
                    dst = os.path.join(out_client_dir, fname)
                    if os.path.isfile(src):
                        _link_or_copy(src, dst)

                clean_test_samples = _load_pickle(clean_test_path)
                noisy_test_samples = _build_noisy_test(
                    clean_test_samples=clean_test_samples,
                    noise_wave=noise_wave,
                    snr_db=snr_db,
                    noise_name=noise_name,
                )
                _save_pickle(out_test_path, noisy_test_samples)

            manifest["splits"].append(
                {
                    "split_name": split_name,
                    "features_root": split_root,
                    "noise_name": noise_name,
                    "snr_db": snr_db,
                }
            )
            print(f"[DONE] Built noisy features: {split_root}")

    manifest_path = os.path.join(out_root, "noise_feature_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[DONE] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
