import argparse
import csv
import glob
import json
import logging
import os
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

LABEL_MAP_4 = {
    "ang": 0,
    "hap": 1,
    "sad": 2,
    "neu": 3,
    "exc": 1,
}

LABEL_MAP_MELD = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}

EMOTION_DIRS = {
    "angry", "happy", "neutral", "sad", "surprise", "fear", "disgust",
    "excited", "exc",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _get_audio_path(item):
    if isinstance(item, dict):
        for key in ("audio_path", "path", "file_path", "filepath", "filename", "wav_path"):
            value = item.get(key)
            if value:
                return value
    elif isinstance(item, (tuple, list)) and item:
        return item[0]
    raise ValueError("Sample does not include an audio path.")


def _get_sample_id(item, audio_path):
    if isinstance(item, dict):
        for key in ("sample_id", "utt_id", "utterance_id", "id"):
            value = item.get(key)
            if value:
                return str(value)
    if audio_path:
        return os.path.basename(audio_path)
    return "unknown"


def _extract_session_from_path(path_value):
    if not path_value:
        return None
    match = re.search(r"[\\/](?:audio[\\/])?session(\d+)[\\/]", path_value, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"Session(\d+)", path_value)
    if match:
        return int(match.group(1))
    match = re.search(r"Ses(\d{2})[FM]", path_value)
    if match:
        return int(match.group(1))
    return None


def _extract_speaker_from_path(path_value):
    if not path_value:
        return None
    match = re.search(r"(Ses\d{2}[FM])", path_value)
    if match:
        return match.group(1)
    path = Path(path_value)
    parent = path.parent.name
    if parent and parent.lower() in EMOTION_DIRS:
        return path.parent.parent.name
    return parent or None


def _extract_speaker(item, audio_path):
    if isinstance(item, dict):
        for key in ("speaker", "speaker_id", "speakerId", "spk", "spk_id"):
            value = item.get(key)
            if value:
                return str(value)
    return _extract_speaker_from_path(audio_path)


def _extract_key(item, split_by):
    audio_path = _get_audio_path(item)
    session_key = _extract_session_from_path(audio_path)
    speaker_key = _extract_speaker(item, audio_path)

    if split_by == "speaker":
        return speaker_key or session_key
    return session_key or speaker_key


def _load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    return data


def _assign_clients_iid(samples, num_clients, seed):
    if not num_clients or num_clients <= 0:
        raise ValueError("num_clients must be set for IID partitioning.")
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    buckets = defaultdict(list)
    for idx, sample_idx in enumerate(indices):
        client_idx = idx % num_clients
        client_name = f"client_{client_idx:03d}"
        buckets[client_name].append(samples[sample_idx])
    return buckets, num_clients


def _assign_clients_label(samples, num_clients, seed, alpha):
    if not num_clients or num_clients <= 0:
        raise ValueError("num_clients must be set for label partitioning.")
    raw_labels = [_extract_label(item) for item in samples]
    normalized = [_normalize_label(label) for label in raw_labels]
    numeric_labels = sorted({label for label in normalized if label is not None})
    string_labels = sorted({
        label.strip().lower()
        for label in raw_labels
        if isinstance(label, str) and label.strip()
        and label.strip().lower() not in LABEL_MAP_4
        and not label.strip().isdigit()
    })
    offset = numeric_labels[-1] + 1 if numeric_labels else 0
    string_map = {label: idx + offset for idx, label in enumerate(string_labels)}

    labels = []
    for raw_label, norm_label in zip(raw_labels, normalized):
        if norm_label is None and isinstance(raw_label, str):
            norm_label = string_map.get(raw_label.strip().lower())
        if norm_label is None:
            raise ValueError("Label partitioning requires all samples to have labels.")
        labels.append(norm_label)

    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    rng = np.random.RandomState(seed)
    buckets = defaultdict(list)
    for label, indices in label_to_indices.items():
        rng.shuffle(indices)
        proportions = rng.dirichlet([alpha] * num_clients)
        counts = rng.multinomial(len(indices), proportions)
        cursor = 0
        for client_idx, count in enumerate(counts):
            if count <= 0:
                continue
            client_name = f"client_{client_idx:03d}"
            for sample_idx in indices[cursor:cursor + count]:
                buckets[client_name].append(samples[sample_idx])
            cursor += count

    return buckets, num_clients


def _assign_clients(keys, num_clients, seed):
    keys = list(sorted(keys))
    if not num_clients or num_clients <= 0 or num_clients >= len(keys):
        mapping = {key: idx for idx, key in enumerate(keys)}
        return mapping, len(keys)

    rng = random.Random(seed)
    rng.shuffle(keys)
    mapping = {key: idx % num_clients for idx, key in enumerate(keys)}
    return mapping, num_clients


def _assign_clients_session_speaker(train_pool, num_clients, seed):
    session_speakers = defaultdict(lambda: defaultdict(list))
    for item in train_pool:
        audio_path = _get_audio_path(item)
        session_key = _extract_session_from_path(audio_path)
        speaker_key = _extract_speaker(item, audio_path)
        if session_key is None:
            raise ValueError(f"Failed to extract session from {audio_path}")
        if speaker_key is None:
            raise ValueError(f"Failed to extract speaker from {audio_path}")
        session_speakers[session_key][speaker_key].append(item)

    sessions = sorted(session_speakers.keys())
    total_speakers = sum(len(speakers) for speakers in session_speakers.values())
    requested = num_clients or 0
    if requested > total_speakers:
        logging.warning(
            "Requested %d clients but only %d unique speakers available; capping clients.",
            requested, total_speakers
        )
        requested = total_speakers

    clients_per_session = {sess: 1 for sess in sessions}
    extra = requested - len(sessions)
    if extra > 0:
        candidates = []
        for sess, speakers in session_speakers.items():
            capacity = max(0, len(speakers) - 1)
            candidates.extend([sess] * capacity)
        rng = random.Random(seed)
        rng.shuffle(candidates)
        for sess in candidates:
            if extra <= 0:
                break
            clients_per_session[sess] += 1
            extra -= 1

    buckets = defaultdict(list)
    client_keys = defaultdict(set)
    client_sessions = defaultdict(set)
    rng = random.Random(seed)
    client_idx = 0
    for sess in sessions:
        speakers = sorted(session_speakers[sess].keys())
        rng.shuffle(speakers)
        k = min(clients_per_session[sess], len(speakers))
        for offset in range(k):
            client_name = f"client_{client_idx:03d}"
            client_idx += 1
            assigned = speakers[offset::k]
            for spk in assigned:
                client_keys[client_name].add(spk)
                client_sessions[client_name].add(sess)
                buckets[client_name].extend(session_speakers[sess][spk])

    return buckets, client_keys, client_sessions, client_idx


def _split_local_train_val(samples, val_ratio, seed):
    if not samples:
        return [], []

    rng = random.Random(seed)
    by_label = defaultdict(list)
    for item in samples:
        label = _normalize_label(_extract_label(item))
        if label is None:
            raise ValueError("speaker_shards partition requires labels for all samples.")
        by_label[label].append(item)

    train_samples = []
    val_samples = []
    for label in sorted(by_label):
        label_samples = list(by_label[label])
        rng.shuffle(label_samples)
        if len(label_samples) <= 1:
            train_samples.extend(label_samples)
            continue
        val_count = int(round(len(label_samples) * float(val_ratio)))
        val_count = max(1, min(len(label_samples) - 1, val_count))
        val_samples.extend(label_samples[:val_count])
        train_samples.extend(label_samples[val_count:])

    if not train_samples and val_samples:
        train_samples.append(val_samples.pop())
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def _assign_speaker_shard_clients(train_pool, shards_per_speaker, emotions_per_client, val_ratio, seed):
    if shards_per_speaker <= 0:
        raise ValueError("shards_per_speaker must be > 0.")
    if emotions_per_client <= 0:
        raise ValueError("emotions_per_client must be > 0.")

    speaker_labels = defaultdict(lambda: defaultdict(list))
    for item in train_pool:
        audio_path = _get_audio_path(item)
        speaker = _extract_speaker(item, audio_path)
        if speaker is None:
            raise ValueError(f"Failed to extract speaker from {audio_path}")
        label = _normalize_label(_extract_label(item))
        if label is None:
            raise ValueError(f"Failed to extract label from {audio_path}")
        speaker_labels[speaker][label].append(item)

    rng = random.Random(seed)
    buckets = {}
    local_vals = {}
    client_keys = defaultdict(set)
    client_sessions = defaultdict(set)
    client_meta = {}
    client_idx = 0

    for speaker in sorted(speaker_labels):
        labels = sorted(speaker_labels[speaker])
        if len(labels) < emotions_per_client:
            raise ValueError(
                f"Speaker {speaker} has only {len(labels)} emotion classes; "
                f"speaker_shards requires at least {emotions_per_client}."
            )

        shard_labels = []
        for shard_idx in range(shards_per_speaker):
            if len(labels) == emotions_per_client:
                selected = list(labels)
            else:
                excluded = labels[shard_idx % len(labels)]
                selected = [label for label in labels if label != excluded]
                selected = selected[:emotions_per_client]
            shard_labels.append(selected)

        shard_samples = [[] for _ in range(shards_per_speaker)]
        for label in labels:
            eligible = [
                shard_idx
                for shard_idx, selected in enumerate(shard_labels)
                if label in selected
            ]
            label_samples = list(speaker_labels[speaker][label])
            rng.shuffle(label_samples)
            for idx, sample in enumerate(label_samples):
                shard_samples[eligible[idx % len(eligible)]].append(sample)

        for shard_idx, samples in enumerate(shard_samples):
            if not samples:
                raise ValueError(f"Speaker {speaker} shard {shard_idx} is empty.")

            observed_labels = sorted({
                _normalize_label(_extract_label(item))
                for item in samples
            })
            if len(observed_labels) != emotions_per_client:
                raise ValueError(
                    f"Speaker {speaker} shard {shard_idx} has {len(observed_labels)} classes; "
                    f"expected exactly {emotions_per_client}."
                )

            train_samples, val_samples = _split_local_train_val(
                samples,
                val_ratio,
                seed + client_idx,
            )
            if not train_samples:
                raise ValueError(f"Speaker {speaker} shard {shard_idx} has no training samples.")

            client_name = f"client_{client_idx:03d}"
            client_idx += 1
            buckets[client_name] = train_samples
            local_vals[client_name] = val_samples
            client_keys[client_name].add(speaker)
            for item in samples:
                session = _extract_session_from_path(_get_audio_path(item))
                if session is not None:
                    client_sessions[client_name].add(session)
            client_meta[client_name] = {
                "speaker": speaker,
                "speaker_shard": shard_idx,
                "emotion_classes": observed_labels,
                "num_emotion_classes": len(observed_labels),
                "source_samples": len(samples),
                "train_ratio": 1.0 - float(val_ratio),
                "val_ratio": float(val_ratio),
            }

    return buckets, local_vals, client_keys, client_sessions, client_meta, client_idx


def _parse_speaker_list(raw_value):
    if not raw_value:
        return None
    speakers = [item.strip() for item in raw_value.split(",") if item.strip()]
    return speakers or None


def _select_speakers(candidates, list_value, ratio, seed, label):
    selected = _parse_speaker_list(list_value)
    if selected:
        missing = [spk for spk in selected if spk not in candidates]
        if missing:
            raise ValueError(f"Unknown {label}: {missing}")
        if len(selected) >= len(candidates):
            raise ValueError(f"{label} must be fewer than total speakers.")
        return sorted(selected)

    ratio = float(ratio)
    if ratio <= 0.0:
        raise ValueError(f"{label}_ratio must be > 0.")
    if ratio >= 1.0:
        raise ValueError(f"{label}_ratio must be < 1.")
    count = max(1, int(round(len(candidates) * ratio)))
    if count >= len(candidates):
        count = len(candidates) - 1

    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return sorted(shuffled[:count])


def _extract_label(item):
    if isinstance(item, dict):
        for key in ("label", "emotion"):
            value = item.get(key)
            if value is not None:
                return value
    if isinstance(item, (tuple, list)) and len(item) > 2:
        return item[2]
    return None


def _normalize_label(label):
    if isinstance(label, str):
        key = label.strip().lower()
        if key in LABEL_MAP_4:
            return LABEL_MAP_4[key]
        if key.isdigit():
            return int(key)
        return None
    if isinstance(label, (np.integer, int)):
        return int(label)
    try:
        return int(label)
    except (TypeError, ValueError):
        return None


def _save_class_distribution(client_class_counts, class_names, output_path):
    if not client_class_counts:
        return
    clients = list(client_class_counts.keys())
    counts = np.array([client_class_counts[client] for client in clients])
    num_clients = len(clients)
    num_classes = len(class_names)
    fig_width = max(6, min(0.6 * num_clients, 16))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    x = np.arange(num_clients)
    total_width = 0.8
    bar_width = total_width / max(num_classes, 1)
    offset = -total_width / 2 + bar_width / 2
    for idx, class_name in enumerate(class_names):
        ax.bar(x + offset + idx * bar_width, counts[:, idx], width=bar_width, label=class_name)
    ax.set_ylabel("Samples (train+val)")
    ax.set_xlabel("Client")
    ax.set_title("Client class distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(clients, rotation=45, ha="right", fontsize=8)
    ax.legend(title="Class", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _resolve_class_names(dataset, samples):
    if dataset == "IEMOCAP":
        return ["ang", "hap", "sad", "neu"]
    labels = [_normalize_label(_extract_label(item)) for item in samples]
    labels = [label for label in labels if label is not None]
    if not labels:
        return []
    max_label = max(labels)
    return [f"class_{idx}" for idx in range(max_label + 1)]


def _find_meld_csv(root, csv_names):
    if not root:
        return None
    for name in csv_names:
        candidate = os.path.join(root, name)
        if os.path.isfile(candidate):
            return candidate
    for dirpath, _, files in os.walk(root):
        for name in csv_names:
            if name in files:
                return os.path.join(dirpath, name)
    return None


def _build_meld_audio_index(audio_root):
    index = {}
    if not audio_root or not os.path.isdir(audio_root):
        return index
    for root, _, files in os.walk(audio_root):
        for name in files:
            if name.lower().endswith(".wav"):
                index[name] = os.path.join(root, name)
    return index


def _resolve_meld_audio_path(audio_root, split_dirs, wav_name, audio_index):
    for split_dir in split_dirs:
        candidates = [
            os.path.join(audio_root, split_dir, wav_name),
            os.path.join(audio_root, f"{split_dir}_splits", wav_name),
            os.path.join(audio_root, "audio", split_dir, wav_name),
            os.path.join(audio_root, "audio", f"{split_dir}_splits", wav_name),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
    if audio_index:
        return audio_index.get(wav_name)
    return None


def _prepare_meld_pkls(data_root, output_dir, ignore_length):
    import soundfile as sf

    split_csvs = {
        "train": ["train_sent_emo.csv"],
        "val": ["dev_sent_emo.csv", "val_sent_emo.csv", "valid_sent_emo.csv"],
        "test": ["test_sent_emo.csv"],
    }
    split_dirs = {
        "train": ["train"],
        "val": ["dev", "val", "valid"],
        "test": ["test"],
    }

    os.makedirs(output_dir, exist_ok=True)
    audio_index = _build_meld_audio_index(data_root)
    if not audio_index:
        logging.warning("No audio files indexed under %s", data_root)

    for split_name, csv_names in split_csvs.items():
        csv_path = _find_meld_csv(data_root, csv_names)
        if csv_path is None:
            raise ValueError(f"Could not find MELD CSV for {split_name} under {data_root}")

        split_dir_candidates = split_dirs.get(split_name, [])
        samples = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in tqdm.tqdm(reader, desc=f"Processing MELD {split_name}"):
                emotion = (row.get("Emotion") or row.get("emotion") or "").strip().lower()
                if emotion not in LABEL_MAP_MELD:
                    continue
                text = row.get("Utterance") or row.get("utterance") or row.get("Text") or row.get("text") or ""
                speaker = (row.get("Speaker") or row.get("speaker")
                           or row.get("Speaker_id") or row.get("speaker_id") or "").strip()
                if not speaker:
                    logging.warning("Missing speaker in MELD %s row; skipping.", split_name)
                    continue
                dialog_id = row.get("Dialogue_ID") or row.get("Dialogue ID") or row.get("dialogue_id")
                utt_id = row.get("Utterance_ID") or row.get("Utterance ID") or row.get("utterance_id")
                if dialog_id is None or utt_id is None:
                    continue

                try:
                    dialog_id_int = int(float(dialog_id))
                    utt_id_int = int(float(utt_id))
                except (TypeError, ValueError):
                    continue

                file_base = f"dia{dialog_id_int}_utt{utt_id_int}"
                wav_name = f"{file_base}.wav"
                audio_path = _resolve_meld_audio_path(
                    data_root, split_dir_candidates, wav_name, audio_index
                )
                if audio_path is None:
                    continue

                try:
                    wav_data, _ = sf.read(audio_path, dtype="int16")
                except Exception:
                    logging.warning("Cannot read %s", audio_path)
                    continue

                if ignore_length and len(wav_data) < ignore_length:
                    logging.warning("Ignored short sample: %s", audio_path)
                    continue

                label = LABEL_MAP_MELD[emotion]
                sample_id = f"{dialog_id_int}_{utt_id_int}"
                samples.append({
                    "audio_path": audio_path,
                    "text": text,
                    "label": label,
                    "emotion": emotion,
                    "speaker": speaker,
                    "dialogue_id": dialog_id_int,
                    "utterance_id": utt_id_int,
                    "sample_id": sample_id,
                })

        out_path = os.path.join(output_dir, f"{split_name}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(samples, f)
        logging.info("%s: %d samples saved to %s", split_name.upper(), len(samples), out_path)

    return output_dir


def _collect_IEMOCAP_samples(args):
    import soundfile as sf

    session_ids = list(range(1, 6))
    ignore_length = args.ignore_length
    seed = args.seed
    data_root = args.data_root

    label_map = LABEL_MAP_4
    valid_emotions = {"ang", "hap", "sad", "neu", "exc"}

    session_samples = {sess_id: [] for sess_id in session_ids}

    for sess_id in tqdm.tqdm(session_ids, desc="Processing IEMOCAP"):
        sess_path = os.path.join(data_root, f"Session{sess_id}")
        audio_root = os.path.join(sess_path, "sentences/wav")
        text_root = os.path.join(sess_path, "dialog/transcriptions")
        label_root = os.path.join(sess_path, "dialog/EmoEvaluation")
        label_files = glob.glob(os.path.join(label_root, "*.txt"))

        for label_file in label_files:
            base_name = os.path.basename(label_file)
            transcript_file = os.path.join(text_root, base_name)

            with open(transcript_file, "r") as f:
                transcript_lines = {
                    line.split(":")[0]: line.split(":")[1].strip()
                    for line in f.readlines()
                }

            with open(label_file, "r") as f:
                for line in f:
                    if not line.startswith("["):
                        continue
                    data = line[1:].split()
                    start_time = float(data[0])
                    end_time = float(data[2][:-1])
                    utt_id = data[3]
                    emotion = data[4]

                    if emotion not in valid_emotions:
                        continue

                    folder = utt_id[:-5]
                    wav_name = utt_id + ".wav"
                    wav_path = os.path.join(audio_root, folder, wav_name)

                    try:
                        wav_data, _ = sf.read(wav_path, dtype="int16")
                    except Exception:
                        logging.warning("Cannot read %s", wav_path)
                        continue

                    if len(wav_data) < ignore_length:
                        logging.warning("Ignored short sample: %s", wav_path)
                        continue

                    text_key = f"{utt_id} [{start_time:08.4f}-{end_time:08.4f}]"
                    text = transcript_lines.get(text_key)

                    if text is None:
                        text_key_alt1 = f"{utt_id} [{start_time:08.4f}-{end_time + 0.0001:08.4f}]"
                        text_key_alt2 = f"{utt_id} [{start_time + 0.0001:08.4f}-{end_time:08.4f}]"
                        text = transcript_lines.get(text_key_alt1) or transcript_lines.get(text_key_alt2)

                    if text is None:
                        logging.warning("Transcript not found: %s", text_key)
                        continue

                    label = label_map.get(emotion)
                    if label is None:
                        continue

                    session_samples[sess_id].append((wav_path, text, label))

    rng = random.Random(seed)
    for sess_id in session_ids:
        rng.shuffle(session_samples[sess_id])

    return session_samples


def _build_speaker_map(session_samples):
    speaker_map = defaultdict(list)
    for samples in session_samples.values():
        for item in samples:
            audio_path = _get_audio_path(item)
            speaker_key = _extract_speaker(item, audio_path)
            if speaker_key is None:
                raise ValueError(f"Failed to extract speaker from {audio_path}")
            speaker_map[speaker_key].append(item)
    return speaker_map


def _prepare_iemocap_splits(args):
    if args.split_by != "speaker_shards" and args.test_by != args.val_by:
        raise ValueError("test_by and val_by must match (session or speaker).")

    session_samples = _collect_IEMOCAP_samples(args)
    session_ids = sorted(session_samples.keys())

    test_samples = []
    val_samples = []
    train_pool = []
    test_speakers = None
    val_speakers = None
    train_speakers = None

    if args.test_by == "session":
        if args.test_session not in session_samples:
            raise ValueError(f"test_session must be in {session_ids}")

        test_samples = list(session_samples[args.test_session])
        if args.split_by == "speaker_shards":
            val_samples = []
            train_sessions = [s for s in session_ids if s != args.test_session]
        else:
            if args.val_session not in session_samples:
                raise ValueError(f"val_session must be in {session_ids}")
            if args.test_session == args.val_session:
                raise ValueError("test_session and val_session must be different.")
            val_samples = list(session_samples[args.val_session])
            train_sessions = [s for s in session_ids if s not in (args.test_session, args.val_session)]
        for sess_id in train_sessions:
            train_pool.extend(session_samples[sess_id])
        if args.split_by == "speaker_shards":
            train_speakers = sorted({
                _extract_speaker(item, _get_audio_path(item))
                for item in train_pool
            })
    else:
        if args.split_by != "speaker":
            raise ValueError("test_by speaker requires split_by speaker.")

        speaker_map = _build_speaker_map(session_samples)
        speakers = sorted(speaker_map.keys())
        test_speakers = _select_speakers(
            speakers, args.test_speakers, args.test_speaker_ratio, args.seed, "test_speakers"
        )
        remaining = [spk for spk in speakers if spk not in test_speakers]
        if args.split_by == "speaker_shards":
            val_speakers = []
            train_speakers = list(remaining)
        else:
            val_speakers = _select_speakers(
                remaining, args.val_speakers, args.val_speaker_ratio, args.seed, "val_speakers"
            )
            train_speakers = [spk for spk in remaining if spk not in val_speakers]
        for spk in train_speakers:
            train_pool.extend(speaker_map[spk])
        for spk in test_speakers:
            test_samples.extend(speaker_map[spk])
        if args.split_by != "speaker_shards":
            for spk in val_speakers:
                val_samples.extend(speaker_map[spk])
        train_sessions = sorted({
            _extract_session_from_path(_get_audio_path(item))
            for item in train_pool
        })

    return {
        "train_pool": train_pool,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "train_sessions": train_sessions,
        "train_speakers": train_speakers,
        "val_speakers": val_speakers,
        "test_speakers": test_speakers,
    }


def _prepare_meld_splits(args):
    meld_dir = args.meld_dir
    train_pkl = args.meld_train_pkl
    val_pkl = args.meld_val_pkl
    test_pkl = args.meld_test_pkl

    if meld_dir and not train_pkl and not val_pkl and not test_pkl:
        train_pkl = os.path.join(meld_dir, "train.pkl")
        val_pkl = os.path.join(meld_dir, "val.pkl")
        test_pkl = os.path.join(meld_dir, "test.pkl")

    if not train_pkl or not val_pkl or not test_pkl or not (
        os.path.isfile(train_pkl) and os.path.isfile(val_pkl) and os.path.isfile(test_pkl)
    ):
        data_root = args.data_root or meld_dir
        if not data_root:
            raise ValueError("MELD preprocessing requires train/val/test PKLs or --data_root to build from CSV.")
        meld_out_dir = args.meld_out_dir or "metadata/MELD_preprocessed"
        meld_dir = _prepare_meld_pkls(data_root, meld_out_dir, args.ignore_length)
        train_pkl = os.path.join(meld_dir, "train.pkl")
        val_pkl = os.path.join(meld_dir, "val.pkl")
        test_pkl = os.path.join(meld_dir, "test.pkl")

    train_pool = list(_load_pickle(train_pkl))
    val_samples = list(_load_pickle(val_pkl))
    test_samples = list(_load_pickle(test_pkl))

    if args.split_by == "speaker":
        first_item = train_pool[0] if train_pool else None
        needs_rebuild = not (isinstance(first_item, dict) and first_item.get("speaker"))
        if needs_rebuild:
            data_root = args.data_root or meld_dir
            if not data_root:
                raise ValueError("MELD speaker split requires data_root to rebuild PKLs.")
            meld_out_dir = args.meld_out_dir or "metadata/MELD_preprocessed"
            logging.info("Rebuilding MELD PKLs to include speaker info.")
            meld_dir = _prepare_meld_pkls(data_root, meld_out_dir, args.ignore_length)
            train_pool = list(_load_pickle(os.path.join(meld_dir, "train.pkl")))
            val_samples = list(_load_pickle(os.path.join(meld_dir, "val.pkl")))
            test_samples = list(_load_pickle(os.path.join(meld_dir, "test.pkl")))

    return {
        "train_pool": train_pool,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "train_sessions": [],
        "train_speakers": None,
        "val_speakers": None,
        "test_speakers": None,
    }


def _partition_train_pool(train_pool, args):
    local_val_buckets = {}
    client_meta = {}
    if args.split_by == "speaker_shards":
        (
            buckets,
            local_val_buckets,
            client_keys,
            client_sessions,
            client_meta,
            total_clients,
        ) = _assign_speaker_shard_clients(
            train_pool,
            args.shards_per_speaker,
            args.emotions_per_client,
            args.client_val_ratio,
            args.seed,
        )
        effective_split_by = (
            f"speaker_shards_{args.shards_per_speaker}x_"
            f"{args.emotions_per_client}emotions_local80_20"
        )
    else:
        client_keys = defaultdict(set)
        client_sessions = defaultdict(set)
        buckets = defaultdict(list)
        total_clients = args.num_clients or 0
        effective_split_by = args.split_by

        if args.split_by in ("speaker", "session"):
            keys = set()
            for item in train_pool:
                key = _extract_key(item, args.split_by)
                if key is None:
                    raise ValueError("Failed to extract split key for a sample.")
                keys.add(key)

            if args.split_by == "session" and args.num_clients and args.num_clients > len(keys):
                logging.info(
                    "Requested %d clients but only %d train sessions; splitting by speaker within sessions.",
                    args.num_clients, len(keys)
                )
                buckets, client_keys, client_sessions, total_clients = _assign_clients_session_speaker(
                    train_pool, args.num_clients, args.seed
                )
                effective_split_by = "session+speaker"
            else:
                key_to_client, total_clients = _assign_clients(keys, args.num_clients, args.seed)
                for item in train_pool:
                    key = _extract_key(item, args.split_by)
                    client_idx = key_to_client[key]
                    client_name = f"client_{client_idx:03d}"
                    buckets[client_name].append(item)
                    client_keys[client_name].add(key)
        elif args.split_by == "iid":
            buckets, total_clients = _assign_clients_iid(train_pool, args.num_clients, args.seed)
        elif args.split_by == "label":
            buckets, total_clients = _assign_clients_label(
                train_pool, args.num_clients, args.seed, args.dirichlet_alpha
            )
            for client_name, samples in buckets.items():
                for item in samples:
                    label = _normalize_label(_extract_label(item))
                    if label is not None:
                        client_keys[client_name].add(label)
        else:
            raise ValueError(f"Unsupported split_by: {args.split_by}")

    return (
        buckets,
        local_val_buckets,
        client_keys,
        client_sessions,
        client_meta,
        total_clients,
        effective_split_by,
    )


def main():
    parser = argparse.ArgumentParser(description="Federated preprocessing with shared test session")
    parser.add_argument("--dataset", type=str, choices=["IEMOCAP", "MELD"], required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for client PKLs")
    parser.add_argument(
        "--split_by",
        type=str,
        default="iid",
        choices=["speaker", "speaker_shards", "session", "iid", "label"],
    )
    parser.add_argument("--num_clients", type=int, default=None)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3,
                        help="Dirichlet alpha for label-based partitioning.")
    parser.add_argument("--meld_dir", type=str, default=None,
                        help="Directory containing MELD train/val/test PKLs.")
    parser.add_argument("--meld_train_pkl", type=str, default=None)
    parser.add_argument("--meld_val_pkl", type=str, default=None)
    parser.add_argument("--meld_test_pkl", type=str, default=None)
    parser.add_argument("--meld_out_dir", type=str, default=None,
                        help="Output directory for MELD train/val/test PKLs when built from CSV.")
    parser.add_argument("--test_by", type=str, default="session", choices=["session", "speaker"])
    parser.add_argument("--test_session", type=int, default=5, help="Session id used as shared test set")
    parser.add_argument("--test_speakers", type=str, default=None,
                        help="Comma-separated speaker ids for shared test set (e.g., Ses01F,Ses01M)")
    parser.add_argument("--test_speaker_ratio", type=float, default=0.2,
                        help="Ratio of speakers reserved for shared test set when --test_by speaker")
    parser.add_argument("--val_by", type=str, default="session", choices=["session", "speaker"])
    parser.add_argument("--val_session", type=int, default=4, help="Session id used as shared validation set")
    parser.add_argument("--val_speakers", type=str, default=None,
                        help="Comma-separated speaker ids for shared validation set (e.g., Ses02F,Ses02M)")
    parser.add_argument("--val_speaker_ratio", type=float, default=0.1,
                        help="Ratio of speakers reserved for shared validation set when --val_by speaker")
    parser.add_argument("--min_labeled_per_client", type=int, default=1)
    parser.add_argument("--shards_per_speaker", type=int, default=4)
    parser.add_argument("--emotions_per_client", type=int, default=3)
    parser.add_argument("--client_val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore_length", type=int, default=0)
    args = parser.parse_args()

    if args.dataset == "IEMOCAP" and not args.data_root:
        raise ValueError("--data_root is required for IEMOCAP preprocessing.")
    if args.dataset == "MELD" and args.split_by == "iid":
        logging.info("MELD defaulting to speaker-based partition.")
        args.split_by = "speaker"
    if args.dataset == "MELD" and args.split_by == "session":
        raise ValueError("MELD preprocessing does not support session splits.")
    if args.split_by == "speaker_shards" and args.dataset != "IEMOCAP":
        raise ValueError("speaker_shards partition is currently implemented for IEMOCAP only.")

    if args.dataset == "IEMOCAP":
        split_info = _prepare_iemocap_splits(args)
    else:
        split_info = _prepare_meld_splits(args)

    train_pool = split_info["train_pool"]
    val_samples = split_info["val_samples"]
    test_samples = split_info["test_samples"]
    train_sessions = split_info["train_sessions"]
    train_speakers = split_info["train_speakers"]
    val_speakers = split_info["val_speakers"]
    test_speakers = split_info["test_speakers"]

    if not train_pool:
        raise ValueError("No training samples available after selecting shared test set.")

    rng = random.Random(args.seed)
    rng.shuffle(train_pool)
    rng.shuffle(test_samples)
    rng.shuffle(val_samples)

    (
        buckets,
        local_val_buckets,
        client_keys,
        client_sessions,
        client_meta,
        total_clients,
        effective_split_by,
    ) = _partition_train_pool(train_pool, args)

    os.makedirs(args.out_dir, exist_ok=True)
    client_stats = {}
    sample_map = {}
    sample_split = {}
    all_local_val_samples = []

    for client_name, samples in buckets.items():
        train_samples = list(samples)
        labeled = train_samples
        val_samples_client = list(local_val_buckets.get(client_name, val_samples))
        test_samples_client = list(test_samples)
        all_local_val_samples.extend(val_samples_client)

        client_dir = os.path.join(args.out_dir, client_name)
        os.makedirs(client_dir, exist_ok=True)

        with open(os.path.join(client_dir, "train_labeled.pkl"), "wb") as f:
            pickle.dump(labeled, f)
        with open(os.path.join(client_dir, "val.pkl"), "wb") as f:
            pickle.dump(val_samples_client, f)
        with open(os.path.join(client_dir, "test.pkl"), "wb") as f:
            pickle.dump(test_samples_client, f)

        for split_name, split_samples in (
            ("train_labeled", labeled),
            ("val", val_samples_client),
        ):
            for item in split_samples:
                audio_path = _get_audio_path(item)
                sample_id = _get_sample_id(item, audio_path)
                sample_map[sample_id] = client_name
                sample_split[sample_id] = split_name

        client_stats[client_name] = {
            "keys": sorted(client_keys[client_name]) if client_keys[client_name] else [],
            "train_total": len(train_samples),
            "train_labeled": len(labeled),
            "val": len(val_samples_client),
            "test": len(test_samples_client),
        }
        if client_name in client_meta:
            client_stats[client_name].update(client_meta[client_name])
        if client_sessions:
            client_stats[client_name]["sessions"] = sorted(client_sessions.get(client_name, []))

    if args.split_by == "speaker_shards":
        val_samples = all_local_val_samples

    if not val_samples:
        raise ValueError("Shared validation set is empty. Adjust val_session or val_speakers.")
    if not test_samples:
        raise ValueError("Shared test set is empty. Adjust test_session or test_speakers.")

    client_totals = {}
    class_names = _resolve_class_names(args.dataset, train_pool)
    client_class_counts = {}
    for client_name in sorted(client_stats):
        stats = client_stats[client_name]
        total = stats["train_total"] + stats["val"]
        client_totals[client_name] = total
        logging.info("Client %s total samples (train+val): %d", client_name, total)
        if class_names:
            client_class_counts[client_name] = [0 for _ in class_names]
            for item in list(buckets[client_name]) + list(local_val_buckets.get(client_name, [])):
                label = _normalize_label(_extract_label(item))
                if label is None or label < 0 or label >= len(class_names):
                    continue
                client_class_counts[client_name][label] += 1

    if client_class_counts and class_names:
        class_plot_path = os.path.join(args.out_dir, "client_class_distribution.png")
        _save_class_distribution(client_class_counts, class_names, class_plot_path)
        logging.info("Saved client class distribution plot to %s", class_plot_path)

    shared_val_path = None
    local_val_path = None
    if args.split_by == "speaker_shards":
        local_val_path = os.path.join(args.out_dir, "all_client_val.pkl")
        with open(local_val_path, "wb") as f:
            pickle.dump(val_samples, f)
    else:
        shared_val_path = os.path.join(args.out_dir, "shared_val.pkl")
        with open(shared_val_path, "wb") as f:
            pickle.dump(val_samples, f)

    shared_test_path = os.path.join(args.out_dir, "shared_test.pkl")
    with open(shared_test_path, "wb") as f:
        pickle.dump(test_samples, f)

    shared_map_splits = [("test", test_samples)]
    if args.split_by != "speaker_shards":
        shared_map_splits.append(("val", val_samples))

    for split_name, split_samples in shared_map_splits:
        for item in split_samples:
            audio_path = _get_audio_path(item)
            sample_id = _get_sample_id(item, audio_path)
            sample_map[sample_id] = "shared"
            sample_split[sample_id] = split_name

    client_map = {
        "split_by": args.split_by,
        "split_by_effective": effective_split_by,
        "dataset": args.dataset,
        "dirichlet_alpha": args.dirichlet_alpha if args.split_by == "label" else None,
        "num_clients": total_clients,
        "requested_num_clients": args.num_clients,
        "seed": args.seed,
        "test_by": args.test_by if args.dataset == "IEMOCAP" else "predefined",
        "test_session": args.test_session if args.dataset == "IEMOCAP" and args.test_by == "session" else None,
        "test_speakers": test_speakers if args.dataset == "IEMOCAP" else None,
        "test_speaker_ratio": args.test_speaker_ratio if args.dataset == "IEMOCAP" and args.test_by == "speaker" else None,
        "val_by": None if args.split_by == "speaker_shards" else (args.val_by if args.dataset == "IEMOCAP" else "predefined"),
        "val_session": None if args.split_by == "speaker_shards" else (
            args.val_session if args.dataset == "IEMOCAP" and args.val_by == "session" else None
        ),
        "val_speakers": None if args.split_by == "speaker_shards" else (
            val_speakers if args.dataset == "IEMOCAP" else None
        ),
        "val_speaker_ratio": None if args.split_by == "speaker_shards" else (
            args.val_speaker_ratio if args.dataset == "IEMOCAP" and args.val_by == "speaker" else None
        ),
        "train_speakers": train_speakers,
        "train_sessions": train_sessions,
        "min_labeled_per_client": args.min_labeled_per_client,
        "shards_per_speaker": args.shards_per_speaker if args.split_by == "speaker_shards" else None,
        "emotions_per_client": args.emotions_per_client if args.split_by == "speaker_shards" else None,
        "client_val_ratio": args.client_val_ratio if args.split_by == "speaker_shards" else None,
        "clients": client_stats,
        "shared_val": os.path.abspath(shared_val_path) if shared_val_path else None,
        "all_client_val": os.path.abspath(local_val_path) if local_val_path else None,
        "shared_test": os.path.abspath(shared_test_path),
        "sample_map": sample_map,
        "sample_split": sample_split,
    }

    map_path = os.path.join(args.out_dir, "client_map.json")
    with open(map_path, "w") as f:
        json.dump(client_map, f, indent=2)

    if args.test_by == "session" and args.split_by == "speaker_shards":
        logging.info(
            "Train sessions: %s | Local client validation ratio: %.3f | Shared test session: %s",
            train_sessions, args.client_val_ratio, args.test_session
        )
    elif args.test_by == "session":
        logging.info(
            "Train sessions: %s | Shared val session: %s | Shared test session: %s",
            train_sessions, args.val_session, args.test_session
        )
    else:
        logging.info(
            "Train speakers: %s | Shared val speakers: %s | Shared test speakers: %s",
            train_speakers, val_speakers, test_speakers
        )
    if args.split_by == "speaker_shards":
        logging.info("Local validation samples across clients: %d", len(val_samples))
    else:
        logging.info("Shared val samples: %d", len(val_samples))
    logging.info("Shared test samples: %d", len(test_samples))
    logging.info("Saved clients to %s", args.out_dir)


if __name__ == "__main__":
    main()
