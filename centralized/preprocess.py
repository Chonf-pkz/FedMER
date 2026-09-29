import argparse
import csv
import glob
import logging
import os
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path
import soundfile as sf
import tqdm
import numpy as np
import torch
from sklearn.model_selection import train_test_split

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
    "exc": 1 
}

LABEL_MAP_ESD = {
    "Angry": 0,
    "Happy": 1,
    "Neutral": 2,
    "Sad": 3,
    "Surprise": 4
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
LABEL_MAP_MSP_IMPROV_4 = {
    "A": 0,
    "H": 1,
    "S": 2,
    "N": 3,
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

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


def _extract_iemocap_speaker(path_value):
    match = re.search(r"(Ses\d{2}[FM])", path_value)
    return match.group(1) if match else None


def _extract_iemocap_session(path_value):
    match = re.search(r"Session(\d+)", path_value)
    if match:
        return int(match.group(1))
    match = re.search(r"Ses(\d{2})[FM]", path_value)
    if match:
        return int(match.group(1))
    return None


def _collect_iemocap_loso_samples(data_root, ignore_length):
    session_ids = list(range(1, 6))
    valid_emotions = {"ang", "hap", "sad", "neu", "exc"}
    speaker_map = defaultdict(list)
    session_map = defaultdict(list)

    for sess_id in tqdm.tqdm(session_ids, desc="Processing IEMOCAP"):
        sess_path = os.path.join(data_root, f"Session{sess_id}")
        audio_root = os.path.join(sess_path, "sentences/wav")
        text_root = os.path.join(sess_path, "dialog/transcriptions")
        label_root = os.path.join(sess_path, "dialog/EmoEvaluation")
        label_files = glob.glob(os.path.join(label_root, "*.txt"))

        for label_file in label_files:
            base_name = os.path.basename(label_file)
            transcript_file = os.path.join(text_root, base_name)
            transcript_lines = {}
            with open(transcript_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    transcript_lines[key] = value.strip()

            with open(label_file, "r", encoding="utf-8", errors="ignore") as f:
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

                    if ignore_length and len(wav_data) < ignore_length:
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

                    label = LABEL_MAP_4.get(emotion)
                    if label is None:
                        continue

                    speaker = _extract_iemocap_speaker(wav_path)
                    session_id = _extract_iemocap_session(wav_path)
                    if speaker is None or session_id is None:
                        raise ValueError(f"Failed to parse speaker/session from {wav_path}")

                    sample = (wav_path, text, label)
                    speaker_map[speaker].append(sample)
                    session_map[session_id].append(sample)

    return speaker_map, session_map


def _collect_msp_improv_loso_samples(data_root, ignore_length):
    audio_root = os.path.join(data_root, "Audio")
    transcript_root = os.path.join(data_root, "Human_transcriptions", "All_human_transcriptions")
    if not os.path.isdir(audio_root):
        raise FileNotFoundError(f"MSP-IMPROV audio root not found: {audio_root}")

    transcript_map = {}
    if os.path.isdir(transcript_root):
        for transcript_path in Path(transcript_root).glob("*.txt"):
            try:
                with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = " ".join(line.strip() for line in f if line.strip())
                transcript_map[transcript_path.stem] = text
            except OSError:
                continue

    speaker_map = defaultdict(list)
    session_map = defaultdict(list)
    pattern = re.compile(r"MSP-IMPROV-S(\d{2})([A-Za-z])-([FM]\d{2})-")
    wav_files = [
        str(path)
        for path in Path(audio_root).rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    ]
    for wav_path in tqdm.tqdm(wav_files, desc="Processing MSP-IMPROV"):
        base_name = os.path.basename(wav_path)
        stem = os.path.splitext(base_name)[0]
        match = pattern.match(stem)
        if not match:
            continue
        session_token, emotion_code, speaker = match.groups()
        label = LABEL_MAP_MSP_IMPROV_4.get(emotion_code.upper())
        if label is None:
            continue

        if ignore_length and ignore_length > 0:
            try:
                wav_data, _ = sf.read(wav_path, dtype="int16")
            except Exception:
                logging.warning("Cannot read %s", wav_path)
                continue
            if len(wav_data) < ignore_length:
                logging.warning("Ignored short sample: %s", wav_path)
                continue

        text = transcript_map.get(stem, "")

        session_match = re.search(r"[\\/](?:audio[\\/])?session(\d+)[\\/]", wav_path, flags=re.IGNORECASE)
        if session_match:
            session_id = int(session_match.group(1))
        else:
            session_id = int(session_token)

        sample = (wav_path, text, label)
        speaker_map[speaker].append(sample)
        session_map[session_id].append(sample)

    return speaker_map, session_map


def collect_loso_samples(dataset, data_root, ignore_length=0, seed=0):
    if dataset == "IEMOCAP":
        speaker_map, session_map = _collect_iemocap_loso_samples(data_root, ignore_length)
    elif dataset == "MSP-IMPROV":
        speaker_map, session_map = _collect_msp_improv_loso_samples(data_root, ignore_length)
    else:
        raise ValueError(f"Unsupported LOSO dataset: {dataset}")

    rng = random.Random(seed)
    for speaker in speaker_map:
        rng.shuffle(speaker_map[speaker])
    for session_id in session_map:
        rng.shuffle(session_map[session_id])
    return speaker_map, session_map

def preprocess_iemocap(args):
    _, session_samples = collect_loso_samples(
        dataset="IEMOCAP",
        data_root=args.data_root,
        ignore_length=args.ignore_length,
        seed=args.seed,
    )
    session_ids = sorted(session_samples.keys())

    val_session = args.val_session
    test_session = args.test_session
    if val_session == test_session:
        raise ValueError("val_session and test_session must be different.")
    if val_session not in session_ids or test_session not in session_ids:
        raise ValueError(f"Session ids must be within {session_ids}.")

    train_sessions = [s for s in session_ids if s not in (val_session, test_session)]

    train_samples = []
    for sess_id in train_sessions:
        train_samples.extend(session_samples[sess_id])
    val_samples = list(session_samples[val_session])
    test_samples = list(session_samples[test_session])

    rng = random.Random(args.seed)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    output_dir = "metadata/IEMOCAP_preprocessed/IEMOCAP_4class_preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(output_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(output_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(
        "Session split - Train sessions: %s | Val session: %s | Test session: %s",
        train_sessions, val_session, test_session
    )
    logging.info(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    logging.info(f"Saved preprocessed data to {output_dir}")


def preprocess_msp_improv(args):
    speaker_map, _ = collect_loso_samples(
        dataset="MSP-IMPROV",
        data_root=args.data_root,
        ignore_length=args.ignore_length,
        seed=args.seed,
    )
    samples = []
    for spk in sorted(speaker_map.keys()):
        samples.extend(speaker_map[spk])
    random.Random(args.seed).shuffle(samples)

    labels = [s[2] for s in samples]
    train, test_samples = train_test_split(samples, test_size=0.2, random_state=args.seed, stratify=labels)
    val_samples, test_samples = train_test_split(
        test_samples,
        test_size=0.5,
        random_state=args.seed,
        stratify=[s[2] for s in test_samples]
    )

    output_dir = args.out_dir or "metadata/MSP_IMPROV_preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), "wb") as f:
        pickle.dump(train, f)
    with open(os.path.join(output_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(output_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info("MSP-IMPROV - Train: %d | Val: %d | Test: %d", len(train), len(val_samples), len(test_samples))
    logging.info("MSP-IMPROV - Saved preprocessed data to %s", output_dir)

def _preprocess_esd_language(args, speaker_ids, desc_label, output_subdir):
    data_root = args.data_root
    ignore_length = args.ignore_length
    seed = args.seed
    speaker_id_set = set(speaker_ids)

    speaker_dirs = []
    for spk in sorted(os.listdir(data_root)):
        if spk.isdigit() and int(spk) in speaker_id_set:
            speaker_dirs.append(os.path.join(data_root, spk))

    samples = []

    for spk_dir in tqdm.tqdm(speaker_dirs, desc=desc_label):
        for emo in os.listdir(spk_dir):
            emo_path = os.path.join(spk_dir, emo)
            if not os.path.isdir(emo_path):
                continue

            if emo not in LABEL_MAP_ESD:
                continue

            label = LABEL_MAP_ESD[emo]
            wav_files = glob.glob(os.path.join(emo_path, "*.wav"))

            for wav_path in wav_files:
                try:
                    wav_data, _ = sf.read(wav_path, dtype="int16")
                except Exception:
                    logging.warning(f"Cannot read {wav_path}")
                    continue

                if len(wav_data) < ignore_length:
                    logging.warning(f"Ignored short sample: {wav_path}")
                    continue

                text = ""  
                samples.append((wav_path, text, label))

    random.Random(seed).shuffle(samples)

    labels = [s[2] for s in samples]
    train, test_samples = train_test_split(samples, test_size=0.2, random_state=seed, stratify=labels)
    val_samples, test_samples = train_test_split(test_samples, test_size=0.5, random_state=seed,
                                                stratify=[s[2] for s in test_samples])

    output_dir = os.path.join("metadata", output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), "wb") as f:
        pickle.dump(train, f)
    with open(os.path.join(output_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(output_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(f"{desc_label} - Train: {len(train)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    logging.info(f"{desc_label} - Saved preprocessed data to {output_dir}")


def preprocess_esd(args):
    _preprocess_esd_language(
        args=args,
        speaker_ids=range(11, 21),
        desc_label="Processing ESD English",
        output_subdir="ESD_preprocessed",
    )


def preprocess_meld(args):
    data_root = args.data_root
    ignore_length = args.ignore_length

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

    output_dir = args.out_dir or "metadata/MELD_preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    audio_index = _build_meld_audio_index(data_root)
    if not audio_index:
        logging.warning("No audio files indexed under %s", data_root)

    for split_name, csv_names in split_csvs.items():
        csv_path = _find_meld_csv(data_root, csv_names)
        if csv_path is None:
            logging.error("Could not find MELD CSV for %s in %s", split_name, data_root)
            continue

        split_dir_candidates = split_dirs.get(split_name, [])
        samples = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in tqdm.tqdm(reader, desc=f"Processing MELD {split_name}"):
                emotion = (row.get("Emotion") or row.get("emotion") or "").strip().lower()
                if emotion not in LABEL_MAP_MELD:
                    continue
                text = row.get("Utterance") or row.get("utterance") or row.get("Text") or row.get("text") or ""
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
                samples.append((audio_path, text, label))

        out_path = os.path.join(output_dir, f"{split_name}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(samples, f)
        logging.info("%s: %d samples saved to %s", split_name.upper(), len(samples), out_path)

    logging.info("Saved MELD preprocessed data to %s", output_dir)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["IEMOCAP", "MSP-IMPROV", "ESD", "MELD"],
        required=True,
    )
    parser.add_argument("--data_root", type=str, required=True, help="Root path to dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore_length", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Optional output directory. Currently used for MELD preprocessing.")
    parser.add_argument("--val_session", type=int, default=4, help="IEMOCAP session id used for validation.")
    parser.add_argument("--test_session", type=int, default=5, help="IEMOCAP session id used for test.")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    if args.dataset == "IEMOCAP":
        preprocess_iemocap(args)
    elif args.dataset == "MSP-IMPROV":
        preprocess_msp_improv(args)
    elif args.dataset == "ESD":
        preprocess_esd(args)
    elif args.dataset == "MELD":
        preprocess_meld(args)
