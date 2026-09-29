import os
import sys
import json
import time
import pickle
import warnings
import argparse
import re
from collections import defaultdict

import numpy as np
import torch
import soundfile as sf
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore")
p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
print(p)
sys.path.append(p)

from src.feature_extract.config import (
    PKL_DIR, OUTPUT_DIR, device,
    TOKENIZER, AUDIO_PROCESSOR,
    TEXT_MODEL, AUDIO_MODEL
)


NRC_EMOTIONS = [
    "anger", "fear", "anticipation", "trust", "surprise",
    "sadness", "joy", "disgust", "positive", "negative",
]


def load_nrc_lexicon(path):
    if not path:
        return None
    lexicon = defaultdict(set)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, emotion, association = parts
            if association == "1":
                lexicon[word.lower()].add(emotion)
    return lexicon


def clean_text(text):
    text = "" if text is None else str(text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"[^a-zA-Z'\s]", " ", text)
    return text.lower()


def tokenize(text):
    return re.findall(r"[a-z']+", clean_text(text))


def extract_nrc_features(text, lexicon):
    tokens = tokenize(text)
    counts = {emotion: 0 for emotion in NRC_EMOTIONS}

    for token in tokens:
        for emotion in lexicon.get(token, ()):
            if emotion in counts:
                counts[emotion] += 1

    n_words = len(tokens)
    values = [
        counts[emotion] / n_words if n_words > 0 else 0.0
        for emotion in NRC_EMOTIONS
    ]
    values.extend([float(sum(counts.values())), float(n_words)])
    names = [f"nrc_{emotion}" for emotion in NRC_EMOTIONS] + [
        "nrc_matched_words",
        "nrc_total_words",
    ]
    return torch.tensor(values, dtype=torch.float32), names



def _load_audio(audio_path):
    array, sr = sf.read(audio_path)

    if isinstance(array, np.ndarray):
        waveform = array.astype(np.float32)
    else:
        waveform = np.array(array, dtype=np.float32)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000

    return np.ascontiguousarray(waveform), sr


def _extract_audio_embedding(waveform, sr, processor, model, device, source=None):
    try:
        inputs = processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            pooled, _ = model(inputs["input_values"])
        return pooled.squeeze().cpu()
    except Exception as e:
        prefix = f"{source}: " if source else ""
        print(f"[ERROR] Audio failed: {prefix}({e})")
        return None


def extract_audio_features(audio_path, processor, model, device):
    try:
        waveform, sr = _load_audio(audio_path)
    except Exception as e:
        print(f"[ERROR] Audio failed: {audio_path} ({e})")
        return None
    return _extract_audio_embedding(
        waveform,
        sr,
        processor,
        model,
        device,
        source=audio_path
    )

def extract_text_features(text, tokenizer, model, device):
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(device)
        with torch.no_grad():
            pooled, _ = model(inputs["input_ids"], inputs["attention_mask"]) 
        return pooled.squeeze().cpu()
    except Exception as e:
        print(f"[ERROR] Text failed: {text[:30]}... ({e})")
        return None

def process_single_sample(
    audio_path,
    text,
    label,
    confidence=None,
    skip_text=False,
    nrc_lexicon=None,
):
    audio_embed = extract_audio_features(audio_path, AUDIO_PROCESSOR, AUDIO_MODEL, device)
    if audio_embed is None:
        return None

    if skip_text:
        text_embed = torch.zeros_like(audio_embed)
    else:
        text_embed = extract_text_features(text, TOKENIZER, TEXT_MODEL, device)
        if text_embed is None:
            return None

    sample = {
        "text_embed": text_embed,
        "audio_embed": audio_embed,
        "label": label,
        "confidence": confidence,
        "sample_id": os.path.basename(audio_path),
        "raw_text": text,
        "audio_path": audio_path
    }

    if nrc_lexicon is not None:
        nrc_embed, nrc_feature_names = extract_nrc_features(text, nrc_lexicon)
        sample["nrc_embed"] = nrc_embed
        sample["nrc_feature_names"] = nrc_feature_names

    return sample


def process_dataset(
    pkl_path,
    wav_base,
    output_path,
    skip_text=False,
    nrc_lexicon=None,
):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    processed_samples = []
    print(f"Processing {len(data)} samples from {pkl_path}")

    for item in tqdm(data, desc=f"Processing {os.path.basename(pkl_path)}"):
        if isinstance(item, dict):
            filename = (
                item.get("filename")
                or item.get("audio_path")
                or item.get("path")
                or item.get("file_path")
                or item.get("filepath")
                or item.get("wav_path")
            )
        elif isinstance(item, (tuple, list)):
            filename = item[0]
        else:
            raise TypeError(f"Unexpected item type: {type(item)}")

        if os.path.isabs(filename):
            audio_path = filename
        else:
            audio_path = os.path.join(wav_base, os.path.basename(filename))

        if isinstance(item, dict):
            text = item.get("text") or item.get("transcript") or item.get("utterance")
        else:
            text = item[1] if len(item) > 1 else ""

        label = None
        conf = None
        if isinstance(item, dict):
            label = item.get("label", item.get("emotion"))
            conf = item.get("confidence", None)
        elif isinstance(item, (tuple, list)) and len(item) > 2:
            label = item[2]

        sample = process_single_sample(
            audio_path,
            text,
            label,
            confidence=conf,
            skip_text=skip_text,
            nrc_lexicon=nrc_lexicon,
        )
        if sample is not None:
            processed_samples.append(sample)
        else:
            print(f"[SKIP] Failed to process: {audio_path}")

    with open(output_path, "wb") as f:
        pickle.dump(processed_samples, f)

    print(f"Saved processed data to: {output_path}")
    print(f"Total processed samples: {len(processed_samples)}")
    return processed_samples


def _write_manifest(out_dir, manifest):
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to: {manifest_path}")


def _process_client_dir(
    client_dir,
    out_dir,
    wav_base,
    skip_splits=None,
    nrc_lexicon=None,
    nrc_lexicon_path=None,
):
    os.makedirs(out_dir, exist_ok=True)
    skip_splits = set(skip_splits or [])
    manifest = {
        "client_dir": os.path.abspath(client_dir),
        "out_dir": os.path.abspath(out_dir),
        "splits": {},
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    split_specs = [
        ("train_labeled", "train_labeled.pkl", "train_labeled_features.pkl", False),
        ("val", "val.pkl", "val_features.pkl", False),
        ("test", "test.pkl", "test_features.pkl", False),
    ]

    for split_name, input_name, output_name, _ in split_specs:
        if split_name in skip_splits:
            continue
        input_path = os.path.join(client_dir, input_name)
        if not os.path.isfile(input_path):
            print(f"[WARN] Missing {split_name} input at {input_path}; skipping.")
            continue
        output_path = os.path.join(out_dir, output_name)
        processed = process_dataset(
            input_path,
            wav_base,
            output_path,
            skip_text=False,
            nrc_lexicon=nrc_lexicon,
        )
        sample_count = len(processed)
        text_dim = None
        audio_dim = None
        if sample_count > 0:
            text_dim = list(processed[0]["text_embed"].shape)
            audio_dim = list(processed[0]["audio_embed"].shape)
        nrc_dim = None
        if sample_count > 0 and processed[0].get("nrc_embed") is not None:
            nrc_dim = list(processed[0]["nrc_embed"].shape)
        manifest["splits"][split_name] = {
            "input_path": os.path.abspath(input_path),
            "output_path": os.path.abspath(output_path),
            "count": sample_count,
            "text_dim": text_dim,
            "audio_dim": audio_dim,
            "nrc_dim": nrc_dim,
        }
    if nrc_lexicon_path:
        manifest["nrc_lexicon"] = os.path.abspath(nrc_lexicon_path)
        manifest["nrc_feature_names"] = [
            f"nrc_{emotion}" for emotion in NRC_EMOTIONS
        ] + ["nrc_matched_words", "nrc_total_words"]

    _write_manifest(out_dir, manifest)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["IEMOCAP", "MSP-IMPROV", "ESD", "MELD"],
        help="Dataset to process",
    )
    parser.add_argument("--wav_base", type=str, default=None, help="Root directory containing the waveform files for the dataset.")
    parser.add_argument("--client_dir", type=str, default=None, help="Directory containing per-client metadata PKLs.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for extracted features.")
    parser.add_argument("--skip_splits", type=str, default="",
                        help="Comma-separated client splits to skip, e.g. test.")
    parser.add_argument("--pkl_path", type=str, default=None,
                        help="Single PKL file to extract.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output feature PKL for --pkl_path.")
    parser.add_argument("--pkl_dir", type=str, default=None,
                        help="Directory containing train.pkl/val.pkl/test.pkl (non-client mode).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for extracted features (non-client mode).")
    parser.add_argument("--nrc_lexicon", type=str, default=None,
                        help="Optional NRC Emotion Lexicon TSV path.")
    args = parser.parse_args()
    nrc_lexicon = load_nrc_lexicon(args.nrc_lexicon)

    if args.pkl_path:
        if args.wav_base is None:
            raise ValueError("Please provide --wav_base to specify the audio root directory.")
        if not args.output_path:
            raise ValueError("Please provide --output_path for --pkl_path.")
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        print("Starting single PKL feature extraction...")
        print(f"Using device: {device}")
        process_dataset(
            args.pkl_path,
            args.wav_base,
            args.output_path,
            skip_text=False,
            nrc_lexicon=nrc_lexicon,
        )
        return

    if args.client_dir:
        if args.wav_base is None:
            raise ValueError("Please provide --wav_base to specify the audio root directory.")
        out_dir = args.out_dir or os.path.join(
            OUTPUT_DIR, "clients", os.path.basename(args.client_dir.rstrip("/"))
        )
        print("Starting client feature extraction...")
        print(f"Using device: {device}")
        skip_splits = [item.strip() for item in args.skip_splits.split(",") if item.strip()]
        _process_client_dir(
            args.client_dir,
            out_dir,
            args.wav_base,
            skip_splits=skip_splits,
            nrc_lexicon=nrc_lexicon,
            nrc_lexicon_path=args.nrc_lexicon,
        )
        return

    output_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    print("Starting feature extraction...")
    print(f"Using device: {device}")

    datasets = []
    if args.dataset == "IEMOCAP":
        pkl_prefix = "IEMOCAP"
        if args.pkl_dir:
            datasets = [
                ("train", os.path.join(args.pkl_dir, "train.pkl"), f"{pkl_prefix}_BERT_Wav2Vec2_train.pkl", False),
                ("val",   os.path.join(args.pkl_dir, "val.pkl"),   f"{pkl_prefix}_BERT_Wav2Vec2_val.pkl", False),
                ("test",  os.path.join(args.pkl_dir, "test.pkl"),  f"{pkl_prefix}_BERT_Wav2Vec2_test.pkl", False),
            ]
        else:
            datasets = [
                ("train", f"{pkl_prefix}_preprocessed/train.pkl", f"{pkl_prefix}_BERT_Wav2Vec2_train.pkl", False),
                ("val",   f"{pkl_prefix}_preprocessed/val.pkl",   f"{pkl_prefix}_BERT_Wav2Vec2_val.pkl", False),
                ("test",  f"{pkl_prefix}_preprocessed/test.pkl",  f"{pkl_prefix}_BERT_Wav2Vec2_test.pkl", False),
            ]
    elif args.dataset == "MSP-IMPROV":
        pkl_prefix = "MSPIMPROV"
        if args.pkl_dir:
            datasets = [
                ("train", os.path.join(args.pkl_dir, "train.pkl"), f"{pkl_prefix}_BERT_Wav2Vec2_train.pkl", False),
                ("val",   os.path.join(args.pkl_dir, "val.pkl"),   f"{pkl_prefix}_BERT_Wav2Vec2_val.pkl", False),
                ("test",  os.path.join(args.pkl_dir, "test.pkl"),  f"{pkl_prefix}_BERT_Wav2Vec2_test.pkl", False),
            ]
        else:
            raise ValueError("Please provide --pkl_dir for MSP-IMPROV feature extraction.")
    elif args.dataset == "ESD":
        pkl_prefix = "ESD"
        if args.pkl_dir:
            datasets = [
                ("train", os.path.join(args.pkl_dir, "train.pkl"), f"{pkl_prefix}_BERT_Wav2Vec2_train.pkl", False),
                ("val",   os.path.join(args.pkl_dir, "val.pkl"),   f"{pkl_prefix}_BERT_Wav2Vec2_val.pkl", False),
                ("test",  os.path.join(args.pkl_dir, "test.pkl"),  f"{pkl_prefix}_BERT_Wav2Vec2_test.pkl", False),
            ]
        else:
            datasets = [
                ("train", f"{pkl_prefix}_preprocessed/train.pkl", f"{pkl_prefix}_BERT_Wav2Vec2_train.pkl", False),
                ("val",   f"{pkl_prefix}_preprocessed/val.pkl",   f"{pkl_prefix}_BERT_Wav2Vec2_val.pkl", False),
                ("test",  f"{pkl_prefix}_preprocessed/test.pkl",  f"{pkl_prefix}_BERT_Wav2Vec2_test.pkl", False),
            ]
    elif args.dataset == "MELD":
        pkl_prefix = "MELD"
        if args.pkl_dir:
            datasets = [
                ("train", os.path.join(args.pkl_dir, "train.pkl"), f"{pkl_prefix}_BERT_Wav2Vec2_train.pkl", False),
                ("val",   os.path.join(args.pkl_dir, "val.pkl"),   f"{pkl_prefix}_BERT_Wav2Vec2_val.pkl", False),
                ("test",  os.path.join(args.pkl_dir, "test.pkl"),  f"{pkl_prefix}_BERT_Wav2Vec2_test.pkl", False),
            ]
        else:
            raise ValueError("Please provide --pkl_dir for MELD feature extraction.")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    for split_name, pkl_file, output_file, skip_text in datasets:
        print(f"\n{'='*50}")
        print(f"Processing {split_name} split: {pkl_file}")
        print(f"{'='*50}")
        wav_base = args.wav_base
        if wav_base is None:
            raise ValueError("Please provide --wav_base to specify the audio root directory.")
        pkl_path = pkl_file if args.pkl_dir else os.path.join(PKL_DIR, pkl_file)
        output_path = os.path.join(output_dir, output_file)
        process_dataset(
            pkl_path,
            wav_base,
            output_path,
            skip_text=skip_text,
            nrc_lexicon=nrc_lexicon,
        )

if __name__ == "__main__":
    main()
