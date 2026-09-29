import os
import sys
import torch
import pickle
import torchaudio
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import BertTokenizer, Wav2Vec2Processor
from src.feature_extract.config import (
    PKL_DIR, OUTPUT_DIR, device,
    TOKENIZER, AUDIO_PROCESSOR,
    TEXT_MODEL, AUDIO_MODEL
)
from src.utils.utils import get_model_stats
import torch

# ==== BERT ====
text = "Hello, this is a test for FLOPs and inference time."
encoded = TOKENIZER(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
sample_text_input = (encoded["input_ids"].to(device), encoded["attention_mask"].to(device))

bert_stats = get_model_stats(TEXT_MODEL, sample_text_input, device)
print("\nBERT Stats:", bert_stats)

# ==== Wav2Vec2 ====
import torchaudio
waveform = torch.randn(1, 16000 * 3)  # 3 sec fake audio
sample_audio_input = (waveform.to(device),)

wav2vec_stats = get_model_stats(AUDIO_MODEL, sample_audio_input, device)
print("\nWav2Vec2 Stats:", wav2vec_stats)
