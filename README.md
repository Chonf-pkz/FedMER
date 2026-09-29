# FedMER: A Federated Benchmark for Multimodal Emotion Recognition under Speaker-Level Data Heterogeneity

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c.svg)

<i>
  Official code repository for the manuscript 
  <b>"A Federated Benchmark for Multimodal Emotion Recognition under Speaker-Level Data Heterogeneity"</b>, 
  accepted at
  <a href="https://csonet-conf.github.io/csonet26/index.html">The 15th International Conference on Computational Science and Network Intelligence.</a>.
</i>

> **Thu Ky Nguyen†, Nhat Khanh Le†, Nhut Minh Nguyen†, Ngoc-Khoa Le, Duc Ngoc Minh Dang\***
> AiTA Lab, Faculty of Information Technology, FPT University, Ho Chi Minh City, Vietnam
> † Equal contribution · \* Corresponding author

---

## Overview

Most MER work assumes centralized access to raw speech and transcripts, which can reveal speaker identity and psychological state. FL lets clients train on their own data and share only model updates, but little is known about how different MER architectures behave under FL.

FedMER provides:

- **One pipeline for every model.** All models share the same feature extraction, speaker-shard non-IID client simulation, federated training loop and evaluation protocol. Differences in results therefore come from the fusion design, not from the inputs.
- **4 MER architectures × 5 FL strategies** on **IEMOCAP** and **MSP-IMPROV**.
- **Centralized upper bounds** for every architecture, so you can measure the federated-to-centralized gap.
- **4 metrics:** Weighted Accuracy (WA), Unweighted Accuracy (UA), Weighted F1 (WF1) and Macro F1 (MF1), averaged over 5 seeds.

### MER architectures

| Model | Fusion mechanism | Inter-modal interaction | Key idea |
|---|---|---|---|
| **3M-SER** | Concatenation + multi-head attention | Implicit, post-concatenation | Concatenates audio and text embeddings, then applies attention to reweight feature dimensions. |
| **MemoCMT** | Cross-modal transformer | Explicit, multi-layer | Audio and text representations attend to each other across transformer layers. |
| **CemoBAM** | Graph attention network | Explicit, graph-structured | Represents audio-text features as graph nodes to capture irregular cross-modal dependencies. |
| **FleSER** | Fuzzy membership + attention | Adaptive, uncertainty-weighted | Estimates each modality's contribution from input uncertainty before attention-based fusion. |

### FL strategies

| Strategy | Heterogeneity targeted | Mechanism |
|---|---|---|
| **FedAvg** | Baseline | Weighted parameter averaging |
| **FedProx** | Client drift | Proximal regularization toward the global model |
| **FedNova** | Objective inconsistency | Normalizes updates by the number of local steps |
| **FedProc** | Representation drift | Global class-prototype alignment |
| **FedBN** | Feature heterogeneity | Keeps Batch Normalization parameters local |

> The code also includes FedDC (`--fl_method feddc`) and an extra model (`fedalmer`). These are not part of the paper's benchmark.

### Speaker-shard non-IID protocol

1. Hold out one session as the shared test set: **Session 5** for IEMOCAP, **Session 6** for MSP-IMPROV.
2. Group the remaining utterances by speaker and split each speaker into **4 shards**, giving **K = 4S clients** (S = number of training speakers). This yields **32 clients** for IEMOCAP and **40** for MSP-IMPROV.
3. Each client gets data from **exactly one speaker** and **exactly three of the four emotion classes**. Within each (speaker, emotion) pair, utterances are shuffled with the experiment seed and dealt round-robin across the three eligible shards, so no sample is dropped or duplicated.
4. Each client's data is split by class into **80% local training / 20% validation**.

---

## Repository structure

```
FedMER/
├── feature_extract/          # Offline Wav2Vec2 / BERT feature extraction
│   └── extract_feature.py
├── federated/                # Federated benchmark
│   ├── configs/              # YAML experiment configs
│   ├── clients/              # Client-side updates (FedAvg, FedProx, FedProc, FedDC)
│   ├── servers/              # Server-side aggregation (FedAvg, FedProx, FedNova, FedProc, FedBN, FedDC)
│   ├── preprocess.py         # Client partitioning
│   ├── loso_runner.py        # End-to-end pipeline: partition → features → train → eval
│   ├── run_federated.py      # Federated training on existing client features
│   ├── evaluate.py           # Evaluate a global checkpoint
│   └── generate_noisy_test_features.py   # Optional noisy-test robustness features
├── centralized/              # Centralized reference training
│   ├── configs/
│   ├── loso_runner.py
│   ├── train.py
│   └── predict.py
├── src/
│   ├── baseline/             # CemoBAM, FleSER, MemoCMT, ThreeMSER (3M-SER)
│   ├── feature_extract/      # Encoder configuration
│   └── model_factory.py      # Model registry
├── metadata/                 # Generated client partitions (PKL + client_map.json)
├── features/                 # Extracted client features
├── checkpoints/              # Saved models
├── logs/                     # Training / evaluation logs and metrics
└── tests/
```

---

## Installation

```bash
git clone https://github.com/Chonf-pkz/FedMER.git
cd FedMER

conda create -n fedmer python=3.10 -y
conda activate fedmer

# Install the PyTorch build that matches your CUDA version first (see pytorch.org), then:
pip install -r requirements.txt
```

The frozen encoders `bert-base-uncased` and `facebook/wav2vec2-base-960h` are downloaded automatically from Hugging Face the first time you extract features.

## Datasets

Both datasets are licensed and have to be requested from their owners:

- **IEMOCAP**: <https://sail.usc.edu/iemocap/> (4 classes: anger 1,103 · happiness 1,636 · neutral 1,708 · sadness 1,084)
- **MSP-IMPROV**: <https://ecs.utdallas.edu/research/researchgroups/msp-lab/databases/MSP-Improv.html> (4 classes: anger 2,511 · happiness 2,267 · neutral 1,625 · sadness 2,035)

Point `paths.data_root` in the config (or `--data_root`) at the extracted dataset directory, for example `IEMOCAP_full_release/`.

---

## Usage

### 1. Full federated pipeline (recommended)

`federated/loso_runner.py` builds the speaker-shard clients, extracts features, trains and evaluates every seed in one run:

```bash
python federated/loso_runner.py \
    --config federated/configs/iemocap.yaml \
    --data_root /path/to/IEMOCAP_full_release \
    --model_name cemobam \
    --fl_method fedavg
```

- `--model_name`: `cemobam`, `fleser`, `memocmt`, `threemser`
- `--fl_method`: `fedavg`, `fedprox`, `fednova`, `fedproc`, `fedbn`
- For MSP-IMPROV, use `--dataset MSP-IMPROV` with a config whose `test_sessions` is `[6]`. `federated/configs/msp_improv_fedproc_corrected.yaml` is an example.
- If metadata or features already exist, skip those stages with `--skip_preprocess`, `--skip_features`, `--skip_train` or `--skip_eval`.

Any command-line flag overrides the matching value in the YAML file.

### 2. Federated training on existing features

```bash
python federated/run_federated.py \
    --dataset IEMOCAP --num_classes 4 \
    --model_name fleser --fl_method fedprox --fedprox_mu 1.0 \
    --clients_root metadata/IEMOCAP_loso/clients/<run_id>/test_session5_speaker_shards/seed42 \
    --features_root features/IEMOCAP/clients/<run_id>/test_session5_speaker_shards/seed42 \
    --rounds_pretrain 50 --local_epochs_pretrain 5 --batch_size 32
```

### 3. Evaluate a checkpoint

```bash
python federated/evaluate.py \
    --dataset IEMOCAP --num_classes 4 --model_name fleser \
    --clients_root <clients_root> --features_root <features_root> \
    --checkpoint checkpoints/federated/<exp_name>/.../global_round_best_<round>.pt
```

The best round for each run is recorded in `best_meta.pt`, saved next to the checkpoint.

### 4. Centralized reference

```bash
python centralized/loso_runner.py --config centralized/configs/iemocap.yaml --model_name cemobam
```

Metrics are written to `logs/federated/<exp_name>/` (for example `eval/pretrain/best/aggregate_metrics.json`) and checkpoints to `checkpoints/federated/`.

### Default hyperparameters (paper setting)

| Setting | Value |
|---|---|
| Communication rounds | 50 |
| Local epochs per round | 5 |
| Batch size | 32 |
| Optimizer lr / weight decay | 1e-4 / 1e-2 |
| FedProx μ | 1.0 |
| FedProc λ / temperature | 1.0 / 0.5 |
| Seeds | 42, 52, 103, 128, 923 |
| Hardware | 1 × NVIDIA A30 |

---

## Citation

If you use FedMER, please cite:

```bibtex
@inproceedings{nguyen2026fedmer,
  title  = {A Federated Benchmark for Multimodal Emotion Recognition under Speaker-Level Data Heterogeneity},
  author = {Nguyen, Thu Ky and Le, Nhat Khanh and Nguyen, Nhut Minh and Le, Ngoc-Khoa and Dang, Duc Ngoc Minh},
  booktitle = {Proceedings of the 15th International Conference on Computational Science and Network Intelligence.},
  year   = {2026},
  month ={November},
  venue ={Ho Chi Minh City, Vietnam},
  note   = {To appear}
}
```

This benchmark builds on the MER baselines **3M-SER** (Tran et al., INISCOM 2023), **MemoCMT** (Khan et al., Scientific Reports 2025), **CemoBAM** (Nguyen et al., APNOMS 2025) and **FleSER** (Nguyen et al., EAAI 2025). Please cite them as well.

## License

Released under the [MIT License](LICENSE).

## Contact

For questions, please open an issue or contact the corresponding author, Duc Ngoc Minh Dang (ducdnm2@fe.edu.vn).
