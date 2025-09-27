# RobustSQ-Whisper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)

This is the official repository for the paper **"Attentive Statistics Pooling and Margin-Based Contrastive Learning for Target-Speaker ASR with Overlapped Enrollment"** submitted to **ICASSP 2026**.

## 📋 Abstract

**RobustSQ-Whisper** is a lightweight, plug-in adaptation of **SQ-Whisper** for **target-speaker ASR (TS-ASR)** under **overlapped/noisy enrollment**. It strengthens the enrollment branch with:

## 🚀 Key Features

- **Attentive Statistics Pooling (ASP)** — query-conditioned, frame-weighted mean and standard deviation to suppress interference and model uncertainty.
- **Margin-based objectives** — **Arc-InfoNCE** (pairwise angular margin) and a **parallel AAM-Softmax** head (class-center angular margin) that enlarge inter-class separation and tighten intra-class variance.

> **No backbone changes.** **No extra inference-time cost.** No separators or external speaker models required.

## 🏗️ Architecture

Our framework includes the following components for robust TS-ASR:

1. **ASP**: Attentive Statistics Pooling for speaker-aware sequence summarization
2. **SQ-Former**: Query-style conditioning to inject target speaker cues
3. **AAM-Softmax**: Angular margin classification head improving inter-/intra-class geometry
4. **Arc-InfoNCE**: Contrastive objective with angular margin against interferers and hard negatives

## 📊 Supported Backbones

| Category    | Backbone                     | Notes                                                |
| ----------- | ---------------------------- | ---------------------------------------------------- |
| **Whisper** | tiny / base / small / medium | Encoder-decoder backbone used by our TS-ASR pipeline |

## 📁 Repository Structure

 This recipe follows ESPnet’s `egs2/librimix` style. The tree below mirrors the main folders and scripts for TS-Whisper.

```
tgt_asr1/
├── run_tswhisper.sh                    # Main TS-Whisper pipeline (train/decode)
├── asr_my.sh                           # Custom ASR baseline/ablation script
├── check_training_status.sh            # Monitor training/decoding status
├── cmd.sh                              # Command launcher settings (local/queue)
├── db.sh                               # Dataset paths (LibriSpeech/WHAM/Libri2Mix)
├── path.sh                             # Environment setup
├── gradscaler_fix.py                   # AMP/GradScaler patch (if needed)
├── conf/
│   ├── fbank.conf                      # Feature configs
│   ├── pitch.conf
│   ├── queue.conf / pbs.conf / slurm.conf
│   └── tswhisper/
│       ├── train_tsasr_whisper_medium_full_con20_q16_l2_crop10_lr5e-5.yaml    #Train config
│       ├── decode_asr_whisper_beam1.yaml
│       ├── train_tsasr_whisper_medium_lora_qkvo_r16_.yaml
│       └── train_tsasr_whisper_medium_masking_.yaml
├── datapre/
│   ├── data_prep.sh                    # Bootstrap manifests/folders
│   ├── data.sh
│   ├── format_sglspk_dataset.py        # Build single-speaker lists
│   ├── create_enrollment_.py           # Enrollment list creation
│   ├── create_overlap_.py              # Overlap mixing (SIR)
│   └── add_wham_noise.py               # Add WHAM! noise (SNR)
├── dump/
│   ├── raw/{train,dev,test}_sglspk/    # Raw manifests
│   └── {train,dev,test}_sglspk/        # Standardized Kaldi/ESPnet files:
│       ├── wav.scp  text  utt2spk  spk2utt
│       ├── enroll.scp  resnet.scp (optional)
│       └── feats_type  utt2num_samples
├── embedding/                          # Offline speaker embeddings
├── pretrain_model/
│   ├── voxceleb_resnet34_LM.onnx       # Speaker embedding model (optional)
│   └── whisper/                        # Whisper weights
├── exp/                                # Experiments, logs, checkpoints
├── parallel/                           # pbs.pl / slurm.pl / run.pl / retry.pl
├── steps/                              # Kaldi-style helper steps
└── utils/                              # Data utilities (combine/filter/split/etc.)
```

## 📚 Datasets

We evaluate on mixtures derived from **LibriSpeech** with **WHAM!** noise via **Libri2Mix**.  
Please follow original licenses when downloading and using datasets.

- **Libri2Mix**: https://github.com/JorisCos/LibriMix  
- **LibriSpeech**: https://www.openslr.org/12  
- **WHAM!**: https://wham.whisper.ai/

## 🚧 Status

**⚠️ Code and configuration are being curated for camera-ready release. Full training/evaluation scripts and all configs will be provided after acceptance.**

## 🙏 Acknowledgments

We thank the following open-source projects and prior works:

- **ESPnet** — End-to-end speech processing toolkit  
- **Libri2Mix / LibriSpeech / WHAM!** — Datasets used in our experiments  
- Community implementations of **AAM-Softmax**, **InfoNCE/ArcFace-style** losses, and **Attentive Statistics Pooling**

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.  
Datasets and pretrained models follow their respective licenses.

## 📞 Contact

For questions and inquiries, please contact **[scyang]** at **scyang0108@163.com**.

---

**Note**: This repository is under active development. Please check back for updates after **ICASSP 2026** acceptance.

