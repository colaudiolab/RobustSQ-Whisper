# RobustSQ-Whisper

**RobustSQ-Whisper** is a lightweight, plug-in adaptation of **SQ-Whisper** for **target-speaker ASR (TS-ASR)** under **overlapped/noisy enrollment**. It strengthens the enrollment branch with:

- **Attentive Statistics Pooling (ASP)** — query-conditioned, frame-weighted mean and standard deviation to suppress interference and model uncertainty.
- **Margin-based objectives** — **Arc-InfoNCE** (pairwise angular margin) and a **parallel AAM-Softmax** head (class-center angular margin) that enlarge inter-class separation and tighten intra-class variance.

> **No backbone changes.** **No extra inference-time cost.** No separators or external speaker models required.

------

## Features

- Robust to **overlapped and noisy enrollment**
- **Plug-in** modules: easy to reuse with existing SQ-Whisper/TS-ASR pipelines
- **Reproducible** and **low-cost**: training-time only; inference unchanged
- Works with **Libri2Mix** (clean/noisy; Train-100/Train-360/Dev/Test)