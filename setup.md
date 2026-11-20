# Setup Guide — QML Galaxy Classification

This guide describes how to set up and run the code in this repository, as it currently exists.
It documents what is required to execute each script; it does not change or re-interpret the
project's reported results (see `README.md` for those).

---

## 1. Project Overview

The repository classifies galaxy images from the Kaggle Galaxy Zoo dataset into three classes
(`Elliptical`, `Spiral`, `Irregular`) and compares a classical model with a hybrid
quantum-classical model under injected Gaussian noise.

Current pipeline (all scripts live in `src/`):

| Stage | Script | What it does |
|---|---|---|
| Dataset labels | `build_dataset_csv.py` | Thresholds Galaxy Zoo vote fractions into 3 classes; writes `master_labels_60k.csv` (labeled) and `unclassified_backlog.csv` (everything else). |
| Classical training | `train_saan.py` (+ `build_saan.py`) | Trains the SAAN model: `LearnableNoiseGate` → frozen ImageNet backbone (EfficientNetV2B0 / DenseNet121 / ConvNeXtTiny) → `SEBlock` → dense head. 64×64 RGB input. |
| Pseudo-labeling | `classify_backlog.py` | Runs a trained SAAN model over `unclassified_backlog.csv`; writes `pseudo_labeled_backlog.csv`. |
| Hybrid training | `train_hybrid_qml.py` | Freezes the trained SAAN-DenseNet121, adds `Dense(4, tanh)` → 4-qubit PennyLane circuit (`AngleEmbedding` + 2 × `BasicEntanglerLayers`, simulated on `default.qubit`) → softmax. |
| Noise stress tests | `stress_test.py`, `stress_test_qml.py` | Evaluate the classical / hybrid model on 2,000 validation images, clean and at SNR 10, 1, 0.1. |
| Grad-CAM | `visualize_gradcam.py`, `visualize_gradcam_qml.py` | Produce the Grad-CAM figures in the repository root. |

Legacy scripts from an earlier project phase (a 6,000-image pre-processed dataset and a 6-qubit
Qiskit VQC) are also in `src/`: `preprocess.py`, `train_cnn.py`, `prepare_quantum_data.py`,
`train_qnn.py`, `noise_showdown.py`, `benchmark_noise.py`, `grad_cam.py`. See §7.6.

`exoplanet_src/` contains four empty (0-byte) placeholder files and has nothing to run.

---

## 2. System Requirements

| Item | Requirement |
|---|---|
| OS | **Linux x86_64** for the pinned `requirements.txt` (captured on Linux; includes Linux-only `nvidia-*-cu12` wheels). The original project runs were made on **Windows** (see the `C:\...` paths in the training logs), using an older TensorFlow build. With TensorFlow 2.21, GPU use on Windows requires WSL2. |
| Python | **3.10** (environment captured with Python 3.10.20). |
| GPU | NVIDIA GPU with CUDA support. **Required** by `train_saan.py` (lines 35–37) and `classify_backlog.py` (lines 8–10), which raise `RuntimeError` if no GPU is visible. The other scripts use a GPU if present and otherwise run on CPU. |
| Reference hardware | Original runs: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM (`training_logs/classical_training_progress.txt`, line 5; `training_logs/logs_cnn_main_2026-04-16.txt`). |
| CUDA | Provided by the pip `nvidia-*-cu12` packages pulled in by `tensorflow[and-cuda]`; only an NVIDIA driver must be installed system-wide. |
| Data | The 61,578 Galaxy Zoo training JPEGs (35,814 labeled + 25,764 backlog, per the committed CSVs) must be stored locally. |
| Run time (reference) | From the committed logs on the RTX 3050: SAAN-DenseNet121 ≈ 40 s/epoch; hybrid QML ≈ 900 s/epoch (7 epochs ≈ 1.75 h). |

---

## 3. Environment Setup

The project environment is a conda environment named `galaxy_gpu` (the same name appears in
`training_logs/logs_cnn_main_2026-04-16.txt`).

```bash
# 1. Create and activate the environment
conda create -n galaxy_gpu python=3.10 -y
conda activate galaxy_gpu

# 2. Install the pinned dependencies (from the repository root)
pip install -r requirements.txt

# 3. Verify
python --version          # expect Python 3.10.x
pip check
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
python -c "import pennylane as qml; print(qml.__version__)"
```

A plain `venv` works equally well:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key pinned packages: `tensorflow==2.21.0`, `keras==3.12.3`, `numpy==2.2.6`, `pandas==2.2.3`,
`scikit-learn==1.7.2`, `opencv-python==5.0.0.93`, `matplotlib==3.10.9`, `pennylane==0.42.3`,
`qiskit==2.5.0`.

---

## 4. Dataset Setup

The dataset is **not included** in the repository (`data/` is listed in `.gitignore`).

- **Source:** Kaggle *Galaxy Zoo – The Galaxy Challenge* (training data). Download it from Kaggle
  (a Kaggle account is required).
- **Required files:**
  - `training_solutions_rev1.csv`: per-galaxy vote fractions (`GalaxyID`, `Class1.1`, …).
  - `images_training_rev1/`: `<GalaxyID>.jpg` images (e.g. `100023.jpg`).

Expected layout (any location; you point the scripts at it, see §5):

```text
<DATA_ROOT>/
├── training_solutions_rev1.csv
└── images_training_rev1/
    ├── 100008.jpg
    ├── 100023.jpg
    └── ...
```

**Committed label files** (repository root): you do not need to regenerate these.

| File | Rows | Columns |
|---|---|---|
| `master_labels_60k.csv` | 35,814 | `filename`, `Label` (Elliptical 19,942 · Spiral 10,338 · Irregular 5,534) |
| `unclassified_backlog.csv` | 25,764 | `filename` |
| `pseudo_labeled_backlog.csv` | 25,764 | `filename`, `Predicted_Label`, `Confidence_Score` |

The scripts join `filename` with the image directory and silently drop rows whose image file
does not exist (`df[df['filepath'].apply(os.path.exists)]`).

**Legacy pipeline layout** (only for the §7.6 scripts), relative to the repository root:

```text
data/raw_images/<GalaxyID>.jpg
data/training_solutions_rev1.csv
data/processed_images/{Elliptical,Spiral,Irregular}/   # created by src/preprocess.py
data/quantum/processed_data.pkl                         # created by src/prepare_quantum_data.py
```

---

## 5. Path Configuration

**All scripts must be run from inside `src/`**: they use paths relative to it
(`../master_labels_60k.csv`, `../models/...`, `../data/...`).

The current code contains the original **Windows** paths from the machine the project was
developed on. These are historical locations, not a requirement; edit them to point at your
`<DATA_ROOT>`:

| File | Line | Hard-coded value |
|---|---|---|
| `src/build_dataset_csv.py` | 6 | `C:\galaxy_datasets\training_solutions_rev1.csv` |
| `src/build_dataset_csv.py` | 30, 35 | `C:\Projects\QML_Galaxy_Project\master_labels_60k.csv`, `...\unclassified_backlog.csv` (outputs) |
| `src/train_saan.py` | 52 | `EXTERNAL_IMAGES_DIR = C:\galaxy_datasets\images_training_rev1` |
| `src/classify_backlog.py` | 20 | `IMAGES_DIR = C:\galaxy_datasets\images_training_rev1` |
| `src/train_hybrid_qml.py` | 40 | `IMAGES_DIR = C:\galaxy_datasets\images_training_rev1` |
| `src/stress_test.py` | 24 | `IMAGES_DIR = C:\galaxy_datasets\images_training_rev1` |
| `src/stress_test_qml.py` | 70 | `IMAGES_DIR = C:\galaxy_datasets\images_training_rev1` |
| `src/visualize_gradcam.py` | 76 | `SAMPLE_IMAGE_PATH = C:\galaxy_datasets\images_training_rev1\102243.jpg` |
| `src/visualize_gradcam_qml.py` | 118 | `SAMPLE_IMAGE_PATH = C:\galaxy_datasets\images_training_rev1\102243.jpg` |

Other settings that are edited in source (there are no command-line arguments):

- `src/train_saan.py` line 97: `TARGET_MODEL = 'efficientnet'` (options: `'efficientnet'`,
  `'densenet'`, `'convnext'`). The hybrid model, classical stress test and classical Grad-CAM
  all load the **DenseNet** checkpoint, so set it to `'densenet'` to produce that file.

---

## 6. Model / Checkpoint Requirements

**No model checkpoints are included in the repository** (`models/` is in `.gitignore`).
Every evaluation, stress-test and visualization script needs a checkpoint that must first be
produced by training.

| Checkpoint (`models/…`) | Produced by | Required by |
|---|---|---|
| `saan_<TARGET_MODEL>_best.keras` | `train_saan.py` (line 110) | — |
| `saan_densenet_best.keras` | `train_saan.py` with `TARGET_MODEL='densenet'` | `train_hybrid_qml.py` (121), `stress_test.py` (16), `visualize_gradcam.py` (16) |
| `saan_hybrid_qml_best.keras` | `train_hybrid_qml.py` (147) | `stress_test_qml.py` (58), `visualize_gradcam_qml.py` (65) |
| `saan_best_model_60k.keras` | **No current script.** It was produced by the earlier scratch-CNN SAAN (`build_saan_model`, now commented out in `build_saan.py` lines 58–88). | `classify_backlog.py` (48) |
| `saan_best_model.keras` | **No current script** (legacy phase). | `benchmark_noise.py` (123), `grad_cam.py` (11) |
| `galaxy_cnn_baseline.keras` | `train_cnn.py` (77) | — |
| `vqc_weights.npy` | `train_qnn.py` (54) | `noise_showdown.py` (18) |

The `models/` directory is created automatically only by `train_cnn.py` and `train_qnn.py`.
Before running `train_saan.py` or `train_hybrid_qml.py`, create it:

```bash
mkdir -p models
```

---

## 7. Execution Instructions

Run everything from `src/` with the environment activated:

```bash
conda activate galaxy_gpu
cd src
```

> **Note — scripts that overwrite committed files.** Re-running these replaces artifacts that
> are tracked in Git:
> - `train_hybrid_qml.py` → `../logs_hybrid_qml_2026.txt` (opened in `"w"` mode)
> - `classify_backlog.py` → `../pseudo_labeled_backlog.csv`
> - `visualize_gradcam.py` → `../gradcam_collapse_proof_with_preds.png`
> - `visualize_gradcam_qml.py` → `../gradcam_qml_immunity_proof.png`
>
> `train_saan.py` writes a new log `../logs_densenet_<timestamp>.txt` (this prefix is used for
> every backbone).

### 7.1 Dataset preparation (optional: the CSVs are already committed)

```bash
python build_dataset_csv.py
```

This requires the paths in lines 6, 30 and 35 to be set (§5).

### 7.2 Classical model training

```bash
python train_saan.py
```

- Stratified 80/20 train/validation split (`random_state=42`), 30 epochs max, early stopping
  (patience 5), balanced class weights, `mixed_float16`.
- Output: `../models/saan_<TARGET_MODEL>_best.keras`.

### 7.3 Pseudo-labeling the backlog

```bash
python classify_backlog.py
```

This requires `../models/saan_best_model_60k.keras`, which no current script produces (§6).

### 7.4 Hybrid quantum-classical training

```bash
python train_hybrid_qml.py
```

- Requires `../models/saan_densenet_best.keras`.
- Same split as 7.2; 10 epochs max, early stopping (patience 3).
- Output: `../models/saan_hybrid_qml_best.keras`.

### 7.5 Stress testing, evaluation and Grad-CAM

```bash
python stress_test.py            # classical SAAN-DenseNet121, clean / SNR 10 / 1 / 0.1
python stress_test_qml.py        # hybrid QML model, same 2,000 validation images
python visualize_gradcam.py      # → ../gradcam_collapse_proof_with_preds.png
python visualize_gradcam_qml.py  # → ../gradcam_qml_immunity_proof.png
```

- The stress tests print their results to the console only; they do not write a file.
- The Grad-CAM scripts open a Matplotlib window (`plt.show()`) after saving.
- The SNR noise uses unseeded `tf.random.normal`, so numbers vary slightly between runs.

### 7.6 Legacy scripts (earlier project phase)

These use `../data/` (§4) and are not part of the README pipeline:

```bash
python preprocess.py             # data/raw_images → data/processed_images (6,000 images)
python train_cnn.py              # → models/galaxy_cnn_baseline.keras
python prepare_quantum_data.py   # → data/quantum/processed_data.pkl (PCA, 6 components)
python train_qnn.py              # Qiskit VQC → models/vqc_weights.npy
python noise_showdown.py
python benchmark_noise.py        # needs models/saan_best_model.keras (not produced by any script)
python grad_cam.py               # needs models/saan_best_model.keras (not produced by any script)
```

`train_qnn.py` and `noise_showdown.py` additionally import `qiskit_algorithms` and
`qiskit_machine_learning`, which are **not installed** in the captured environment and are
therefore not in `requirements.txt`. To run them, install those packages separately; their
compatibility with `qiskit==2.5.0` has not been verified.

---

## 8. Reproducing Historical Results

| Result | Status with the current repository |
|---|---|
| Dataset split into 35,814 labeled / 25,764 backlog images | **Reproducible.** The committed CSVs contain these counts; regeneration needs the Kaggle CSV. |
| Pseudo-label distribution (E 10,845 / I 8,309 / S 6,610) | **Historically documented.** The committed CSV matches `training_logs/logs_cnn_main_2026-04-16.txt`; the checkpoint that produced it is absent and not producible by current code. |
| Classical training curves (scratch SAAN 81.28%, DenseNet121 81.59%, EfficientNetV2 75.46% best val. accuracy) | **Historically documented** in `training_logs/`. Re-running requires the image data and a GPU. The logs were produced on Windows with an older TensorFlow; `classical_training_progress.txt` §4 also records configuration changes (backbone freezing, input scaling) made between the DenseNet and EfficientNet runs. Identical numbers are therefore not guaranteed. |
| Hybrid QML training curve | **Historically documented** in `logs_hybrid_qml_2026.txt`. Re-trainable with `train_hybrid_qml.py` once `saan_densenet_best.keras` exists. |
| SNR stress-test accuracies quoted in `README.md` | **Historical; runtime output not present.** The scripts print to the console and no output log is committed. Re-running requires both checkpoints. |
| Grad-CAM figures | **Committed as images.** Regenerating them requires the checkpoints. `gradcam_collapse_proof.png` corresponds to an earlier version of `visualize_gradcam.py` (sample `100023.jpg`, different output filename). |
| Legacy-phase numbers (e.g. in `noise_showdown.py`) | **Historical; artifacts absent** (`data/processed_images`, `vqc_weights.npy`, legacy checkpoints). |

Seeds: data splits and stress-test sampling use `random_state=42`. TensorFlow weight
initialization, dropout, augmentation and noise are not seeded, so retrained models will differ
somewhat from the logged runs.

---

## 9. Verification

```bash
python --version                                    # Python 3.10.x
pip check                                           # no conflicts among installed packages
pip freeze | grep -iE "^(tensorflow|keras|pennylane|qiskit|numpy)=="
nvidia-smi                                          # NVIDIA driver and GPU visible
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"   # must be non-empty for train_saan.py
python -c "import pandas as pd; print(len(pd.read_csv('master_labels_60k.csv')))"     # 35814 (run from repo root)
```

Data-path check: `train_saan.py` prints `Valid images ready for training: N`. The original runs
report `35814`.

---

## 10. Troubleshooting

| Symptom | Cause | Resolution |
|---|---|---|
| `RuntimeError: ❌ NO GPU DETECTED!` | `train_saan.py` / `classify_backlog.py` require a GPU. | Check `nvidia-smi` and the TensorFlow GPU check in §9. |
| TensorFlow log: `Cannot dlopen some GPU libraries … Skipping registering GPU devices` | TensorFlow cannot load the CUDA/cuDNN libraries. This was observed in the captured environment, which also contains CUDA 13 wheels from other packages alongside the `-cu12` ones. | Use a fresh environment created only from `requirements.txt`, and confirm the NVIDIA driver supports CUDA 12. |
| `Valid images ready for training: 0` or a `train_test_split` error about empty input | The image directory path does not exist on this machine, so every row is dropped. | Set the paths listed in §5. |
| `FileNotFoundError` / `ValueError` loading `../models/*.keras` | Checkpoints are not in the repository. | Train them first (§6, §7); create `models/` with `mkdir -p models`. |
| `FileNotFoundError: ../master_labels_60k.csv` | Script run from the repository root instead of `src/`. | `cd src` first. |
| `ModuleNotFoundError: qiskit_algorithms` / `qiskit_machine_learning` | Legacy Qiskit scripts only; packages not in `requirements.txt`. | Install separately (§7.6). |
| `ModuleNotFoundError: pennylane` | Environment not created from the current `requirements.txt`. | `pip install -r requirements.txt`. |
| Errors loading a checkpoint saved by the original Windows setup | Original runs used an older TensorFlow/Keras 2; this environment uses Keras 3. | Retrain in the current environment. |
| Out-of-memory on GPU | Reference GPU had 4 GB VRAM; batch size is 64. | Reduce `BATCH_SIZE` (`train_saan.py` line 70; `train_hybrid_qml.py` uses `.batch(64)` at lines 58–59). |
| No window appears / script blocks at the end | Grad-CAM scripts call `plt.show()`. | The PNG is already saved; close the window, or run with `MPLBACKEND=Agg`. |
