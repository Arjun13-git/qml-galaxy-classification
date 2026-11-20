# Quantum Machine Learning for Galaxy Classification 🌌⚛️
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-tf.keras-red.svg)
![PennyLane](https://img.shields.io/badge/PennyLane-QML-purple.svg)
![Qiskit](https://img.shields.io/badge/Qiskit-VQC-6929C4.svg)
![CUDA](https://img.shields.io/badge/NVIDIA-CUDA%20%7C%20cuDNN-green.svg)

A research pipeline that classifies Galaxy Zoo galaxy images into **Elliptical**, **Spiral** and **Irregular** morphologies with a custom attention-based classical network (**SAAN**), then splices that network into a **4-qubit Hybrid Quantum-Classical Neural Network** and compares both under an SNR (signal-to-noise ratio) degradation stress test and Grad-CAM visual analysis.

**Institution:** Sahyadri College of Engineering & Management, Mangaluru
**Section:** 6A CSE
**Team Number:** CS055
**Team Members:** Ajith Goveas, Arjun Shenoy R, Ashish Shenoy K, Krrish Raj
**Project Guide:** Dr. Mustafa Basthikodi (HoD, Dept. of CSE)

---

## 📖 Project Overview

### Project Abstract

This project develops a **Hybrid Quantum Machine Learning (QML) pipeline** integrating classical attention-based neural networks with Variational Quantum Circuits (VQCs). By mapping deep-space morphological data into a quantum Hilbert space, the project examines the trade-off between peak classical accuracy and the quantum information bottleneck, and **evaluates the behavior of classical and hybrid quantum-classical models under the project's SNR degradation stress test**.

Galaxy morphology classification from survey imagery is complicated by faint, low-resolution objects, ambiguous crowd-sourced labels and image noise. The project is organized into four phases:

1. **Phase 1 & 2 — Data Science Pipeline & SAAN Architecture:** Construction of a high-confidence 3-class dataset from Galaxy Zoo vote fractions, training of the **SNR-Adaptive Attention Network (SAAN)** with interchangeable backbones (custom CNN, DenseNet121, EfficientNetV2B0), and pseudo-labeling of the ambiguous backlog.
2. **Phase 3 — Hybrid Quantum Leap:** A frozen SAAN-DenseNet121 feature extractor compressed to 4 latent variables and fed into a 4-qubit PennyLane variational circuit.
3. **Phase 4 — SNR Stress Test & Visual Analysis:** A Gaussian-noise SNR sweep (Clean, SNR 10, SNR 1, SNR 0.1) comparing the classical and hybrid models, plus Classical and Quantum Grad-CAM visualizations.

---

## ✨ Highlights

- Galaxy Zoo dataset: **61,578** galaxy images, of which **35,814** high-confidence images are labeled across 3 classes
- Custom **SNR-Adaptive Attention Network (SAAN)**: Learnable Noise Gate + Squeeze-and-Excitation attention
- Transfer-learning backbone switchboard: **DenseNet121**, **EfficientNetV2B0**, **ConvNeXtTiny**
- **Pseudo-labeling** of a **25,764-image** ambiguous backlog with confidence scores
- **4-qubit Hybrid Quantum-Classical Neural Network** (PennyLane + custom `tf.keras.layers.Layer`)
- **Cosmic Radiation Stress Test**: SNR sweep at 10, 1 and 0.1
- **Classical vs Quantum Grad-CAM** visual decoding
- Mixed-precision (`mixed_float16`) GPU pipelines with `tf.data.AUTOTUNE` streaming

---

## 🌌 Dataset

This project uses galaxy imagery and crowd-sourced morphological classifications from the **Galaxy Zoo** citizen-science project, distributed as the Kaggle Galaxy Zoo training set (61,000+ images).

- **Galaxy Zoo** – Volunteer-generated morphological classifications of galaxy images, recorded as vote fractions for each decision-tree question (e.g. "smooth vs. features/disk", "spiral arms", "odd features").

#### Dataset URL: https://data.galaxyzoo.org/

### Dataset Overview

- **Primary Source:** Galaxy Zoo
- **Files used by the code:** `training_solutions_rev1.csv` (vote fractions per `GalaxyID`) and `images_training_rev1/<GalaxyID>.jpg` (RGB galaxy cutouts)
- **Data Type:** RGB galaxy images + per-galaxy vote-fraction labels
- **Task:** 3-class image classification
- **Input:** 64×64×3 resized galaxy image
- **Output:** Elliptical / Spiral / Irregular
- **Availability:** Images are **not** stored in this repository; the derived label CSVs are (see below)

### Label Construction ("Pristine Thresholding")

`src/build_dataset_csv.py` assigns a label from the Galaxy Zoo vote fractions using strict thresholds, evaluated in this order:

| Class | Rule (vote fractions) |
|-------|-----------------------|
| **Elliptical** | `Class1.1 ≥ 0.60` |
| **Spiral** | `Class1.2 ≥ 0.60` **and** `Class4.1 ≥ 0.50` |
| **Irregular** | `Class6.1 ≥ 0.50` **or** `Class8.4 ≥ 0.50` |
| *Unclassified* | none of the above → sent to the backlog |

### Dataset Composition (committed CSVs)

| File | Rows | Contents |
|------|-----:|----------|
| `master_labels_60k.csv` | 35,814 | `filename`, `Label` — high-confidence training set |
| `unclassified_backlog.csv` | 25,764 | `filename` — ambiguous galaxies |
| `pseudo_labeled_backlog.csv` | 25,764 | `filename`, `Predicted_Label`, `Confidence_Score` |

| Class | Images | Share |
|-------|-------:|------:|
| Elliptical | 19,942 | 55.68% |
| Spiral | 10,338 | 28.87% |
| Irregular | 5,534 | 15.45% |
| **Total** | **35,814** | 100% |

---

# 📁 Repository Structure

```text
qml-galaxy-classification/
├── README.md                              # Project documentation (this file)
├── setup.md                               # Detailed environment, path & execution guide
├── requirements.txt                       # Full pip freeze of the galaxy_gpu environment
├── .gitignore                             # Ignores qml_env/, data/, models/, __pycache__/
├── master_labels_60k.csv                  # 35,814 high-confidence labeled galaxies
├── unclassified_backlog.csv               # 25,764 ambiguous galaxies
├── pseudo_labeled_backlog.csv             # SAAN pseudo-labels + confidence scores for the backlog
├── logs_hybrid_qml_2026.txt               # Hybrid QML training log
├── gradcam_collapse_proof.png             # Classical Grad-CAM across SNR levels (earlier sample)
├── gradcam_collapse_proof_with_preds.png  # Classical Grad-CAM across SNR levels with predictions
├── gradcam_qml_immunity_proof.png         # Quantum Grad-CAM across SNR levels with predictions
├── training_logs/                         # Classical training logs & engineering report
│   ├── classical_training_progress.txt    # Phase 1–3 comparative report (SAAN vs DenseNet vs EfficientNet)
│   ├── logs_cnn_main_2026-04-16.txt       # Custom SAAN training + backlog pseudo-labeling run
│   ├── logs_densenet_2026-04-18_10-16.txt # DenseNet121 run (verbose progress output)
│   ├── logs_densenet_main_2026-04-18_10-25.txt
│   ├── logs_efficientnet1_2026-04-18_10-36.txt
│   ├── logs_efficientnet2_2026-04-18_10-49.txt
│   └── logs_efficientnet_main_2026-04-22_14-43.txt
├── src/                                   # Source code (run all scripts from inside src/)
│   ├── build_dataset_csv.py               # Vote-fraction thresholding → label / backlog CSVs
│   ├── build_saan.py                      # LearnableNoiseGate, SEBlock, build_sota_saan()
│   ├── train_saan.py                      # SAAN training (backbone switchboard)
│   ├── classify_backlog.py                # Pseudo-labeling of the unclassified backlog
│   ├── train_hybrid_qml.py                # Hybrid SAAN-DenseNet121 + 4-qubit PennyLane VQC
│   ├── stress_test.py                     # Classical SNR stress test
│   ├── stress_test_qml.py                 # Hybrid QML SNR stress test
│   ├── visualize_gradcam.py               # Classical Grad-CAM figure
│   ├── visualize_gradcam_qml.py           # Quantum Grad-CAM figure
│   ├── preprocess.py                      # [Early phase] 6,000-image crop/blur/augment pipeline
│   ├── train_cnn.py                       # [Early phase] baseline CNN
│   ├── prepare_quantum_data.py            # [Early phase] StandardScaler + PCA (6 components)
│   ├── train_qnn.py                       # [Early phase] 6-qubit Qiskit VQC
│   ├── noise_showdown.py                  # [Early phase] Qiskit VQC SNR comparison
│   ├── benchmark_noise.py                 # [Early phase] SNR benchmark on processed images
│   └── grad_cam.py                        # [Early phase] Grad-CAM on the custom SAAN
└── exoplanet_src/                         # Placeholder files (empty) for the companion exoplanet project
```

---

# ⚙️ Tech Stack

## Programming Language

- Python 3.10

## Deep Learning

- TensorFlow 2.x / Keras (`tf.keras`)
- Keras Applications — DenseNet121, EfficientNetV2B0, ConvNeXtTiny (ImageNet weights)
- Mixed precision (`mixed_float16`) and `tf.data` streaming (`AUTOTUNE`)

## Quantum Machine Learning

- PennyLane — `default.qubit` simulator, TensorFlow interface, `AngleEmbedding`, `BasicEntanglerLayers`
- Qiskit — `zz_feature_map`, `real_amplitudes`, `VQC` (early-phase 6-qubit experiments; uses `qiskit-algorithms` and `qiskit-machine-learning`)

## Classical Machine Learning

- Scikit-learn — `LabelEncoder`, stratified `train_test_split`, `compute_class_weight`, `PCA`, `StandardScaler`, `accuracy_score`

## Computer Vision / Image Processing

- TensorFlow image ops (`decode_jpeg`, `resize`, random flips)
- OpenCV — Grad-CAM heatmap colormaps/overlays and early-phase image preprocessing

## Data Processing

- NumPy
- Pandas

## Visualization

- Matplotlib
- tqdm (progress reporting)

## Development Environment

- Conda (`galaxy_gpu` environment)

## GPU / CUDA

- NVIDIA CUDA & cuDNN (Mixed Precision Pipelines)

---

# 🔬 Pipeline & Methodology

```text
Galaxy Zoo vote fractions + images
  → 1. Label construction (thresholding)        build_dataset_csv.py
  → 2. Image preprocessing & augmentation       train_saan.py
  → 3. Stratified split & class weighting       train_saan.py
  → 4. SAAN classical training                  train_saan.py + build_saan.py
  → 5. Backlog pseudo-labeling                  classify_backlog.py
  → 6. Hybrid quantum-classical training        train_hybrid_qml.py
  → 7. SNR stress test                          stress_test.py / stress_test_qml.py
  → 8. Classical & Quantum Grad-CAM             visualize_gradcam.py / visualize_gradcam_qml.py
```

## 1. Dataset Acquisition & Label Construction

The Galaxy Zoo vote-fraction table (`training_solutions_rev1.csv`) is thresholded into three morphologies (see [Label Construction](#label-construction-pristine-thresholding)). Each `GalaxyID` is mapped directly to its image filename (`<GalaxyID>.jpg`). High-confidence galaxies are written to `master_labels_60k.csv` (35,814 rows) and all remaining galaxies to `unclassified_backlog.csv` (25,764 rows).

---

## 2. Image Preprocessing & Augmentation

Images are streamed from disk with `tf.data` (`map(..., num_parallel_calls=AUTOTUNE)`, batch size 64, `prefetch`):

- **Decoding:** `tf.image.decode_jpeg(..., channels=3)` — 3-channel RGB.
- **Resizing:** `tf.image.resize` to **64 × 64**.
- **Augmentation:** `random_flip_left_right` and `random_flip_up_down` in `train_saan.py`; `random_flip_left_right` in `train_hybrid_qml.py`. The same mapping function is applied to the training and validation streams.
- **Input scaling:** the manual `img / 255.0` step was removed from `train_saan.py` during the EfficientNetV2 work, because EfficientNetV2 contains an internal rescaling layer (documented in `training_logs/classical_training_progress.txt`, §4). Scaling used by each script:

| Script | Pixel range fed to the model |
|--------|------------------------------|
| `train_saan.py` (current) | 0–255 |
| `train_hybrid_qml.py` | 0–255 |
| `classify_backlog.py` | 0–1 (`/ 255.0`) |
| `stress_test.py` | 0–1 (`/ 255.0`) |
| `stress_test_qml.py` | 0–255 |
| `visualize_gradcam.py`, `visualize_gradcam_qml.py` | 0–1 (`/ 255.0`) |

---

## 3. Data Splitting & Class Balancing

- **Label encoding:** `LabelEncoder` (alphabetical): `0 = Elliptical`, `1 = Irregular`, `2 = Spiral`.
- **Split:** stratified 80 / 20 train/validation split with `random_state=42` → **28,651** training and **7,163** validation images.
- **Class Balancing:** robust `class_weights` (`compute_class_weight('balanced')`) are applied in `train_saan.py` to mathematically penalize lazy predictions and force the model to identify rare Irregular and Spiral galaxies.

---

## 4. Classical Architecture — SNR-Adaptive Attention Network (SAAN)

`build_saan.py` defines two custom layers and a backbone "switchboard":

- **Learnable Noise Gate:** A custom pre-processing layer that dynamically learns to suppress deep-space background static. Implemented as a 3×3 `Conv2D` with sigmoid activation (one filter per input channel) producing a 0–1 mask that is multiplied with the input image.
- **Squeeze-and-Excitation (SE) Block:** Channel-wise attention that allows the model to focus on galactic cores while muting noise-heavy feature maps — global average pooling → `Dense(C/8, ReLU)` → `Dense(C, sigmoid)` → channel-wise rescaling.

**SOTA SAAN (`build_sota_saan`)**

```text
Input (64×64×3)
 → LearnableNoiseGate
 → Pre-trained backbone (ImageNet, include_top=False, frozen)
      'efficientnet' → EfficientNetV2B0
      'densenet'     → DenseNet121
      'convnext'     → ConvNeXtTiny
 → SEBlock
 → GlobalAveragePooling2D
 → Dense(128, ReLU) → Dropout(0.5)
 → Dense(3, softmax)
```

**Custom SAAN (scratch-built baseline):** `LearnableNoiseGate` → three `Conv2D` (32/64/128) + `MaxPooling` + `SEBlock` stages → `Flatten` → `Dense(128)` → `Dropout(0.5)` → `Dense(3, softmax)`. This definition (`build_saan_model`) is retained in `build_saan.py` in commented-out form.

**Training configuration (`train_saan.py`)**

| Setting | Value |
|---------|-------|
| Optimizer | Adam, learning rate 0.001 |
| Loss | Categorical cross-entropy |
| Batch size | 64 |
| Max epochs | 30 |
| Early stopping | `val_loss`, patience 5, restore best weights |
| Checkpointing | `../models/saan_<backbone>_best.keras` (best `val_loss`) |
| Precision | `mixed_float16` |
| Backbone selection | `TARGET_MODEL` (line 97) |

---

## 5. AI-Driven Inference (Pseudo-Labeling)

`classify_backlog.py` loads the trained SAAN model and runs inference (no augmentation) over the 25,764-image backlog, recording the arg-max class and the maximum softmax probability as `Confidence_Score` in `pseudo_labeled_backlog.csv`. The pseudo-labels are an output dataset; they are not used to train any model in this repository.

---

## 6. Hybrid Quantum-Classical Neural Network

`train_hybrid_qml.py` performs "digital surgery" on the trained SAAN-DenseNet121:

- **Classical Feature Extraction:** The trained SAAN model is loaded, its classification head removed (output taken from the `Dense(128)` layer) and the whole network **frozen**.
- **Dimensionality Compression (The Bottleneck):** A trainable `Dense(4, tanh)` layer (`Classical_Bottleneck`) compresses the 128 extracted features into exactly 4 latent variables in the range (−1, 1).
- **Quantum Integration (VQC):** PennyLane natively integrated with a custom `tf.keras.layers.Layer` (`QuantumKerasLayer`), applied per sample via `tf.map_fn`:
  - **Device:** `default.qubit` (4 wires), `interface="tf"`, `diff_method="backprop"`.
  - **Encoding:** The 4 classical variables are mapped to a 4-qubit system using `AngleEmbedding`.
  - **Entanglement:** 2 × `BasicEntanglerLayers` with trainable weights of shape `(2, 4)` (8 quantum parameters).
  - **Measurement:** expectation values ⟨Z⟩ of qubits 0, 1 and 2 — one per galaxy class.
- **Output:** a `softmax` activation (`Final_Measurement`) collapses the three expectation values into class probabilities.

```text
Image (64×64×3)
 → Frozen SAAN-DenseNet121 (up to Dense 128)
 → Dense(4, tanh)                         Classical_Bottleneck
 → AngleEmbedding (4 qubits)
 → BasicEntanglerLayers × 2               Quantum_Entanglement_Layer
 → ⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩
 → Softmax                                Final_Measurement → Elliptical / Irregular / Spiral
```

| Setting | Value |
|---------|-------|
| Precision policy | `float32` (mixed precision disabled for the quantum layer) |
| Optimizer | Adam, learning rate 0.001 |
| Loss | Categorical cross-entropy |
| Batch size | 64 |
| Max epochs | 10 |
| Early stopping | `val_loss`, patience 3, restore best weights |
| Checkpoint | `../models/saan_hybrid_qml_best.keras` |
| Class weights | not used |

---

## 7. The Cosmic Radiation Stress Test (SNR Sweep)

`stress_test.py` (classical SAAN-DenseNet121) and `stress_test_qml.py` (hybrid QML) evaluate the same **2,000 images** sampled (`random_state=42`) from the validation split, under four conditions: **Clean, SNR 10, SNR 1, SNR 0.1**.

Gaussian noise mimicking deep-space cosmic radiation is injected per image after resizing, with noise power derived from the image's own signal power (pixel variance):

$$
P_{signal} = \mathrm{Var}(x), \qquad \sigma^2 = \frac{P_{signal}}{\mathrm{SNR}}, \qquad x_{noisy} = \mathrm{clip}\big(x + \mathcal{N}(0, \sigma^2),\ 0,\ 1\big)
$$

SNR is a linear power ratio (not dB). Noise is drawn with `tf.random.normal` without a fixed seed, and accuracy is reported via `model.evaluate`.

---

## 8. Classical & Quantum Grad-CAM Visual Decoding

`visualize_gradcam.py` and `visualize_gradcam_qml.py` render Grad-CAM heatmaps for one galaxy (`102243.jpg`) at Clean, SNR 10, SNR 1 and SNR 0.1:

- **Target layer:** the SE Block output (`model.layers[3]`).
- **Heatmap:** gradients of the top predicted class with respect to the target feature map are average-pooled per channel, used to weight the feature map, passed through ReLU and normalized to [0, 1]. At 64×64 input the DenseNet121/SE feature map is 2×2 spatially and is bilinearly upsampled to the image size.
- **Quantum Grad-CAM:** in the hybrid model the gradients flow backward from the quantum layer through the bottleneck into the classical image space.
- **Overlay:** OpenCV `COLORMAP_JET`, α = 0.4; each panel shows the model's predicted class and confidence.

---

## 9. Earlier Exploratory Phase (legacy scripts)

Before the Galaxy Zoo CSV pipeline, the project used a smaller, balanced 6,000-image dataset and a pure Qiskit VQC:

- **`preprocess.py`:** 300×300 center crop → 3×3 Gaussian anti-aliasing blur → 64×64 resize (`INTER_AREA`); minority classes augmented with flips/90° rotations to 2,000 images per class; saved as lossless PNG.
- **`train_cnn.py`:** 3-layer CNN baseline (`Rescaling` → Conv32/64/64 → Dense 64 → Dropout 0.5), 10 epochs.
- **`prepare_quantum_data.py`:** grayscale 64×64 images flattened to 4,096 features → `StandardScaler` → **PCA (6 components)** → 80/20 split.
- **`train_qnn.py`:** 6-qubit Qiskit `VQC` (`zz_feature_map` reps=1, `real_amplitudes` reps=1, COBYLA, 50 iterations) trained on 100 samples.
- **`noise_showdown.py`, `benchmark_noise.py`, `grad_cam.py`:** SNR comparisons and Grad-CAM for that phase.

---

# 📊 Experimental Results & Benchmarks

All results below are **historical results** recorded during the original project runs (Windows, NVIDIA RTX 3050). The trained model checkpoints are not stored in this repository (see [Reproducibility & Artifacts](#-reproducibility--artifacts)).

## Benchmark Summary

- 3 SAAN variants trained (Custom CNN, DenseNet121, EfficientNetV2B0)
- 1 Hybrid Quantum-Classical model (4 qubits, 2 entangling layers)
- 25,764 backlog galaxies pseudo-labeled
- 4-condition SNR stress test (Clean, SNR 10, SNR 1, SNR 0.1) on 2,000 validation images

---

## 1. Classical SAAN Models (training logs)

| Model | Backbone | Best Val. Accuracy | Best Val. Loss | Best Epoch | Epochs Run | Time / Epoch | Log |
|-------|----------|------------------:|---------------:|-----------:|-----------:|-------------:|-----|
| **SAAN-DenseNet121** | DenseNet121 (ImageNet) | **81.59%** | 0.4464 | 3 | 8 | ~40 s | `logs_densenet_main_2026-04-18_10-25.txt` |
| Custom SAAN | Scratch CNN + SE | 81.28% | 0.4513 | 13 | 18 | ~10 s | `logs_cnn_main_2026-04-16.txt` |
| SAAN-EfficientNetV2 | EfficientNetV2B0 (ImageNet) | 75.46% | 0.6224 | 19 | 24 | ~75 s | `logs_efficientnet_main_2026-04-22_14-43.txt` |

SAAN-DenseNet121 was selected as the classical champion and used as the hybrid model's feature extractor (`classical_training_progress.txt`).

**Reported SAAN performance:** a highly stable **80.85% accuracy** on clean data — the clean-condition figure from the SNR stress test (Table 4).

**Engineering runs** (documented in `classical_training_progress.txt`, §4):

| Run | Observation | Resolution |
|-----|-------------|------------|
| `logs_efficientnet1_…` | Validation accuracy collapsed to ~15.45% | Backbone frozen (`base_model.trainable = False`) |
| `logs_efficientnet2_…` | Validation accuracy flat at 55.67% for 7 epochs ("Black Image" double-scale bug) | Manual `/ 255.0` removed (EfficientNetV2 rescales internally) |

---

## 2. Pseudo-Labeling of the Backlog

| Predicted Class | Galaxies | Mean Confidence |
|-----------------|---------:|----------------:|
| Elliptical | 10,845 | 0.836 |
| Irregular | 8,309 | 0.650 |
| Spiral | 6,610 | 0.667 |
| **Total** | **25,764** | — |

Inference over the full backlog completed in **22 seconds** (403 batches) on the RTX 3050 — under 30 seconds (`logs_cnn_main_2026-04-16.txt`).

---

## 3. Hybrid Quantum-Classical Model (training log)

| Epoch | Train Loss | Train Accuracy | Val. Loss | Val. Accuracy | Epoch Time |
|------:|-----------:|---------------:|----------:|--------------:|-----------:|
| 1 | 1.0610 | 43.38% | 1.0245 | 55.69% | 955 s |
| 2 | 1.0049 | 55.70% | 0.9919 | 55.67% | 934 s |
| 3 | 0.9851 | 55.68% | 0.9798 | 55.67% | 905 s |
| 4 | 0.9765 | 55.68% | **0.9741** | 55.67% | 899 s |
| 5 | 0.9790 | 55.68% | 1.0201 | 55.67% | 900 s |
| 6 | 0.9873 | 55.68% | 0.9768 | 55.67% | 904 s |
| 7 | 0.9752 | 55.68% | 0.9743 | 55.67% | 898 s |

Early stopping ended training after epoch 7 (best `val_loss` at epoch 4); total training time ≈ 1.8 h (`logs_hybrid_qml_2026.txt`).

**Reported hybrid performance:** the Hybrid QML model converged at **56.75% accuracy** (stress-test evaluation, Table 4). This established the quantum baseline, reflecting the expected trade-off of compressing 128 high-resolution classical features into a 4-qubit state space.

---

## 4. Cosmic Radiation Stress Test (SNR Sweep)

| Condition | Classical SAAN-DenseNet121 | Hybrid QML (4 qubits) |
|-----------|---------------------------:|----------------------:|
| Clean (no noise) | **80.85%** | **56.75%** |
| SNR 10 (Light Static) | 78.8% | 56.75% |
| SNR 1 (Heavy Static) | 48.4% | 56.75% |
| SNR 0.1 (Severe Cosmic Radiation) | **28.10%** | **56.75%** |

Evaluated on 2,000 validation images. Clean and SNR 0.1 values are as reported in the project write-up; the classical SNR 10 and SNR 1 values (one decimal place) are the figures recorded in `visualize_gradcam.py`. The stress-test scripts print to the console; their raw output is not stored in the repository.

---

## 5. Grad-CAM Visualizations

| Figure | Model | Description |
|--------|-------|-------------|
| `gradcam_collapse_proof_with_preds.png` | Classical SAAN-DenseNet121 | Grad-CAM + predicted class at Clean / SNR 10 / SNR 1 / SNR 0.1 |
| `gradcam_collapse_proof.png` | Classical SAAN-DenseNet121 | Same sweep, earlier sample galaxy, without prediction labels |
| `gradcam_qml_immunity_proof.png` | Hybrid QML | Quantum Grad-CAM + predicted class at Clean / SNR 10 / SNR 1 / SNR 0.1 |

![Classical Grad-CAM](gradcam_collapse_proof_with_preds.png)
![Quantum Grad-CAM](gradcam_qml_immunity_proof.png)

---

# 💡 Key Findings

- **Classical Champion:** SAAN-DenseNet121 achieved the best classical validation accuracy (81.59%), ahead of the Custom SAAN (81.28%) and SAAN-EfficientNetV2 (75.46%). The project attributes EfficientNetV2's lower score to a receptive-field mismatch with 64×64 inputs, since it was optimized for high-resolution (e.g. 224×224) images.

- **The Classical Collapse:** The standard SAAN architecture proved fragile to data corruption. As noise increased to SNR 0.1 (Severe Cosmic Radiation), classical accuracy collapsed from **80.85% down to 28.10%**.

- **The Quantum Shield:** The Hybrid QML model held **56.75% accuracy** across Clean Data, SNR 10, SNR 1 and SNR 0.1 in the project's stress test, i.e. its measured accuracy did not change across the tested noise levels.

- **Accuracy–Robustness Trade-off:** The hybrid model sacrifices peak classical accuracy (80.85% → 56.75% on clean data), which the project attributes to the quantum information bottleneck of compressing 128 features into 4 qubits.

- **Quantum Grad-CAM Visual Decoding:** The project's Grad-CAM analysis shows the classical model's attention drifting into the background static under SNR 0.1, while the Quantum Grad-CAM heatmaps were interpreted as the entangled qubits maintaining a lock on the core of the galaxy, filtering out the chaotic visual noise.

- **Rapid Pseudo-Labeling:** The trained SAAN classified the 25,764-galaxy backlog in 22 seconds, producing a custom-labeled dataset with confidence scores (mean 0.836 for Elliptical predictions).

---

# 🛠️ Installation & Quickstart

Full details — path configuration, checkpoint map and troubleshooting — are in **[`setup.md`](setup.md)**.

## Prerequisites

- **Linux (recommended)** for the pinned `requirements.txt`; the original experiments were run on **Windows**
- Python 3.10
- Conda (or `venv`)
- NVIDIA GPU with CUDA support and NVIDIA drivers — **required** by `train_saan.py` and `classify_backlog.py`
- The Galaxy Zoo images and `training_solutions_rev1.csv` (https://data.galaxyzoo.org/)
- Git

> **Note**
>
> `requirements.txt` is a full `pip freeze` of the project's `galaxy_gpu` environment (Python 3.10.20, Linux). It contains the project's dependencies together with other packages that were installed in that environment.
>
> The early-phase Qiskit scripts (`train_qnn.py`, `noise_showdown.py`) additionally import `qiskit-algorithms` and `qiskit-machine-learning`, which are not in that environment.

---

## Software Requirements

### Deep Learning

- TensorFlow / Keras

### Quantum Computing

- PennyLane
- Qiskit

### Machine Learning

- Scikit-learn

### Scientific Computing

- NumPy
- Pandas

### Computer Vision & Visualization

- OpenCV
- Matplotlib

These are installed through `requirements.txt`.

---

## 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/Arjun13-git/qml-galaxy-classification.git
cd qml-galaxy-classification

# Create and activate conda environment
conda create -n galaxy_gpu python=3.10 -y
conda activate galaxy_gpu

# Install dependencies
pip install -r requirements.txt

# Verify
python --version
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 2. Dataset & Path Configuration

1. Obtain the Galaxy Zoo data from https://data.galaxyzoo.org/ and place it as:

   ```text
   <DATA_ROOT>/
   ├── training_solutions_rev1.csv
   └── images_training_rev1/<GalaxyID>.jpg
   ```

2. The scripts contain the original development paths (e.g. `C:\galaxy_datasets\images_training_rev1`). Edit them to point to `<DATA_ROOT>`:

   | File | Line(s) |
   |------|---------|
   | `src/build_dataset_csv.py` | 6, 30, 35 |
   | `src/train_saan.py` | 52 |
   | `src/classify_backlog.py` | 20 |
   | `src/train_hybrid_qml.py` | 40 |
   | `src/stress_test.py` | 24 |
   | `src/stress_test_qml.py` | 70 |
   | `src/visualize_gradcam.py` | 76 |
   | `src/visualize_gradcam_qml.py` | 118 |

3. Create the checkpoint directory (ignored by Git):

   ```bash
   mkdir -p models
   ```

The label CSVs are already committed, so `build_dataset_csv.py` only needs to be re-run to regenerate them.

---

## 3. Model Training

All scripts use paths relative to `src/` and take no command-line arguments.

```bash
cd src

# (Optional) Regenerate label CSVs from Galaxy Zoo vote fractions
python build_dataset_csv.py

# Train SAAN — set TARGET_MODEL in train_saan.py (line 97) to 'densenet', 'efficientnet' or 'convnext'
python train_saan.py              # → ../models/saan_<backbone>_best.keras

# Train the Hybrid Quantum-Classical model (requires ../models/saan_densenet_best.keras)
python train_hybrid_qml.py        # → ../models/saan_hybrid_qml_best.keras
```

---

## 4. Inference, Evaluation & Visualization

```bash
# Pseudo-label the unclassified backlog (loads ../models/saan_best_model_60k.keras)
python classify_backlog.py        # → ../pseudo_labeled_backlog.csv

# Cosmic Radiation Stress Test (Clean, SNR 10, SNR 1, SNR 0.1)
python stress_test.py             # Classical SAAN-DenseNet121
python stress_test_qml.py         # Hybrid QML

# Grad-CAM Visual Decoding
python visualize_gradcam.py       # → ../gradcam_collapse_proof_with_preds.png
python visualize_gradcam_qml.py   # → ../gradcam_qml_immunity_proof.png
```

> Re-running `train_hybrid_qml.py`, `classify_backlog.py` or the Grad-CAM scripts overwrites the committed log, CSV and figures listed above.

---

# 📦 Reproducibility & Artifacts

| Component | Status |
|-----------|--------|
| Source code for all phases | ✅ Included (`src/`) |
| Label CSVs (35,814 / 25,764) and pseudo-label CSV | ✅ Included |
| Training logs and engineering report | ✅ Included (`training_logs/`, `logs_hybrid_qml_2026.txt`) |
| Grad-CAM figures | ✅ Included |
| Galaxy Zoo images | ⬇️ Download required (https://data.galaxyzoo.org/) |
| Trained checkpoints (`models/*.keras`) | ❌ Not included — re-train with the commands above |
| Stress-test raw output | ❌ Not stored (console output only) |
| Checkpoint for `classify_backlog.py` (`saan_best_model_60k.keras`) | ❌ Produced by the Custom SAAN, whose builder is commented out in `build_saan.py` |

- The historical results were produced on Windows with an older TensorFlow release; the pinned environment uses TensorFlow 2.21 / Keras 3 on Linux. Re-trained models are expected to be close to, but not identical with, the logged runs.
- The data split and stress-test sample are seeded (`random_state=42`); weight initialization, dropout, augmentation and noise are not.

---

# 🚀 Hardware Used for Benchmarking

### Historical Benchmark Hardware

| Component | Specification |
|-----------|---------------|
| System | Lenovo LOQ laptop |
| GPU | NVIDIA GeForce RTX 3050 A Laptop GPU, 4 GB VRAM (compute capability 8.9) |
| OS | Windows |
| Environment | Conda `galaxy_gpu` |
| Precision | `mixed_float16` (classical), `float32` (hybrid QML) |

Measured epoch times on this hardware: Custom SAAN ≈ 10 s, SAAN-DenseNet121 ≈ 40 s, SAAN-EfficientNetV2 ≈ 75 s, Hybrid QML ≈ 900 s.

### Recommended Hardware

| Component | Recommendation |
|-----------|----------------|
| GPU | NVIDIA CUDA-capable GPU, ≥ 4 GB VRAM |
| OS | Linux (for the pinned environment) |
| Storage | Local disk space for the Galaxy Zoo image set |

### Current Environment Notes

In the captured `galaxy_gpu` environment (Linux, TensorFlow 2.21), TensorFlow reports `Cannot dlopen some GPU libraries` and does not register the GPU, so the GPU-only scripts (`train_saan.py`, `classify_backlog.py`) cannot run until the CUDA libraries load correctly. See `setup.md` → Troubleshooting.

---

# 📚 Research Focus

This repository investigates how a hybrid quantum-classical neural network compares with its classical counterpart for galaxy morphology classification under image noise.

The primary objectives include:

- Building a high-confidence 3-class galaxy dataset from Galaxy Zoo vote fractions.
- Designing an attention-based classical baseline (SAAN) with a learnable noise gate and squeeze-and-excitation attention, and comparing transfer-learning backbones.
- Integrating a frozen classical feature extractor with a 4-qubit variational quantum circuit.
- Evaluating classical and hybrid models under a controlled SNR degradation sweep.
- Visually interpreting both models with Classical and Quantum Grad-CAM.

---

# 📜 License

No license file is currently included in this repository. All rights are reserved by the authors unless a license is added.
