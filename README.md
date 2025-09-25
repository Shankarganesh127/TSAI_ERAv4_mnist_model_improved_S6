## 📘 Model Progression & Evaluation Report

This document presents a clear, progressive narrative of three MNIST CNN model versions (`model_v1.py`, `model_v2.py`, `model_v3.py`). Each iteration pursued explicit Targets, delivered measurable Results, and produced actionable Analysis to guide the next step.

> All metrics below are extracted from training logs in `logs/training_v*.log` (15 epochs each). Best accuracies are taken as the maximum value observed across epochs. Where earlier draft notes differed, the values here reflect the actual logged results.

---

### 🔍 Quick Comparison

| Feature | Model v1 | Model v2 | Model v3 |
|---------|----------|----------|----------|
| Total Parameters | 10,970 | 4,232 | 4,968 |
| Best Train Accuracy | 98.20% (Ep15) | 98.21% (Ep15) | 98.75% (Ep15) |
| Best Test Accuracy | 99.35% (Ep11) | 99.08% (Ep14/15) | 99.45% (Ep14) |
| Scheduler | StepLR | StepLR | OneCycleLR (per-batch) |
| Batch Size (train/test) | 128 / 1000 | 128 / 1000 | 64 / 1000 |
| Augmentation | ±15° rot + 0.1 shift | Same as v1 | ±5° rot + 0.03 shift |
| Regularization | Dropout2d (2×) | Dropout2d (1×) | (No dropout) |
| Architectural Theme | Deeper, wide mid-blocks | Parameter minimization | Same depth, widened selective channels |

---

## 🧪 Version Details

### 1️⃣ Model v1 – Baseline Structured CNN
**Targets**
1. Establish a clean modular convolutional backbone.
2. Keep parameters reasonably low (< 12K) while achieving ≥99.3% test accuracy.
3. Integrate logging, augmentation, and stable scheduling (StepLR).

**Results**
- Parameters: 10,970
- Best Test Accuracy: 99.35% (Epoch 11)
- Best Train Accuracy: 98.20% (Epoch 15)
- Gap (best test − best train epoch 11 vs 15 high): Test slightly higher than train mid-run (healthy generalization)

**Analysis**
- The model underfits slightly (train never exceeds ~98.2%) indicating headroom for optimization.
- Two dropout placements plus relatively strong augmentation slow convergence but support generalization.
- StepLR (step=6, gamma=0.1) causes a noticeable stabilization after epoch 6; further gains plateau after epoch 11.
- Serves as a strong reference for pareto trade-offs (accuracy vs capacity).

**Key Architectural Notes**
- Multi-stage conv “tower” with reduction to 1×1 via a 7×7 kernel late.
- Use of 1×1 + 3×3 transitions (implicit bottlenecking) not yet exploited aggressively.

---

### 2️⃣ Model v2 – Parameter Efficiency Focus
**Targets**
1. Reduce parameter count by >50% vs v1 while retaining ≥99.0% test accuracy.
2. Preserve training stability with minimal architectural complexity.
3. Keep the same optimizer & scheduler for controlled comparison.

**Results**
- Parameters: 4,232 (−61.4% vs v1)
- Best Test Accuracy: 99.08% (Epochs 14 & 15)
- Best Train Accuracy: 98.21% (Epoch 15)
- Efficiency: 23.41 test-accuracy-per-kiloparam (99.08 / 4.232)

**Analysis**
- Massive compression with only a 0.27 pp absolute drop from v1’s best test performance.
- Train/test curves remain tightly coupled → model is capacity-limited but not overfitting.
- Late-epoch test improvements (99.04 → 99.08) suggest scheduler annealing still effective.
- This version establishes the “lean backbone” used as a base for performance-tuned variants.

**Key Architectural Changes vs v1**
- Channel widths reduced throughout (peaks at 12 vs 20 in v1 mid-blocks).
- Added final AvgPool2d + Flatten classification head.
- Single dropout retained early; later dropout removed for efficiency.

---

### 3️⃣ Model v3 – Training Dynamics Optimization
**Targets**
1. Improve test accuracy toward ≥99.4% without a large parameter jump.
2. Accelerate convergence via a modern LR schedule (OneCycleLR).
3. Reduce augmentation intensity to enhance feature fidelity for a small model.
4. Introduce gradient stability (clipping) & per-batch LR stepping visibility.

**Results**
- Parameters: 4,968 (+17.4% vs v2; still <50% of v1)
- Best Test Accuracy: 99.45% (Epoch 14)
- Best Train Accuracy: 98.75% (Epoch 15)
- Efficiency: 20.02 test-accuracy-per-kiloparam (slightly lower than v2 due to added channels)
- OneCycleLR Config: max_lr=0.065, pct_start=0.1, div_factor=5, final_div_factor=100 (per-batch stepping)

**Analysis**
- OneCycleLR + milder augmentation unlocked additional performance without overfitting (test still leads train mid-to-late epochs).
- Parameter increase (selective widening) contributed modest capacity gains; could be rolled back if strict parity with v2 is required.
- Gradient clipping (norm=2) likely helped stabilization at higher early-phase LR.
- Further headroom may require label smoothing or stochastic weight averaging rather than widening.

**Key Training Enhancements vs v2**
- Per-batch LR logging (diagnostic transparency).
- Lighter transforms (±5° / 0.03 shift) to reduce excessive deformation noise.
- Removed dropout to let added channels express capacity.

---

## 📊 Epoch-by-Epoch Accuracy

| Epoch | v1 Train | v1 Test | v2 Train | v2 Test | v3 Train | v3 Test |
|-------|----------|---------|----------|---------|----------|---------|
| 01 | 89.72% | 98.16% | 85.78% | 96.94% | 88.84% | 97.56% |
| 02 | 95.69% | 97.90% | 95.65% | 97.10% | 96.29% | 97.76% |
| 03 | 96.28% | 98.48% | 96.47% | 97.79% | 97.00% | 97.96% |
| 04 | 96.70% | 99.05% | 96.98% | 98.04% | 97.42% | 98.58% |
| 05 | 97.05% | 98.90% | 97.25% | 98.47% | 97.71% | 98.78% |
| 06 | 97.19% | 98.96% | 97.28% | 98.78% | 97.81% | 98.67% |
| 07 | 97.73% | 99.26% | 97.87% | 98.98% | 97.93% | 98.71% |
| 08 | 97.99% | 99.30% | 98.08% | 99.01% | 98.18% | 98.99% |
| 09 | 97.98% | 99.34% | 98.08% | 99.00% | 98.23% | 99.13% |
| 10 | 98.03% | 99.33% | 98.15% | 99.02% | 98.34% | 99.05% |
| 11 | 98.13% | 99.35% | 98.16% | 99.04% | 98.42% | 99.09% |
| 12 | 98.05% | 99.24% | 98.15% | 99.03% | 98.57% | 99.33% |
| 13 | 98.12% | 99.24% | 98.21% | 99.04% | 98.71% | 99.42% |
| 14 | 98.07% | 99.30% | 98.17% | 99.08% | 98.71% | 99.45% |
| 15 | 98.20% | 99.31% | 98.21% | 99.08% | 98.75% | 99.42% |

> Bold best per column (Summary): v1 test 99.35% (Ep11), v2 test 99.08% (Ep14/15), v3 test 99.45% (Ep14).

---

## 📐 Parameter Efficiency

| Model | Params | Best Test Acc | Acc / 1K Params |
|-------|--------|---------------|-----------------|
| v1 | 10,970 | 99.35% | 9.06 |
| v2 | 4,232 | 99.08% | 23.41 |
| v3 | 4,968 | 99.45% | 20.02 |

Observation: v2 maximizes accuracy density; v3 trades a modest efficiency drop for absolute peak accuracy.

---

## 🔄 Evolution Timeline

| Phase | Focus | Notable Actions | Outcome |
|-------|-------|-----------------|---------|
| v1 Baseline | Structured depth & stability | Multi-block conv stack, dropout, StepLR | Solid generalization, mild underfit |
| v2 Compression | Aggressive parameter reduction | Narrow channels, late GAP-style head, prune dropout | 61% fewer params, accuracy retained |
| v3 Optimization | Training dynamics & fine fit | OneCycleLR, reduced aug, batch size ↓, grad clipping, selective widening | +0.37 pp over v2 test accuracy |

---

## 🧪 Training Configuration Reference

| Aspect | v1 & v2 (shared) | v3 |
|--------|------------------|----|
| Optimizer | SGD (lr 0.05, momentum 0.9) | SGD (initial low lr ramped by OneCycle) |
| Scheduler | StepLR(step=6, gamma=0.1) | OneCycleLR (pct_start=0.1) |
| Loss | NLLLoss (log_softmax outputs) | Same |
| Augment | Rot ±15°, trans 0.1 | Rot ±5°, trans 0.03 |
| Gradient Clipping | No | Yes (norm=2) |
| LR Monitoring | Epoch-level | Per-batch logged |

---

## 🧭 Recommended Next Steps
1. Re-introduce a very light regularizer (e.g., label smoothing ε=0.05) to safely explore raising train accuracy without overfitting.
2. Try Stochastic Weight Averaging (SWA) from epochs 10–15 for potential +0.02–0.05 pp test lift.
3. Profile inference latency: compare v2 vs v3 to decide if the 736 extra params justify deployment.
4. Automate results extraction (script to parse logs → JSON → table injection into README).
5. Add learning rate and loss curves (matplotlib) for visual diagnostics.

---

## 🗂️ Log Provenance (Traceability)
| Model | Log File | Verified Sections |
|-------|----------|-------------------|
| v1 | `logs/training_v1.log` | Dataloader args, 15 epoch loop, architecture dump |
| v2 | `logs/training_v2.log` | Parameter count 4,232, epoch metrics, layer breakdown |
| v3 | `logs/training_v3.log` | OneCycleLR config, widened channels, per-epoch metrics |

All metrics in this report were re-derived directly from these logs (not manually retyped summaries).

---

## ✅ Summary
- Achieved a peak of 99.45% test accuracy (v3) while staying <5K parameters.
- Demonstrated >60% compression (v1 → v2) with negligible performance loss.
- Showed that training dynamics (scheduler + augmentation tuning) can outperform pure architectural pruning for final gains.
- Clear decision trade-off: choose v2 for efficiency or v3 for absolute accuracy.

> If you would like an auto-generated plot bundle or a rollback of v3 to strict v2 parameter parity with retained OneCycleLR, request: "generate diagnostic bundle" or "roll back v3 params".

---

### 🔖 Appendix: Architecture Parameter Totals
| Model | Total Params | Trainable | Non-Trainable |
|-------|--------------|-----------|---------------|
| v1 | 10,970 | 10,970 | 0 |
| v2 | 4,232 | 4,232 | 0 |
| v3 | 4,968 | 4,968 | 0 |

---

© 2025 – MNIST Lightweight CNN Iterative Optimization Report


# 🧠 MNIST Lightweight CNN Suite

This repository contains multiple progressively refined CNN models for the MNIST dataset along with a reproducible training pipeline, logging, and architecture introspection utilities.

---

## 📁 Folder & File Structure

```
ERA_v4_MNIST_model_S6/
├── main.py                 # Entry point: choose & run model versions interactively
├── model_v0.py             # (Optional baseline) Larger reference architecture
├── model_v1.py             # Baseline structured CNN (~10.9K params)
├── model_v2.py             # Parameter–efficient variant (~4.2K params)
├── model_v3.py             # Tuned training dynamics variant (~5.0K params)
├── data_setup.py           # DataLoader + transforms configuration
├── train_test.py           # Training/testing loop with progress bars & metrics
├── summarizer.py           # Architecture + parameter introspection utilities
├── logger_setup.py         # Tqdm-safe logging initialization (console + file)
├── README.md               # (This file)
├── README_1909.md          # Detailed model progression & analysis report
├── pyproject.toml          # Python project & dependency specification
├── uv.lock                 # Locked dependency versions (managed by uv/pdm/poetry style)
├── logs/                   # Generated log files per model run
│   ├── training_v1.log
│   ├── training_v2.log
│   └── training_v3.log
└── __pycache__/            # Python bytecode cache (ignored in VCS usually)
```

### 🔍 Key Directories
- `logs/` — Persistent run artifacts (epoch metrics, architecture dumps).
- `__pycache__/` — Auto-generated; safe to ignore/delete.

---

## 🗂️ File Descriptions

| File | Purpose | Notable Highlights |
|------|---------|-------------------|
| `main.py` | Interactive runner for one or all model versions | Captures model summary + layer breakdown into logs; prompts for version & mode (params check vs full training) |
| `model_v0.py` | (If present) Larger earlier baseline | Useful for contrast; not always used in current experiments |
| `model_v1.py` | Structured baseline CNN | StepLR scheduling; moderate depth; dropout for regularization |
| `model_v2.py` | Parameter–efficient compressed model | Aggressive channel pruning and global pooling-style classifier |
| `model_v3.py` | Training dynamics optimized variant | OneCycleLR, lighter augmentation, gradient clipping, widened selective channels |
| `data_setup.py` | Data pipeline setup | Defines train/test transforms; returns DataLoaders with configurable batch sizes |
| `train_test.py` | Epoch loops for training & evaluation | Per-batch LR stepping support; tqdm progress; gradient clipping; metric logging |
| `summarizer.py` | Architecture introspection & param stats | Layer-wise parameter listing; type frequency summary; torchsummary integration |
| `logger_setup.py` | Central logging initialization | Idempotent setup; tqdm-friendly handler; optional file logging (`logs/training.log`) |
| `README_1909.md` | Extended model evolution report | Targets / Results / Analysis for v1–v3 |
| `pyproject.toml` | Project metadata & dependencies | Ensures reproducibility of environment |
| `uv.lock` | Locked versions snapshot | Guarantees deterministic installs |

---

## ⚙️ Environment Setup

Install dependencies (Python ≥3.10 recommended). If using `uv` (fast Python package manager):

```powershell
uv sync
```

Or using pip (if you manually extract dependencies from `pyproject.toml`):

```powershell
pip install -r requirements.txt
```

> If `requirements.txt` doesn’t exist, you can generate one via: `uv export --format requirements-txt > requirements.txt`.

---

## 🚀 How to Run

### 1. Run Interactively (Recommended)
You will be prompted for a model version and whether to run only parameter checks (architecture dump) or full training.

```powershell
python main.py
```

Prompts:
1. `Enter model versions or leave blank for all versions one by one:`
	- Blank → runs versions 0–3 sequentially if all exist (adjust as needed)
	- `1` → only model_v1
	- `2` → only model_v2, etc.
2. `Enter 1 for params check only, 0 for full training/testing:`
	- `1` → Only summary + architecture checks logged
	- `0` → Full 15-epoch (or configured) training cycle

### 2. Full Training for a Single Model (Example: v3)

```powershell
python main.py
# When prompted:
# Enter model versions...: 3
# Enter 1 for params check...: 0
```

### 3. Architecture / Parameter Inspection Only

```powershell
python main.py
# Enter model versions...: 2
# Enter 1 for params check...: 1
```

### 4. Logging Output
- Console displays tqdm progress with dynamic LR (if OneCycleLR).
- File logs (if enabled in `main.py` via `setup_logging(log_to_file=True)`) are written to:
  - `logs/training.log` (current session consolidated)
  - `logs/training_v*.log` (per version if you have version-specific handlers already generated from earlier runs)

### 5. Viewing Architecture Details
Each run appends two structured blocks to the log:
1. Torchsummary-style layer output
2. `--- Model Architecture Checks ---` with per-layer parameter accountability

---

## 🧪 Example: What a Training Log Looks Like (Excerpt)

```text
2025-09-25 17:06:15,861 - INFO - Epoch 10/15: Train set final results: Average loss: 51.1472, Accuracy: 59006/60000 (98.34%)
2025-09-25 17:06:33,657 - INFO - Epoch 10/15:Test set final results: Average loss: 0.0297, Accuracy: 9905/10000 (99.05%)
...
2025-09-25 17:14:50,045 - INFO - --- Model Architecture Checks ---
2025-09-25 17:14:50,045 - INFO - Total Parameters: 4,968
```

---

## 🧬 Choosing a Model

| Version | When to Use | Trade-Off |
|---------|-------------|-----------|
| v1 | Need a stable baseline with moderate params | Slight underfit; good generalization |
| v2 | Need max parameter efficiency | Slight accuracy drop vs v1; very dense performance |
| v3 | Need highest accuracy (<5K params) | Slight param increase vs v2; more complex schedule |

For detailed comparative Targets / Results / Analysis see `README_1909.md`.

---

## 🔧 Modifying Hyperparameters

Edit inside the model config functions (`set_config_v*()` in each `model_v*.py`):
- Learning rate / optimizer
- Scheduler (swap OneCycleLR ↔ StepLR)
- Batch size / transform overrides

Augmentations can be customized via:
- `model_v*_config.get_train_transforms()` (if defined)
- Or override DataSetup arguments before DataLoader creation.

---

## 🧾 Reproducing Results
1. Sync/install dependencies.
2. Run each model with full training (param check flag = 0).
3. Inspect `logs/training_v*.log` for final and best epoch metrics.
4. Cross-reference architecture via appended parameter summaries.

---

## 🛠 Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Log truncates mid-run | Re-init logging in subprocess | Ensure running via `python main.py` (not importing main elsewhere) |
| No LR shown in tqdm | Using StepLR (epoch-based) | Switch to OneCycleLR (v3) or add batch stepping flag |
| CUDA not used | GPU not available | Check `torch.cuda.is_available()`; install CUDA-enabled PyTorch |
| Slow start on Windows | Dataset download or first transform compile | Allow first epoch to finish; subsequent runs faster |

---

## 📌 Next Additions (Optional Enhancements)
- Auto log parsing → JSON → metrics table regeneration
- Learning curve plots export (`plot_results()` already scaffolded)
- SWA / EMA improvements path
- Label smoothing for final tenths of a percent

---

## 🙌 Attribution
Experimental workflow aligned with incremental design → compression → training dynamics optimization. Logs provide ground-truth traceability.

---

### ✅ Quick Start Recap
```powershell
uv sync              # or pip install -r requirements.txt
python main.py       # choose model version and mode
type logs\training_v3.log | more  # inspect results (Windows PowerShell)
```

---

© 2025 MNIST Lightweight CNN Experiment Suite

