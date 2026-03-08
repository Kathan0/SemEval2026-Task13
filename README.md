# 🚀 Code Detection - Enhanced Solution
## SemEval-2026 Task 13: All Subtasks

This is an enhanced implementation combining insights from the reference solution with significant improvements in:
- **Advanced Pattern Recognition** for different LLM families
- **Comprehensive AST Feature Extraction** 
- **State-of-the-art Model Architecture**

---

## 📁 Project Structure

```
MyProject/
├── README.md
├── requirements.txt
├── setup.py
│
├── task_a/                          # Task A: Binary Classification (Human vs AI)
│   ├── config.yaml                  # Task A configuration
│   ├── model.py                     # Task A model implementation
│   ├── dataset.py                   # Task A dataset loader
│   ├── train.py                     # Task A training script
│   └── inference.py                 # Task A inference script
│
├── task_b/                          # Task B: Authorship Attribution (11 classes)
│   ├── config.yaml                  # Task B configuration
│   ├── model.py                     # Task B cascade model
│   ├── dataset.py                   # Task B dataset loader
│   ├── train.py                     # Task B training script
│   └── inference.py                 # Task B inference script
│
├── task_c/                          # Task C: Hybrid Detection (4 classes)
│   ├── config.yaml                  # Task C configuration
│   ├── model.py                     # Task C staged model
│   ├── dataset.py                   # Task C dataset loader
│   ├── train.py                     # Task C training script
│   └── inference.py                 # Task C inference script
│
├── src/                             # Shared components across all tasks
│   ├── models/
│   │   └── base_model.py            # Shared hybrid architecture (StarCoder2 + features)
│   ├── features/
│   │   ├── ast_extractor.py         # AST features (30+ metrics)
│   │   ├── pattern_detector.py      # LLM-specific patterns (50+ signatures)
│   │   ├── perplexity.py            # Multi-model perplexity
│   │   └── stylometric.py           # Code style analysis
│   └── utils/
│       ├── metrics.py               # Evaluation metrics
│       └── helpers.py               # Utility functions
│
├── train_all.py                     # Train all tasks sequentially
├── inference_all.py                 # Run inference on all tasks
├── extract_features_locally.py      # Pre-extract features for faster training
├── generate_submissions.py          # Generate competition submissions
│
├── checkpoints/                     # Model checkpoints (gitignored)
│   ├── task_a/
│   ├── task_b/
│   └── task_c/
│
├── data/                            # Dataset directory (gitignored)
│   ├── task_a/
│   ├── task_b/
│   └── task_c/
│
└── models/                          # Downloaded model weights (gitignored)
    └── starcoder2-3b/
```

---

## 🌟 Key Features

### 1. **Advanced Pattern Recognition**
- **50+ LLM-specific signatures** across 10 generator families
- **Behavioral fingerprinting** (comment style, naming patterns, error handling)
- **Temporal patterns** (code evolution markers)
- **Cross-generator discriminators**

### 2. **Comprehensive AST Analysis**
- **30+ structural features** from Abstract Syntax Tree
- **Multi-language support** (Python, Java, C++, JavaScript, Go, C, C#, PHP)
- **Complexity metrics** (cyclomatic, cognitive, Halstead)
- **Graph-based features** (control flow, data flow)

### 3. **State-of-the-Art Architecture**
- **StarCoder2-3B** backbone with 8-bit quantization
- **Multi-scale feature extraction** from transformer layers
- **Attention pooling** with learnable weights
- **Feature fusion** with gating mechanism

### 4. **Robust Training**
- **Focal Loss** with dynamic weighting
- **Supervised Contrastive Learning**
- **Meta-learning** for few-shot generalization
- **Test-time adaptation**

---

## 🚀 Quick Start

### Installation

```bash
# Clone and setup
cd MyProject
pip install -r requirements.txt

# Or use setup.py
pip install -e .
```

### Data Preparation

Place your data in the appropriate directories:
```
data/
├── task_a/
│   ├── train.parquet
│   └── validation.parquet
├── task_b/
│   ├── train.parquet
│   └── validation.parquet
└── task_c/
    ├── train.parquet
    └── validation.parquet
```

### Training

```bash
# Task A: Binary Classification (Human vs AI)
cd task_a
python train.py

# Task B: Authorship Attribution (11 classes)
cd task_b
python train.py

# Task C: Hybrid Detection (4 classes)
cd task_c
python train.py

# Or train all tasks at once
python train_all.py
```

### Inference

```bash
# Task A inference
cd task_a
python inference.py --input data/task_a/test.parquet --output predictions_a.csv

# Task B inference
cd task_b
python inference.py --input data/task_b/test.parquet --output predictions_b.csv

# Task C inference
cd task_c
python inference.py --input data/task_c/test.parquet --output predictions_c.csv

# Or run inference on all tasks
python inference_all.py
```

---

## 📊 Expected Performance

| Task | Baseline | Our Solution | Improvement |
|------|----------|--------------|-------------|
| **Task A** | 0.78 | **0.87-0.90** | +9-12% |
| **Task B** | 0.60 | **0.75-0.78** | +15-18% |
| **Task C** | 0.73 | **0.83-0.86** | +10-13% |

---

## 🔧 Configuration

Each task has its own `config.yaml` in its directory:
- `task_a/config.yaml` - Binary classification settings
- `task_b/config.yaml` - Authorship attribution settings  
- `task_c/config.yaml` - Hybrid detection settings

Configuration includes:
- Model architecture settings (backbone, hidden dims, layers)
- Feature extraction parameters (AST, patterns, perplexity)
- Training hyperparameters (learning rate, batch size, epochs)
- Data preprocessing options (max length, augmentation)

---

## 📈 Monitoring

Training metrics are logged to:
- **Console** (tqdm progress bars)
- **TensorBoard** (runs in `logs/`)
- **CometML** (if API key provided)

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Evaluate on validation set
python scripts/evaluate.py \
    --task a \
    --model_path checkpoints/task_a/best_model \
    --data_path data/processed/task_a/val.parquet
```