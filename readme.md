# Learning Dynamics in Sequence Learning

A collection of analyses investigating how motor learning unfolds during sequence learning tasks. This repository contains three independent studies analyzing force pattern dynamics, execution time improvements, and the effects of reward manipulation on motor learning.

## 📚 Project Overview

This repository contains Jupyter notebooks and supporting utilities for analyzing trial-by-trial force pattern dynamics in motor sequence learning. Each study employs different experimental manipulations to understand the mechanisms underlying skill acquisition.
---

## 📁 Repository Structure

```
LearningDynamicsProjects/
├── README.md                          # This file
├── utils.py                           # Shared utility functions
├── Data/                              # Data files (in .gitignore)
├── Figs/                              # Generated figures
│
├── MultiSequenceForceDynamics.ipynb    # Ariani repetition experiment
├── MultiSequenceForceDynamics.md       # Study description
│
├── SingleSequenceForceDynamics.ipynb   # Wiestler transfer experiment
├── SingleSequenceForceDynamics.md      # Study description
│
├── RewardManipulation.ipynb            # Reward manipulation experiment
└── RewardManipulation.md               # Study description
```

---

## 🧪 Included Studies

### 1. Ariani Multi-Sequence Repetition Experiment
**File**: `MultiSequenceForceDynamics.ipynb`

Studies how force patterns change across different sequences when subjects practice multiple repetitions in quick succession (within-bout repetition).

**Data**: N=38 subjects, 2 days, 8 sequences, multiple repetitions per block

---

### 2. Single-Sequence Learning Experiments
**File**: `SingleSequenceForceDynamics.ipynb`

This notebook contains two related experiments.

### 2a. Wiestler Transfer Experiment

Investigates transfer learning for a single sequence across three days.

**Data**: N=36 subjects, 3 days of training

### 2b. Speed Clamping Experiment

Studies a common sequence trained across groups with a speed-clamp manipulation.

**Experimental Conditions**:
- Unconstrained: normal training
- Speed Clamped: execution speed held constant during induction
- Explicit + general: random-sequence practice plus explicit recall of the original sequence

**Data**: Two days of training on one sequence, with pre-test, induction, post-test, and post-test clamp phases

---

### 3. Reward Manipulation Experiment
**File**: `RewardManipulation.ipynb`

Examines how external reward (points) based on execution time percentiles affects trial-by-trial force pattern dynamics.

**Experimental Design**:
- Participants earn points based on execution time percentile
- Zone 4 (top 10% fast): 3 points
- Zone 3 (10-45% percentile): 1 or 3 points (random)
- Zone 2 (45-80% percentile): 0 or 1 point (random)
- Zone 1 (slowest): 0 points

**Data**: N=11 subjects, 4 days, one 7-keystroke sequence

---

## 🚀 Getting Started

### Requirements
```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
pingouin
statsmodels
tqdm
```

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn pingouin statsmodels tqdm
```

### Running the Notebooks
```bash
# Activate environment
source venv/bin/activate

# Start Jupyter
jupyter notebook

# Open desired notebook:
# - MultiSequenceForceDynamics.ipynb
# - SingleSequenceForceDynamics.ipynb
# - RewardManipulation.ipynb
```

**Note**: Data files are not included in the repository. Please contact the authors for data access.

---

## 📝 Citation

If you use these analyses in your work, please cite the original data sources and publications referenced in each notebook.

---

## 📧 Contact

For questions about the analyses or to request data access, please contact the authors.

---