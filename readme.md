# Learning Dynamics in Sequence Learning

A collection of analyses investigating how motor learning unfolds during sequence learning tasks. This repository contains three independent studies analyzing force pattern dynamics, execution time improvements, and the effects of reward manipulation on motor learning.

## 📚 Project Overview

This repository contains Jupyter notebooks and supporting utilities for analyzing trial-by-trial force pattern dynamics in motor sequence learning. Each study employs different experimental manipulations to understand the mechanisms underlying skill acquisition.

### Key Analyses
- **Force pattern distance calculations**: Euclidean distances between force vectors as a measure of motor pattern consistency
- **Execution time dynamics**: Speed improvements as a proxy for motor learning
- **Trial-by-trial learning effects**: How force patterns evolve within and across blocks
- **Transfer and generalization**: Understanding how learned patterns transfer under different conditions

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

**Key Findings**:
- Analysis of within-bout vs between-bout force pattern changes
- Force pattern stability increases with practice
- Different sequences show distinct learning trajectories

**Data**: N=38 subjects, 2 days, 8 sequences, multiple repetitions per block

---

### 2. Wiestler Single-Sequence Transfer Experiment
**File**: `SingleSequenceForceDynamics.ipynb`

Investigates long-term learning dynamics of a single sequence across three days, with a critical manipulation: speed clamping during blocks 5-12 on Day 1.

**Key Findings**:
- Speed-dependent vs speed-independent learning
- Within-subject force pattern changes during clamping
- Template-based learning: regression analysis reveals composition of learning from clamped experience and post-clamp changes

**Experimental Conditions**:
- Unclamped: normal practice
- Clamped: execution speed held constant, allowing isolated analysis of force pattern changes
- Verbal: explicit instruction group

**Data**: N=36 subjects, 3 days training, 2 days post-test blocks

---

### 3. Reward Manipulation Experiment
**File**: `RewardManipulation.ipynb`

Examines how external reward (points) based on execution time percentiles affects trial-by-trial force pattern dynamics.

**Key Findings**:
- Reward zones create distinct learning environments
- Force pattern consistency varies by reward condition
- Trial-by-trial dynamics reveal moment-to-moment motor adjustments

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