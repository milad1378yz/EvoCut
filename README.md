
# EvoCut
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/github/license/yourusername/evocut)
![Build](https://img.shields.io/github/actions/workflow/status/yourusername/evocut/ci.yml)
![Stars](https://img.shields.io/github/stars/yourusername/evocut?style=social)

**EvoCut** is a Python library that accelerates Mixed-Integer Linear Programming (MILP) by injecting problem-specific cutting planes into the LP relaxation. These cuts reduce the feasible set of the LP relaxation and improve solver efficiency.


## Installation

### 1) (Optional) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
````

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Python 3.9 recommended.
> Requires a licensed MILP solver (e.g., [Gurobi](https://www.gurobi.com/)).

---

## Configuration

In the [`configs/`](./configs) directory:

1. Copy each file whose name contains `_template`.
2. Rename the copy (remove `_template`).
3. Fill in credentials and hyperparameters as needed.

---

## Data Preparation

Use the relevant preprocessing script in `data/`:

```bash
python data/data_prepare<suffix>.py
```

Replace `<suffix>` with the correct option for your dataset variant (e.g., `rand`).

---

## Usage

Run EvoCut on a problem instance:

```bash
python src/main.py <args>
```

See all options with:

```bash
python src/main.py -h
```

---

## Verification of Cuts (OSP)

To check the **optimal solution preservation rate** of generated cuts:

```bash
python experiments/OSP_cuts.py <args>
```

---

## Evaluation on Test Data

Evaluate EvoCut on held-out instances:

```bash
python experiments/evaluate_cut.py <args>
```

---

## Reproducibility

* Determinism: seeds are configurable in configs or CLI flags.
* Hardware/solver versions may affect runtime but not correctness.
* Minimal dependencies ensure reproducibility across machines.



