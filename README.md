
# EvoCut
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](./requirements.txt)
[![Stars](https://img.shields.io/github/stars/milad1378yz/EvoCut?style=social)](https://github.com/milad1378yz/EvoCut/stargazers)
![Issues](https://img.shields.io/github/issues/milad1378yz/EvoCut)
![Repo Size](https://img.shields.io/github/repo-size/milad1378yz/EvoCut)
![Top Language](https://img.shields.io/github/languages/top/milad1378yz/EvoCut)

**EvoCut** is a Python library that accelerates Mixed-Integer Linear Programming (MILP) by injecting problem-specific cutting planes into the LP relaxation. These cuts reduce the feasible set of the LP relaxation and improve solver efficiency.

---
## Appendix Result: IMO 2025 Problem 6 (P6)

Rectangular tiling with one hole per row and column (IMO 2025 P6). We evaluate EvoCut on a compact 2D interval‑flow MILP for this problem as described in the paper appendix.

- Benchmark sizes: `N ∈ {4, 9, 16, 25}` (Appendix E.4/F.5)
- Baseline MILP: `RT-2DFlow`; Augmented with EvoCut family: `EC-RT-Breaks`
- Budget/metric: 10,000s wall‑clock; MIPGap‑time trajectories (Figure 10)
- Outcome: EvoCut consistently lowers the gap curves across all four sizes, reaching earlier plateaus and maintaining lower gaps than the baseline throughout the time budget (see Figure 10 in the appendix).
- Implication: Demonstrates EvoCut's applicability to new, previously unseen problems for LLMs, the IMO 2025 P6 tiling task sits outside standard MILP benchmarks, indicating the method does not rely on prior exposure to that specific problem.

Notes
- Ground‑truth tiling counts from the problem discussion: `N=4 --> 5 tiles` and `N=9 --> 12 tiles`.
- Full experiment details, model, and trajectories are in the paper appendix (see "Appendix F.5" and "Figure 10").
---
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



