# EvoCut

**EvoCut** is a Python library that enhances Mixed-Integer Linear Programming (MILP) solving by injecting valid inequalities (cuts) into the model, effectively reducing the number of explored nodes and improving solver efficiency.

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
````

> **Note**: Python 3.9 is recommended.
> You must also install [Gurobi](https://www.gurobi.com/) and ensure it is properly licensed on your system.

### Step 2: Configure Settings

In the `configs/` directory:

* Copy each file that includes `_template` in its name.
* Remove `_template` from the filename.
* Fill in your credentials and modify hyperparameters or prompts as needed.

## Data Preparation

To download and preprocess data, run:

```bash
python data/data_prepare<suffix>.py
```

Replace `<suffix>` with the appropriate suffix found in the `data/` directory (e.g., options containing `rand` or other indicators).

## Usage

To run EvoCut for a specific problem:

```bash
python src/main.py <args>
```

Replace `<args>` with your desired configuration and options.

### Validate Cuts

To validate the performance of the generated cuts:

```bash
python experiments/validate_cuts.py <args>
```

### Evaluate on Test Data

To evaluate EvoCut on test data:

```bash
python experiments/evaluate_cut.py <args>
```