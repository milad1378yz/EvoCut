import os
import pickle
import logging
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
import random

# logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# helper functions
def save_to_pickle(data: List[Dict], output_path: str) -> None:
    """Save a list of dictionaries to *output_path* using pickle."""
    try:
        logging.info(f"Saving {len(data)} instance(s) to '{output_path}' …")
        with open(output_path, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info("Done.")
    except IOError as err:
        logging.error(f"Could not save to '{output_path}': {err}")
        raise


#   Single‑instance generator
def gen_single_mclp_instance(
    n_demand: int,
    n_sites: int,
    p_budget: int,
    radius: float,
    rng: np.random.Generator,
) -> Dict:
    """Create **one** random MCLP instance

    A smaller coverage radius and a very small facility budget make the
    resulting instances much harder: many demand nodes compete for very
    few facilities, creating highly fractional LP optima and forcing the
    solver to perform deeper branch-and-bound searches.
    """
    # Random 2‑D coordinates in the unit square (vectorised for speed)
    coords_d = rng.random((n_demand, 2))  # |I| × 2
    coords_s = rng.random((n_sites, 2))  # |J| × 2

    # Population weights a_i ∈ {50, …, 500}
    a = {i + 1: int(rng.integers(50, 501)) for i in range(n_demand)}

    # Build coverage sets N_i
    Ni: Dict[int, List[int]] = {}
    rad2 = radius**2
    for i, d_pt in enumerate(coords_d, start=1):
        sq_dists = np.sum((coords_s - d_pt) ** 2, axis=1)
        coverers = np.nonzero(sq_dists <= rad2)[0] + 1  # 1‑based IDs
        # Ensure every demand node has at least one potential coverer
        if coverers.size == 0:
            coverers = np.array([rng.integers(n_sites) + 1])
        Ni[i] = coverers.tolist()

    I = list(range(1, n_demand + 1))
    J = list(range(1, n_sites + 1))

    return {
        "args": {"I": I, "J": J, "Ni": Ni, "a": a, "P": p_budget},
        "meta": {
            "n_demand": n_demand,
            "n_sites": n_sites,
            "radius": radius,
        },
    }


#   Batch generator  produces a list of hard instances


def generate_random_mclp_problems(
    num_problems: int = 10,
    demand_range: Tuple[int, int] = (4000, 8000),
    site_range: Tuple[int, int] = (6000, 12000),
    p_frac: float = 0.01,  # open ≈1 % of sites
    radius_range: Tuple[float, float] = (0.015, 0.03),
    base_seed: int = 42,
) -> List[Dict]:
    """Generate *num_problems* difficult random MCLP instances.

    * **Large |I| / |J|** → size alone stresses memory & presolve.
    * **Tiny facility budget (P) and small radius** → highly fractional LP
      relaxation ⇒ deeper B&B trees.
    """
    problems: List[Dict] = []
    for k in tqdm(range(num_problems), desc="Generating MCLP problems"):
        rng = np.random.default_rng(base_seed + k)

        n_demand = rng.integers(*demand_range)
        n_sites = rng.integers(*site_range)
        P = max(1, int(np.ceil(p_frac * n_sites)))
        radius = rng.uniform(*radius_range)

        inst = gen_single_mclp_instance(
            n_demand=n_demand,
            n_sites=n_sites,
            p_budget=P,
            radius=radius,
            rng=rng,
        )
        inst["name"] = f"hard_mclp_{n_demand}x{n_sites}_P{P}_r{radius:.3f}"
        problems.append(inst)

    return problems


#   Main script


def main() -> None:
    OUTPUT_PATH = "data/mclp.pkl"
    NUM_PROBLEMS = 20
    VALID_SPLIT = 0.2  # 20 % hold‑out for validation

    random.seed(42)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    logging.info("Starting hard‑instance generation …")

    mclp_problems = generate_random_mclp_problems(
        num_problems=NUM_PROBLEMS,
        demand_range=(4000, 8000),  # |I| 4 000 – 8 000
        site_range=(1000, 4000),  # |J| 6 000 – 12 000
        p_frac=0.01,  # ~1 % of sites can open
        radius_range=(0.015, 0.09),  # very local coverage
        base_seed=42,
    )

    # Sort instances by |I| for convenience
    mclp_problems.sort(key=lambda d: d["meta"]["n_demand"])
    logging.info(f"Total instances generated: {len(mclp_problems)}")

    # Train/validation split
    val_count = int(len(mclp_problems) * VALID_SPLIT)
    train_problems = mclp_problems[:-val_count] if val_count else mclp_problems
    val_problems = mclp_problems[-val_count:] if val_count else []

    logging.info(f"Training instances   : {len(train_problems)}")
    logging.info(f"Validation instances : {len(val_problems)}")

    # Save pickles
    save_to_pickle(train_problems, OUTPUT_PATH)
    base_name = os.path.splitext(os.path.basename(OUTPUT_PATH))[0]
    valid_path = os.path.join(os.path.dirname(OUTPUT_PATH), base_name + "_val.pkl")
    if val_problems:
        save_to_pickle(val_problems, valid_path)


if __name__ == "__main__":
    main()
