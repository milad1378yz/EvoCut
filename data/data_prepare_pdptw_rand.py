import logging
import os
import pickle
import random
from typing import Dict, List

from src.utils.random_generators import generate_random_pdptw_instances


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def save_to_pickle(data: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("Saved %s", output_path)


def main() -> None:
    OUTPUT_PATH = "data/pdptw.pkl"
    NUM_INSTANCES = 50
    VALID_SPLIT = 0.2
    SEED = 42

    instances = generate_random_pdptw_instances(
        NUM_INSTANCES,
        seed=SEED,
        n_range=(10, 40),
        k_range=(2, 8),
    )

    random.seed(SEED)
    random.shuffle(instances)

    val_count = int(len(instances) * VALID_SPLIT)
    train_instances = instances[:-val_count] if val_count else instances
    val_instances = instances[-val_count:] if val_count else []

    save_to_pickle(train_instances, OUTPUT_PATH)
    if val_instances:
        base_name = os.path.splitext(os.path.basename(OUTPUT_PATH))[0]
        val_path = os.path.join(os.path.dirname(OUTPUT_PATH), base_name + "_val.pkl")
        save_to_pickle(val_instances, val_path)


if __name__ == "__main__":
    main()
