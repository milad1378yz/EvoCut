import os
import pickle
import logging
import random
from typing import Dict, List, Tuple
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def save_to_pickle(data: List[Dict], output_path: str) -> None:
    """Save data to a pickle file."""
    try:
        logging.info(f"Saving data to {output_path}...")
        with open(output_path, "wb") as file:
            pickle.dump(data, file)
        logging.info("Data successfully saved.")
    except IOError as e:
        logging.error(f"Failed to save data: {e}")
        raise


def generate_random_bin_packing_problems(
    num_problems: int = 5,
    n_range: Tuple[int, int] = (10, 20),
    bin_dim: Tuple[float, float] = (15.0, 15.0),
    item_dim_range: Tuple[float, float] = (2.0, 10.0),
    random_seed: int = 42,
) -> List[Dict]:
    """
    Generate random 2D Bin Packing problem instances.

    Each instance contains:
      - "n": number of items.
      - "W": bin width.
      - "H": bin height.
      - "h": a dict mapping item index to its height (sorted in non-increasing order).
      - "w": a dict mapping item index to its width (corresponding order to heights).

    Parameters:
    -----------
    num_problems : int
        Number of problem instances to generate.
    n_range : Tuple[int, int]
        Range (inclusive) for number of items per instance.
    bin_dim : Tuple[float, float]
        Dimensions for the bin: (W, H).
    item_dim_range : Tuple[float, float]
        Lower and upper bounds for item dimensions (applied to both widths and heights).
    random_seed : int
        Random seed for reproducibility.

    Returns:
    --------
    List[Dict]
        List of randomly generated bin packing problem instances.
    """
    random.seed(random_seed)
    problems = []
    W, H = bin_dim

    for problem_idx in tqdm(range(num_problems), desc="Generating Bin Packing problems"):
        # Randomly choose the number of items for this problem instance
        n = random.randint(*n_range)

        items = []
        # Generate random dimensions for each item.
        # We ensure that each item is feasibly packable into the bin by keeping dimensions lower than bin dimensions.
        for _ in range(n):
            height_i = random.uniform(item_dim_range[0], min(item_dim_range[1], H - 1))
            width_i = random.uniform(item_dim_range[0], min(item_dim_range[1], W - 1))
            items.append((height_i, width_i))

        # Sort items by non-increasing height (as required by the formulation)
        items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
        h = {i + 1: items_sorted[i][0] for i in range(n)}
        w = {i + 1: items_sorted[i][1] for i in range(n)}

        problem_dict = {
            "name": f"random_bin_packing_{problem_idx}",
            "args": {
                "n": n,
                "W": W,
                "H": H,
                "h": h,
                "w": w,
            },
        }
        problems.append(problem_dict)

    return problems


def main():
    # Configuration
    OUTPUT_PATH = "data/bin_packing.pkl"
    NUM_PROBLEMS_TO_GENERATE = 8
    VALIDATION_SPLIT = 0.2  # fraction of instances kept for validation

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Generate random 2D bin packing problems
    logging.info("Starting random 2D Bin Packing problem generation...")
    bin_packing_problems = generate_random_bin_packing_problems(
        num_problems=NUM_PROBLEMS_TO_GENERATE,
        n_range=(10, 20),  # Each problem will have between 10 and 20 items
        bin_dim=(15.0, 15.0),  # Bin dimensions can be set as desired
        item_dim_range=(
            2.0,
            10.0,
        ),  # Items will have dimensions in this range, ensuring they fit into the bin.
        random_seed=42,
    )

    # Optionally sort them by number of items (or any other criterion)
    bin_packing_problems.sort(key=lambda x: x["args"]["n"])

    logging.info(f"Number of bin packing problems generated: {len(bin_packing_problems)}")

    # Split problems into training and validation sets
    total_problems = len(bin_packing_problems)
    validation_count = int(total_problems * VALIDATION_SPLIT)
    training_count = total_problems - validation_count

    training_problems = bin_packing_problems[:training_count]
    validation_problems = bin_packing_problems[training_count:]

    logging.info(f"Training problems: {len(training_problems)}")
    logging.info(f"Validation problems: {len(validation_problems)}")

    # Save training problems to the original output path
    save_to_pickle(training_problems, OUTPUT_PATH)

    # Save validation problems to a separate pickle file.
    base_filename = os.path.splitext(os.path.basename(OUTPUT_PATH))[0]
    valid_output_path = os.path.join(os.path.dirname(OUTPUT_PATH), base_filename + "_val.pkl")
    save_to_pickle(validation_problems, valid_output_path)


if __name__ == "__main__":
    main()
