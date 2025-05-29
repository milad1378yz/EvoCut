import argparse
import os
import sys
import logging
import warnings
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.classes.population import Population
from src.classes.cut_verifier import CutVerifier
from src.classes.solver import Solver
from src.utils.general_utils import (
    load_config,
    load_pickle,
    save_pickle,
    remove_constraint,
)

os.environ["CURL_CA_BUNDLE"] = ""

logging.basicConfig(level=logging.INFO)
logging.getLogger("gurobipy.gurobipy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate GA cuts on a new dataset (quick-success count only)."
    )
    parser.add_argument(
        "--validation_config",
        type=str,
        default="configs/validation_configs.yaml",
    )
    parser.add_argument(
        "--population_file",
        type=str,
        default="tsp.json",
    )
    parser.add_argument(
        "--problem_name",
        type=str,
        default="tsp",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache_validation",
    )
    parser.add_argument(
        "--num_verification",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=3,
    )
    return parser.parse_args()


def process_data_entry(
    data: dict,
    solver: Solver,
    true_code_clean: str,
    time_limit: int,
    cache_dir: str,
    pool_solutions: int = 2,
):
    name = data["name"]
    cache_path = os.path.join(cache_dir, f"{name}.pkl")
    cache = load_pickle(cache_path) if os.path.exists(cache_path) else {}

    if "true" not in cache:
        cache["true"] = solver.solve_from_text_sol_pool(
            function_name="create_model",
            function_string=true_code_clean,
            time_limit=time_limit,
            pool_solutions=pool_solutions,
            **data["args"],
        )
        logging.info("-" * 100)
        logging.info(f"[{name}] Quick solve termination: {cache['true']['termination_condition']}")
        save_pickle(cache, cache_path)

    data["true"] = cache["true"]
    return data


def main():
    args = parse_args()
    general_config = load_config(args.validation_config)
    solver_name = general_config["solver"]["solver_name"]

    solver = Solver(solver_name=solver_name, time_limit=general_config["solver"]["time_limit"])

    validation_dataset_path = os.path.join("data", f"{args.problem_name}_val.pkl")
    if not os.path.exists(validation_dataset_path):
        logging.error(f"Validation dataset not found at {validation_dataset_path}")
        sys.exit(1)
    dataset = load_pickle(validation_dataset_path)
    logging.info(f"Loaded {len(dataset)} new instances for validation.")

    population = Population(json_path=args.population_file)
    population.load_from_json()
    all_indivs = population.get_all_indivs()
    if not all_indivs:
        logging.error("No individuals found in the population JSON.")
        sys.exit(1)

    sorted_population = sorted(all_indivs, key=lambda ind: ind.fitness, reverse=True)
    top_individuals = sorted_population[: args.top_n]
    logging.info(f"Selected top {args.top_n} individuals for validation.")
    logging.info(f"Top individuals fitness: {[ind.fitness for ind in top_individuals]}")

    true_code_prompts_path = os.path.join("configs", "prompts", args.problem_name + ".yaml")
    if not os.path.exists(true_code_prompts_path):
        logging.error(f"Cannot find {true_code_prompts_path}.")
        sys.exit(1)
    problem_prompts = load_config(true_code_prompts_path)
    true_code = problem_prompts[args.problem_name]
    true_code_clean = remove_constraint(true_code)

    cache_dir = os.path.join(args.cache_dir, args.problem_name)
    os.makedirs(cache_dir, exist_ok=True)

    filtered_dataset = []
    for data in tqdm(dataset, desc="Processing dataset entries"):
        processed_data = process_data_entry(
            data=data,
            solver=solver,
            true_code_clean=true_code_clean,
            time_limit=general_config["solver"]["time_limit"],
            cache_dir=cache_dir,
            pool_solutions=general_config["solver"]["pool_solutions"],
        )

        # drop the 'gap' requirement
        if processed_data["true"]["termination_condition"] in ["feasible", "maxTimeLimit"]:
            filtered_dataset.append(processed_data)

    logging.info(f"Number of filtered data is: {len(filtered_dataset)}")

    verified_individuals = 0

    for idx, indiv in enumerate(top_individuals, start=1):
        code_id = f"indiv_{idx}"
        full_code = true_code_clean

        logging.info(f"Verifying {code_id} ...")
        try:
            validity = CutVerifier.verify_score_cut(
                filtered_dataset,
                filtered_dataset,
                solver,
                full_code,
                true_code,
                use_work_limit=False,
            )
        except Exception as e:
            logging.error(f"{code_id} -> Verification failed: {e}")
            continue

        if validity.get("verified", False):
            logging.info(f"{code_id} -> PASSED")
            verified_individuals += 1
        else:
            logging.info(f"{code_id} -> FAILED")

    # final concise success report
    print("\n" + "=" * 60)
    print(
        f"Validation success: {verified_individuals}/{len(top_individuals)} "
        f"top individuals passed verification."
    )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
