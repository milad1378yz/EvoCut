import os
import argparse
import logging
import warnings
import requests
from tqdm import tqdm

# Local module imports
from classes.llm_handler import LlmHandler
from classes.cut_verifier import CutVerifier
from classes.solver import Solver
from classes.genetic_algorithm import GeneticAlgorithm
from utils.general_utils import (
    load_config,
    save_pickle,
    load_pickle,
    remove_constraint,
    make_lp_relax_model,
)

# Environment and Logging setup
os.environ["CURL_CA_BUNDLE"] = ""
logging.basicConfig(level=logging.INFO)
logging.getLogger("gurobipy.gurobipy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

session = requests.Session()
session.verify = False

GAP_LOWERBOUND = 0.2
GAP_UPPERBOUND = 50.0


def arg_parser():
    parser = argparse.ArgumentParser(description="LLM4Cuts Argument Parser")
    parser.add_argument(
        "--general_configs_path",
        type=str,
        default="configs/general_configs.yaml",
        help="Path to the general config file",
    )
    parser.add_argument(
        "--api_tokens_path",
        type=str,
        default="configs/api_tokens.yaml",
        help="Path to the API tokens configuration file",
    )
    parser.add_argument(
        "--general_prompts_path",
        type=str,
        default="configs/prompts/general.yaml",
        help="Path to the general prompts for LLMs",
    )
    parser.add_argument(
        "--mutations_prompts_path",
        type=str,
        default="configs/prompts/mutations.yaml",
        help="Path to the mutations prompts for LLMs",
    )
    parser.add_argument(
        "--cross_overs_prompts_path",
        type=str,
        default="configs/prompts/cross_overs.yaml",
        help="Path to the cross overs prompts for LLMs",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        help="Path to the problems file",
    )
    parser.add_argument(
        "--problem_name",
        type=str,
        default="tsp",  # e.g. tsp, mcnd, "jssp", "cwlp"
        help="The name of the problem to solve",
    )
    parser.add_argument(
        "--num_evaluation",
        type=int,
        default=8,
        help="Number of evaluation instances (quick solve only)",
    )
    parser.add_argument(
        "--num_verification",
        type=int,
        default=2,
        help="Number of verification instances (full solve)",
    )
    return parser.parse_args()


def preload_verification_from_cache(dataset, cache_dir, need) -> list:
    """
    Preload verification data from cache.
    Args:
        dataset (list): List of problem instances.
        cache_dir (str): Directory where cached results are stored.
        need (int): Number of items to preload.
    Returns:
        list: List of problem instances with preloaded verification data.
    """
    verification = []
    for data in tqdm(dataset, desc="Preloading verification from cache"):
        cache_path = os.path.join(cache_dir, f"{data['name']}.pkl")
        if not os.path.exists(cache_path):
            continue
        cache = load_pickle(cache_path)
        if "true" in cache and "true_r" in cache:
            data["true"] = cache["true"]
            data["true_r"] = cache["true_r"]
            data["true_score"] = cache["true_score"]
            if (
                data["true_score"]["termination_condition"] in ["feasible", "maxTimeLimit"]
                and "gap" in data["true_score"]["solver_report"]
                and float(data["true_score"]["solver_report"]["gap"]) > GAP_LOWERBOUND
                and data["true"]["termination_condition"] == "optimal"
            ):
                verification.append(data)
            if len(verification) >= need:
                break
    return verification


def process_data_entry(
    data: dict,
    solver: Solver,
    true_code_clean: str,
    time_limit: int,
    max_time_out: int,
    cache_dir: str,
    full_solve: bool = False,
    pool_solution: int = 1,
) -> dict:
    """
    Process a single data entry, performing quick and heavy solves as needed.
    Args:
        data (dict): The data entry to process.
        solver (Solver): The solver instance.
        true_code_clean (str): The cleaned problem code.
        time_limit (int): Time limit for the solver.
        max_time_out (int): Maximum time out for the solver.
        cache_dir (str): Directory to store cached results.
        full_solve (bool): Whether to perform a full solve or not.
    Returns:
        dict: The processed data entry with results.
    """
    name = data["name"]
    cache_path = os.path.join(cache_dir, f"{name}.pkl")
    cache = load_pickle(cache_path) if os.path.exists(cache_path) else {}

    # QUICK
    if "true_score" not in cache:
        quick_model = solver.get_model_from_text("create_model", true_code_clean, **data["args"])
        cache["true_score"] = solver.solve_from_model(model=quick_model, time_limit=time_limit)
        logging.info("-" * 100)
        logging.info(
            f"[{name}] Quick solve termination: {cache['true_score']['termination_condition']}"
        )
        save_pickle(cache, cache_path)

    data["true_score"] = cache["true_score"]

    # HEAVY + RELAXED (only if needed for verification set)
    if full_solve:
        if "true" not in cache:
            heavy_model = solver.get_model_from_text(
                "create_model", true_code_clean, **data["args"]
            )
            cache["true"] = solver.solve_from_model(
                model=heavy_model,
                time_limit=max_time_out,
                pool_solution=pool_solution,
                use_work_limit=False,
            )
            # relaxed depends on heavy-model structure, so create after heavy
            logging.info("-" * 100)
            logging.info(
                f"[{name}] Heavy solve termination: {cache['true']['termination_condition']}"
            )
            relaxed_model = make_lp_relax_model(model=heavy_model)
            cache["true_r"] = solver.solve_from_model(model=relaxed_model, time_limit=max_time_out)
            save_pickle(cache, cache_path)

        data["true"] = cache["true"]
        data["true_r"] = cache["true_r"]

    return data


def main(args):
    # Load configuration files
    general_config = load_config(args.general_configs_path)
    api_tokens = load_config(args.api_tokens_path)

    # Determine dataset path and load dataset
    dataset_path = args.dataset_path or os.path.join("data", f"{args.problem_name}.pkl")
    dataset = load_pickle(dataset_path)
    logging.info(f"Number of raw data items: {len(dataset)}")

    # Load prompt files
    general_prompts = load_config(args.general_prompts_path)
    cross_overs_prompts = load_config(args.cross_overs_prompts_path)
    mutations_prompts = load_config(args.mutations_prompts_path)
    problem_prompts_path = os.path.join("configs", "prompts", f"{args.problem_name}.yaml")
    problem_prompts = load_config(problem_prompts_path)

    # LLM initialization
    if general_config["llm"]["backend"] == "hf":
        llm_handler = LlmHandler(
            backend="hf",
            api_key=api_tokens["huggingface"],
            **general_config["llm"]["hf_args"],
        )
    elif general_config["llm"]["backend"] == "openai":
        llm_handler = LlmHandler(
            backend="openai",
            api_key=api_tokens["openai"],
            **general_config["llm"]["openai_args"],
        )
    else:
        raise ValueError(f"Unsupported LLM Backend: {general_config['llm']['backend']}")

    # Prepare the problem code (cleaned)
    true_code = problem_prompts[args.problem_name]
    true_code_clean = remove_constraint(true_code)

    # Solver initialization
    max_time_out = general_config["solver"]["max_time_out"]
    time_limit = general_config["solver"]["time_limit"]
    solver = Solver(
        solver_name=general_config["solver"]["solver_name"],
        time_limit=time_limit,
        lazy_value=general_config["solver"]["lazy_value"],
        apply_lazy=general_config["solver"]["apply_lazy"],
    )

    cache_dir = os.path.join("cache", args.problem_name)
    os.makedirs(cache_dir, exist_ok=True)

    # desired set sizes
    need_eval = args.num_evaluation
    need_verif = args.num_verification

    # pre-load verification from cache
    verification_dataset = preload_verification_from_cache(dataset, cache_dir, need_verif)
    logging.info(
        f"Loaded {len(verification_dataset)}/{need_verif} verification instances from cache"
    )

    evaluation_dataset = []
    solver_reports_eval = []

    # Process each problem instance in the dataset
    for data in tqdm(dataset, desc="Processing dataset entries"):
        name = data.get("name", "unnamed")
        if len(evaluation_dataset) >= need_eval and len(verification_dataset) >= need_verif:
            break  # both sets are full

        # Decide whether a full solve is required *for this entry*
        need_more_verif = len(verification_dataset) < need_verif
        processed_data = process_data_entry(
            data=data,
            solver=solver,
            true_code_clean=true_code_clean,
            time_limit=time_limit,
            max_time_out=max_time_out,
            cache_dir=cache_dir,
            full_solve=need_more_verif,
            pool_solution=general_config["solver"]["pool_solutions"],
        )

        logging.info(
            f"[{name}] Quick solve termination: {processed_data['true_score']['termination_condition']}"
        )

        # evaluation-set filter (quick solve only)
        quick_report = processed_data["true_score"]["solver_report"]

        # Qualification for evaluation set (quick solve only)
        qual_eval = (
            processed_data["true_score"]["termination_condition"] in ["feasible", "maxTimeLimit"]
            and "gap" in quick_report
            and GAP_UPPERBOUND > float(quick_report["gap"]) > GAP_LOWERBOUND
        )

        # Qualification for verification set (needs full results & optimal heavy solve)
        qual_verif = (
            need_more_verif
            and qual_eval
            and "true" in processed_data
            and processed_data["true"]["termination_condition"] == "optimal"
        )

        # add to datasets, guaranteeing DISJOINT sets
        if qual_verif and len(verification_dataset) < need_verif:
            verification_dataset.append(processed_data)
            logging.info(f"[{name}] added to verification set")

        if qual_eval and len(evaluation_dataset) < need_eval:
            evaluation_dataset.append(processed_data)
            solver_reports_eval.append(quick_report)
            logging.info(f"[{name}] added to evaluation set")

    logging.info(
        f"#evaluation = {len(evaluation_dataset)}, #verification = {len(verification_dataset)}"
    )
    if verification_dataset:
        times_str = ", ".join(
            str(data["true"]["solver_report"]["total_time"]) for data in verification_dataset
        )
        logging.info(f"Verification solve times: {times_str}")

    # Calculate initial score for the filtered dataset.
    score_init = CutVerifier.cal_fitness(
        solver_reports=solver_reports_eval,
        solver_reports_ref=solver_reports_eval,
        time_limit=time_limit,
        solver_name=general_config["solver"]["solver_name"],
    )
    logging.info(f"Score without additional cut: {score_init}")

    # Initialize and run Genetic Algorithm
    genetic_algorithm = GeneticAlgorithm(
        solver=solver,
        evaluation_dataset=evaluation_dataset,
        verification_dataset=verification_dataset,
        number_population=general_config["genetic_algorithm"]["number_population"],
        time_limit=time_limit,
        population_file=f"{args.problem_name}.json",
        max_llm_attempts=general_config["genetic_algorithm"]["max_llm_attempts"],
    )

    genetic_algorithm.initialize_population(
        llm_handler=llm_handler,
        prompt_initializer=general_prompts["initializer"],
        true_code=true_code,
        model_name=general_config["llm"]["model_name"],
    )

    for i in range(general_config["genetic_algorithm"]["num_steps_ga"]):
        logging.info(f"Running Genetic Algorithm step: {i}")
        genetic_algorithm.step(
            cross_overs_prompts=cross_overs_prompts,
            mutations_prompts=mutations_prompts,
            llm_handler=llm_handler,
            model_name=general_config["llm"]["model_name"],
            true_code=true_code,
            mutation_rate=general_config["genetic_algorithm"]["mutation_rate"],
            elitism_ratio=general_config["genetic_algorithm"]["elitism_ratio"],
            cross_over_ratio=general_config["genetic_algorithm"]["cross_over_ratio"],
        )

    logging.info("Code Completed")


if __name__ == "__main__":
    args = arg_parser()
    main(args)
