import argparse
import copy
import json
import logging
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
from gurobipy import GRB
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classes.population import Population
from src.utils.general_utils import (
    adopt_code_to_fucntion,
    load_config,
    load_pickle,
    remove_constraint,
    find_main_constraints,
)

# Environment & logging setup
os.environ["CURL_CA_BUNDLE"] = ""

logging.basicConfig(level=logging.INFO)
logging.getLogger("gurobipy.gurobipy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


# Argument parsing
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Validate EVOcuts on a new dataset (quick-success count only).",
    )
    parser.add_argument("--solver_name", type=str, default="gurobi_persistent")
    parser.add_argument("--problem_name", type=str, default="tsp")
    parser.add_argument("--population_file", type=str, default="tsp.json")
    parser.add_argument("--time_limit", type=int, default=300)

    return parser.parse_args()


# Utility helpers
def get_model_from_text(function_name: str, function_string: str, **kwargs):
    """Dynamically execute *function_string* and return the resulting model."""

    exec(function_string, globals())
    function = globals()[function_name]
    return function(**kwargs)


def get_added_cut_name(model_base: pyo.AbstractModel, model_cut: pyo.AbstractModel) -> str:
    """
    Get the name of the added cut in the model.
    Args:
        model_base: The base model.
        model_cut: The model with the added cut.
    Returns:
        str: The name of the added cut.
    """
    # Find the main constraints in the base model
    main_constraints = find_main_constraints(model_base)

    # Find the added cut in the cut model
    for constr in main_constraints:
        if constr not in model_cut.component_objects(pyo.Constraint, active=True):
            return constr
    return None


# Callback
def my_callback(cb_model, cb_opt, cb_where, is_before_cut: bool = True):
    """Collect incumbent solution info whenever Gurobi finds a new MIP solution."""

    global last_time, start_time
    global Times_before_cut, Gaps_before_cut, Nodes_before_cut, Bounds_before_cut
    global Times_after_cut, Gaps_after_cut, Nodes_after_cut, Bounds_after_cut

    if cb_where == GRB.Callback.MIPSOL:
        current_time = time.time()
        if current_time - last_time >= 0:
            last_time = current_time

            obj_val = cb_opt.cbGet(GRB.Callback.MIPSOL_OBJ)
            obj_bnd = cb_opt.cbGet(GRB.Callback.MIPSOL_OBJBND)
            node_cnt = cb_opt.cbGet(GRB.Callback.MIPSOL_NODCNT)
            mip_gap = (obj_val - obj_bnd) / (obj_val + 1e-10)

            if mip_gap > 100:
                return

            if is_before_cut:
                Gaps_before_cut.append(mip_gap)
                Nodes_before_cut.append(node_cnt)
                Bounds_before_cut.append(obj_bnd)
                Times_before_cut.append(current_time - start_time)
            else:
                Gaps_after_cut.append(mip_gap)
                Nodes_after_cut.append(node_cnt)
                Bounds_after_cut.append(obj_bnd)
                Times_after_cut.append(current_time - start_time)


def main():
    """Run validation over the dataset."""
    args = parse_args()

    file_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(file_dir, "..", "data", f"{args.problem_name}_val.pkl")
    dataset = load_pickle(dataset_path)

    prompts_path = os.path.join(file_dir, "..", "configs", "prompts", f"{args.problem_name}.yaml")
    problem_prompts = load_config(prompts_path)

    logging.info(f"Number of datasets: {len(dataset)}")

    true_code = problem_prompts[args.problem_name]
    true_code_clean = remove_constraint(true_code)

    population = Population(json_path=args.population_file)
    population.load_from_json()
    all_indivs = population.get_all_indivs()
    if not all_indivs:
        logging.error("No individuals found in the population JSON.")
        sys.exit(1)

    top_individual = max(all_indivs, key=lambda ind: ind.fitness)
    full_code = adopt_code_to_fucntion(
        copy.deepcopy(true_code), top_individual.chromosome["added_cut"]
    )

    for data in tqdm(dataset, desc="Processing datasets"):
        name = data["name"]

        if os.path.exists(f"{args.problem_name}/{name}_1.json"):
            logging.info(f"Passing {name}")
            continue

        logging.info(f"Running {name}")

        # BEFORE valid cut
        global Times_before_cut, Gaps_before_cut, Nodes_before_cut, Bounds_before_cut
        Times_before_cut, Gaps_before_cut, Nodes_before_cut, Bounds_before_cut = [], [], [], []

        solver = SolverFactory(args.solver_name, solver_io="python", manage_env=True)
        solver.options.update({"Seed": 14, "Method": 4, "TimeLimit": args.time_limit})
        solver.set_callback(
            lambda cb_model, cb_opt, cb_where: my_callback(
                cb_model, cb_opt, cb_where, is_before_cut=True
            )
        )

        model_base = get_model_from_text("create_model", true_code_clean, **data["args"])
        solver.set_instance(model_base)

        global start_time, last_time
        start_time = last_time = time.time()
        solver.solve(tee=True)

        # Append final metrics
        gm = solver._solver_model
        final_gap_b = (gm.ObjVal - gm.ObjBound) / (gm.ObjVal + 1e-10)
        final_nodes_b = gm.NodeCount
        final_bound_b = gm.ObjBound

        # Persist final metrics
        Gaps_before_cut.append(final_gap_b)
        Nodes_before_cut.append(final_nodes_b)
        Bounds_before_cut.append(final_bound_b)
        Times_before_cut.append(time.time() - start_time)

        Times_b, Gaps_b, Nodes_b, Bounds_b = (
            copy.deepcopy(Times_before_cut),
            copy.deepcopy(Gaps_before_cut),
            copy.deepcopy(Nodes_before_cut),
            copy.deepcopy(Bounds_before_cut),
        )

        # AFTER valid cut
        for lazy_num in [-1, 0, 1, 2, 3]:
            if os.path.exists(f"{args.problem_name}/{name}_{lazy_num}.json"):
                continue

            global Times_after_cut, Gaps_after_cut, Nodes_after_cut, Bounds_after_cut
            Times_after_cut, Gaps_after_cut, Nodes_after_cut, Bounds_after_cut = [], [], [], []

            solver = SolverFactory("gurobi_persistent", solver_io="python", manage_env=True)
            solver.options.update({"Seed": 14, "Method": 4, "TimeLimit": args.time_limit})
            solver.set_callback(
                lambda cb_model, cb_opt, cb_where: my_callback(
                    cb_model, cb_opt, cb_where, is_before_cut=False
                )
            )

            model_cut = get_model_from_text("create_model", full_code, **data["args"])
            solver.set_instance(model_cut)

            cut_name = get_added_cut_name(model_base, model_cut)

            if hasattr(model_cut, cut_name):
                cut = getattr(model_cut, cut_name)
                for idx in cut:
                    solver.set_linear_constraint_attr(
                        cut[idx],
                        "Lazy",
                        lazy_num,
                    )

            start_time = last_time = time.time()
            solver.solve(tee=True)

            gm = solver._solver_model
            final_gap_a = (gm.ObjVal - gm.ObjBound) / (gm.ObjVal + 1e-10)
            final_nodes_a = gm.NodeCount
            final_bound_a = gm.ObjBound

            Gaps_after_cut.append(final_gap_a)
            Nodes_after_cut.append(final_nodes_a)
            Bounds_after_cut.append(final_bound_a)
            Times_after_cut.append(time.time() - start_time)

            # Plotting
            os.makedirs(f"{args.problem_name}", exist_ok=True)

            # Gap-vs-Time
            plt.figure()
            plt.plot(Times_b, Gaps_b, label="Without Valid Cut")
            plt.plot(Times_after_cut, Gaps_after_cut, label=f"With Valid Cut (Lazy={lazy_num})")
            plt.xlabel("Time (s)")
            plt.ylabel("Relative MIP Gap")
            plt.title("MIP Gap vs Time")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{args.problem_name}/{name}_{lazy_num}_gap.png")
            plt.close()

            # Nodes-vs-Time
            plt.figure()
            plt.step(Times_b, Nodes_b, where="post", label="Without Valid Cut")
            plt.step(
                Times_after_cut,
                Nodes_after_cut,
                where="post",
                label=f"With Valid Cut (Lazy={lazy_num})",
            )
            plt.xlabel("Time (s)")
            plt.ylabel("# Branch-and-Bound Nodes")
            plt.title("Node Count vs Time")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{args.problem_name}/{name}_{lazy_num}_nodes.png")
            plt.close()

            # Best-Bound-vs-Time
            plt.figure()
            plt.plot(Times_b, Bounds_b, label="Without Valid Cut")
            plt.plot(
                Times_after_cut,
                Bounds_after_cut,
                label=f"With Valid Cut (Lazy={lazy_num})",
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Best Bound on Cmax")
            plt.title("Best Bound vs Time")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{args.problem_name}/{name}_{lazy_num}_bound.png")
            plt.close()

            # Persist results
            with open(f"{args.problem_name}/{name}_{lazy_num}.json", "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "Times_before_cut": Times_b,
                        "Gaps_before_cut": Gaps_b,
                        "Nodes_before_cut": Nodes_b,
                        "Bounds_before_cut": Bounds_b,
                        "Times_after_cut": Times_after_cut,
                        "Gaps_after_cut": Gaps_after_cut,
                        "Nodes_after_cut": Nodes_after_cut,
                        "Bounds_after_cut": Bounds_after_cut,
                    },
                    fp,
                    indent=4,
                )


if __name__ == "__main__":
    main()
