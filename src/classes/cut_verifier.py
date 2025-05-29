from typing import List, Dict, Any
import logging
from tqdm import tqdm
from .solver import Solver
import pyomo.environ as pyo
import os
import sys
import math

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.general_utils import make_lp_relax_model, remove_constraint, find_main_constraints


class CutVerifier:
    _cached_main_constraints: List[str] = None

    @staticmethod
    def cal_fitness(
        solver_reports: List[Dict[str, Any]],
        solver_reports_ref: List[Dict[str, Any]],
        time_limit: int = 10,
        solver_name: str = "gurobi",
    ) -> float:
        """
        Calculate the fitness of the individual based on the solver reports.
        Parameters:
            solver_reports: List of solver reports for each data point.
            solver_reports_ref: List of reference solver reports for each data point.
            time_limit: Time limit for the solver.
            solver_name: Name of the solver used.
        Returns:
            fitness: The fitness of the individual.
        """
        if solver_name == "appsi_highs":
            gaps = []
            times = []
            alpha = 0.9
            for report in solver_reports:
                primal_bound = float(report["Primal bound"])
                dual_bound = float(report["Dual bound"])
                total_time = float(report["Timing"]["total"])
                gap = (primal_bound - dual_bound) / primal_bound
                gaps.append(gap)
                times.append(total_time / time_limit)

            loss = alpha * (sum(gaps) / len(gaps)) + (1 - alpha) * (sum(times) / len(times))
            fitness = 1.0 / (loss + 1e-7)

            return fitness * 100
        elif solver_name in ["gurobi", "gurobi_persistent"]:
            alpha = 1.0
            rel_gaps = []
            rel_time = []
            counter_falshe = 0
            for ref_report, report in zip(solver_reports_ref, solver_reports):
                if not "gap" in report:
                    counter_falshe -= 1
                    continue
                rel_gaps.append(
                    (float(report["gap"]) - float(ref_report["gap"])) / float(ref_report["gap"])
                )
                rel_time.append(
                    (float(report["total_time"]) - float(ref_report["total_time"]))
                    / float(ref_report["total_time"])
                )
            avg_rel_time = sum(rel_time) / len(rel_time)
            avg_rel_gap = sum(rel_gaps) / len(rel_gaps)
            cost = alpha * avg_rel_gap + (1 - alpha) * avg_rel_time
            fitness = math.exp(-cost)
            return (fitness) * 10

        else:
            raise ValueError(f"Unsupported solver_name: {solver_name}")

    @staticmethod
    def solve_for_one(
        solver: Solver,
        data: dict,
        full_code: str,
        main_constraints: List[str],
        **solver_args,
    ) -> Dict[str, Any]:
        model = solver.get_model_from_text("create_model", full_code, **data["args"])
        results = solver.solve_from_model(
            model=model,
            main_constraints=main_constraints,
            **solver_args,
        )
        return results

    @staticmethod
    def verify_score_cut(
        verification_dataset: List[Dict[str, Any]],
        evaluation_dataset: List[Dict[str, Any]],
        solver: Solver,
        full_code: str,
        true_code: str,
        **solver_args,
    ) -> Dict[str, Any]:
        """
        Phase 1  (verification_dataset)   check validity & usefulness.
        Phase 2  (evaluation_dataset)    only if Phase 1 succeeds, collect solver-reports used for scoring.

        Heavy / relaxed solves are executed only for the verification set.
        The evaluation set is solved once with the cut (quick solve, time-limit
        already embedded in the Solver instance).

        Parameters:
            verification_dataset: Dataset for verification.
            evaluation_dataset: Dataset for evaluation.
            solver: Solver instance to use.
            full_code: Full code of the model with the cut.
            true_code: Full code of the true model.
            solver_args: Additional arguments for the solver.
        Returns:
            verified: True if the cut is valid and useful, False otherwise.
            solver_reports: List of solver reports for the evaluation dataset.
        """
        # PHASE 1 – VALIDATE
        is_valid = True
        is_useful = True

        for data in tqdm(verification_dataset, desc="Verifying cut on verification set"):
            result_true = data["true"]  # heavy baseline

            # cache the list of “main constraints” once
            if CutVerifier._cached_main_constraints is None:
                CutVerifier._cached_main_constraints = find_main_constraints(
                    model=solver.get_model_from_text(
                        "create_model",
                        remove_constraint(true_code),
                        **data["args"],
                    )
                )
                logging.info(f"Main constraints: {CutVerifier._cached_main_constraints}")

            # 1) feasibility with variables fixed to baseline tour
            if isinstance(result_true["model"], list):
                for model in result_true["model"]:
                    if not CutVerifier.check_feasibility_with_fixed_vars(
                        solver=solver,
                        model1=model,
                        model2=solver.get_model_from_text(
                            "create_model", full_code, **data["args"]
                        ),
                    ):
                        is_valid = False
                        break
                if not is_valid:
                    break

            else:
                if not CutVerifier.check_feasibility_with_fixed_vars(
                    solver=solver,
                    model1=result_true["model"],
                    model2=solver.get_model_from_text("create_model", full_code, **data["args"]),
                ):
                    is_valid = False
                    break

            # 2) usefulness check (LP-relaxation)
            if not is_useful and CutVerifier.check_usefullness_with_relaxarion(
                solver, full_code, true_code, data
            ):
                is_useful = True

        verified = is_valid and is_useful
        if not verified:
            return {"verified": False, "solver_reports": []}

        # PHASE 2 – EVALUATE
        solver_reports: List[Dict[str, Any]] = []
        for data in tqdm(evaluation_dataset, desc="Calculating score on evaluation set"):
            # quick solve of the cut-augmented model
            try:
                result_cut = CutVerifier.solve_for_one(
                    solver=solver,
                    data=data,
                    full_code=full_code,
                    main_constraints=CutVerifier._cached_main_constraints,
                    **solver_args,
                )
                if "gap" not in result_cut["solver_report"]:
                    logging.warning(
                        f"[evaluation] solver report lacks 'gap': {result_cut['solver_report']}"
                    )
                solver_reports.append(result_cut["solver_report"])
            except Exception as e:
                is_valid = False
                logging.warning(
                    f"[evaluation] instance skipped due to an error: {e} verification failed"
                )
                break

        if all("gap" not in report for report in solver_reports):
            is_valid = False
            logging.warning(
                f"[evaluation] all instances could not met the gap: verification failed"
            )

        return {"verified": is_valid, "solver_reports": solver_reports}

    @staticmethod
    def check_feasibility_with_fixed_vars(
        solver: Solver, model1: pyo.AbstractModel, model2: pyo.AbstractModel
    ) -> bool:
        """
        Check if fixing variables from `model1` in `model2` results in a feasible solution.

        Parameters:
            solver: Solver instance to use.
            model1: Solved Pyomo model (variables contain values).
            model2: Pyomo model to test feasibility with fixed variables.

        Returns:
            feasible: True if the fixed variables lead to a feasible solution, False otherwise.
        """
        # Fix variables in model2 using values from model1
        for var1 in model1.component_objects(pyo.Var, active=True):
            var1_name = var1.name
            if hasattr(model2, var1_name):
                var2 = getattr(model2, var1_name)
                for index in var1:
                    if index in var2:
                        raw = pyo.value(var1[index])
                        lb = var2[index].lb if var2[index].lb is not None else -1e20
                        ub = var2[index].ub if var2[index].ub is not None else 1e20
                        if raw < lb:
                            raw = lb
                        elif raw > ub:
                            raw = ub
                        var2[index].fix(raw)
        try:
            results = solver.solve_from_model(model=model2)
            feasible = results["results_model"].solver.termination_condition in [
                pyo.TerminationCondition.optimal,
                pyo.TerminationCondition.feasible,
            ]

        except Exception as e:
            logging.error(f"Error in check_feasibility_with_fixed_vars:\n{e}")
            return False

        logging.info(f"Feasibility check: {feasible}")
        return feasible

    @staticmethod
    def check_usefullness_with_relaxarion(
        solver: Solver, full_code: str, true_code: str, data: dict
    ) -> bool:
        """
        Check if the cut is useful by comparing the best LP point of the true model with the cut model.
        If the optimal value of true_code is feasible for the cut model, then the cut is not useful.

        Parameters:
            solver: Solver instance to use.
            full_code: Full code of the model with the cut.
            true_code: Full code of the true model.
            data: Data to solve the models.

        Returns:
            is_useful: True if the cut is useful, False otherwise.
        """

        model_true = solver.get_model_from_text(
            "create_model", remove_constraint(true_code), **data["args"]
        )
        # Apply LP relaxation
        model_true_lp = make_lp_relax_model(model=model_true)
        results_true = solver.solve_from_model(model=model_true_lp)
        model_true_lp = results_true["model"]

        model_cut = solver.get_model_from_text("create_model", full_code, **data["args"])
        # Apply LP Relaxation
        model_cut_lp = make_lp_relax_model(model=model_cut)

        return not CutVerifier.check_feasibility_with_fixed_vars(
            solver=solver, model1=model_true_lp, model2=model_cut_lp
        )
