import pyomo.environ as pyo
import time
import io
import contextlib
import re
import pyomo
from typing import List
import logging
import copy


class Solver:
    def __init__(
        self,
        solver_name: str = "gurobi_persistent",
        time_limit: int = 10,
        lazy_value: int = 1,
        apply_lazy: bool = False,
    ):
        self.solver_name = solver_name
        self.lazy_value = lazy_value
        self.apply_lazy = apply_lazy
        self._reset_solver()

        self.time_limit = time_limit

    def _reset_solver(self):
        if self.solver_name == "gurobi":
            self.solver = pyo.SolverFactory(self.solver_name)
            self.extract_solver_report = Solver.extract_solving_report_gurobi

        elif self.solver_name == "gurobi_persistent":
            self.solver = pyomo.opt.SolverFactory(
                self.solver_name, solver_io="python", manage_env=True
            )
            self.extract_solver_report = Solver.extract_solving_report_gurobi

        elif self.solver_name == "appsi_highs":
            self.solver = pyo.SolverFactory(self.solver_name)
            self.extract_solver_report = Solver.extract_solving_report_highs
        else:
            self.solver = pyo.SolverFactory(self.solver_name)

    def _configure_gurobi_persistent(self, time_limit, use_work_limit):
        """Centralized options for gurobi_persistent."""
        opt = self.solver.options
        opt.update({"Seed": 42, "Method": 4})
        if use_work_limit:
            opt["WorkLimit"] = time_limit
            opt.pop("TimeLimit", None)
        else:
            opt["TimeLimit"] = time_limit
            opt.pop("WorkLimit", None)

    def _solve_and_capture(
        self,
        model,
        time_limit=None,
        main_constraints=None,
        use_work_limit=True,
        reset_solver=False,
        pool_solutions=None,
    ):
        """Generic solve wrapper used by every public helper."""
        time_limit = time_limit or self.time_limit
        if reset_solver:
            self._reset_solver()
        passes_model = self.solver_name not in {"gurobi_persistent"}

        # Configure solver
        solver = self.solver
        if self.solver_name == "appsi_highs":
            solve_kwargs = dict(tee=True, timelimit=time_limit, report_timing=True)
        elif self.solver_name == "gurobi":
            solver.options.update({"Seed": 42, "TimeLimit": time_limit})
            solve_kwargs = dict(tee=True, report_timing=True)
        elif self.solver_name == "gurobi_persistent":
            self._configure_gurobi_persistent(time_limit, use_work_limit)
            solver.set_instance(model)
            if self.apply_lazy and main_constraints:
                self.apply_lazy_constraints(model, main_constraints)
            if pool_solutions:
                solver.set_gurobi_param("PoolSolutions", pool_solutions)
                solver.set_gurobi_param("PoolSearchMode", 2)
            solve_kwargs = dict(tee=True, report_timing=True)
        else:
            raise ValueError(f"Unsupported solver {self.solver_name!r}")

        # Run & capture
        output = io.StringIO()
        start = time.time()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                if passes_model:
                    results_model = solver.solve(model, **solve_kwargs)
                else:  # gurobi_persistent
                    results_model = solver.solve(**solve_kwargs)
            except ValueError as e:  # Gurobi work-limit quirk
                logging.warning("Solver warning: %s", e)
                results_model = None
        elapsed = time.time() - start
        solver_log = output.getvalue()
        solver_report = self.extract_solver_report(solver_log)

        # Handle solution pool (optional)
        model_or_models = model
        if pool_solutions and self.solver_name == "gurobi_persistent":
            gmodel = solver._solver_model
            g2p = solver._solver_var_to_pyomo_var_map
            found = gmodel.SolCount
            logging.info("Found %s solutions in Gurobi pool", found)
            models = []
            for idx in range(min(found, pool_solutions)):
                solver.set_gurobi_param("SolutionNumber", idx)
                solver.load_vars()
                models.append(copy.deepcopy(model.clone()))
            model_or_models = models

        # Return one tidy dict
        return {
            "results_model": results_model,
            "model": model_or_models,
            "time": elapsed,
            "solver_log": solver_log,
            "solver_report": solver_report,
            "termination_condition": (
                results_model.solver.termination_condition if results_model else "feasible"
            ),
        }

    def solve_from_text(
        self,
        function_name: str,
        function_string: str,
        time_limit: int = None,
        main_constraints: List[str] = None,
        pool_solutions=None,
        **kwargs,
    ):
        model = self.get_model_from_text(function_name, function_string, **kwargs)
        return self._solve_and_capture(
            model, time_limit, main_constraints, use_work_limit=True, pool_solutions=pool_solutions
        )

    def solve_from_model(
        self,
        model,
        time_limit=None,
        main_constraints=None,
        use_work_limit=True,
        reset_solver=False,
        pool_solutions=None,
    ):
        return self._solve_and_capture(
            model,
            time_limit,
            main_constraints,
            use_work_limit=use_work_limit,
            reset_solver=reset_solver,
            pool_solutions=pool_solutions,
        )

    def solve_from_text_sol_pool(
        self,
        function_name,
        function_string,
        time_limit=None,
        main_constraints=None,
        use_work_limit=False,
        reset_solver=True,
        pool_solutions=2,
        **kwargs,
    ):
        if self.solver_name != "gurobi_persistent":
            raise ValueError("solve_from_text_sol_pool requires gurobi_persistent")
        model = self.get_model_from_text(function_name, function_string, **kwargs)
        return self._solve_and_capture(
            model,
            time_limit,
            main_constraints,
            use_work_limit=use_work_limit,
            reset_solver=reset_solver,
            pool_solutions=pool_solutions,
        )

    def get_model_from_text(self, function_name: str, function_string: str, **kwargs):
        exec(function_string, globals())
        # Apply kwargs on the function with name function_name using getattr
        function = globals()[function_name]
        model = function(**kwargs)
        return model

    @staticmethod
    def extract_solving_report_highs(text: str) -> dict:
        """
        Extracts the "Solving report" section from the given text and returns a dictionary.

        For keys with multiple lines (e.g., Timing), sub-lines are stored in a nested dictionary
        using the pattern "value (subkey)".
        """
        report_dict = {}
        lines = text.splitlines()
        start_extract = False
        current_key = None
        current_indent = None

        # Regex to match a "value (subkey)" pattern.
        value_subkey_re = re.compile(r"^(.*?)\s*\((.*?)\)$")
        # Regex to extract a top-level key and value (splitting on 2+ spaces).
        key_value_re = re.compile(r"^\s*(.+?)\s{2,}(.*)$")

        for line in lines:
            if "Solving report" in line:
                start_extract = True
                continue
            if not start_extract or not line.strip():
                continue

            indent = len(line) - len(line.lstrip())

            # New top-level key if the indent is less than or equal to the last key's indent.
            if current_key is None or indent <= current_indent:
                match = key_value_re.match(line)
                if match:
                    key = match.group(1).strip()
                    value_str = match.group(2).strip()
                    # Check if the value_str matches the "value (subkey)" pattern.
                    m = value_subkey_re.match(value_str)
                    if m:
                        subkey = m.group(2).strip()
                        subvalue = m.group(1).strip()
                        report_dict[key] = {subkey: subvalue}
                    else:
                        report_dict[key] = value_str
                    current_key = key
                    current_indent = indent
            else:
                # This line is a continuation for the current key.
                sub_line = line.strip()
                m = value_subkey_re.match(sub_line)
                if m:
                    sub_key = m.group(2).strip()
                    sub_value = m.group(1).strip()
                    if not isinstance(report_dict[current_key], dict):
                        # Convert the current value to a dict, preserving the original value.
                        report_dict[current_key] = {"value": report_dict[current_key]}
                    report_dict[current_key][sub_key] = sub_value
                else:
                    # If sub-line doesn't match, you could choose to append it.
                    if isinstance(report_dict[current_key], dict):
                        report_dict[current_key]["extra"] = (
                            report_dict[current_key].get("extra", "") + "\n" + sub_line
                        )
                    else:
                        report_dict[current_key] += "\n" + sub_line
        return report_dict

    @staticmethod
    def extract_solving_report_gurobi(log: str) -> dict:
        """
        Parses the given log string and extracts key metrics from a Gurobi solve.

        Returned dictionary keys:
        - 'gap': The percentage gap (as a float).
        - 'total_time': Seconds required for solver.
        - 'read_logfile': Seconds required to read the logfile.
        - 'read_solution_file': Seconds required to read the solution file.
        - 'postsolve': Seconds required for postsolve.
        - 'explored_nodes': Number of explored branch-and-bound nodes.
        - 'simplex_iterations': Number of simplex iterations.
        - 'explored_time': Time spent exploring nodes (seconds).
        - 'work_units': Gurobi "work units" spent.

        Parameters:
            log (str): The complete log as a single string.

        Returns:
            dict: A dictionary with the extracted metrics.
        """
        result = {}

        # Extract gap value (e.g., "gap 1.4302%")
        gap_match = re.search(r"gap\s+([\d\.]+)%", log)
        if gap_match:
            result["gap"] = float(gap_match.group(1))

        # Extract total time for solver (e.g., "40.13 seconds required for solver")
        total_time_match = re.search(r"([\d\.]+)\s+seconds required for solver", log)
        if total_time_match:
            result["total_time"] = float(total_time_match.group(1))

        # Extract time to read logfile
        logfile_match = re.search(r"([\d\.]+)\s+seconds required to read logfile", log)
        if logfile_match:
            result["read_logfile"] = float(logfile_match.group(1))

        # Extract time to read solution file
        solfile_match = re.search(r"([\d\.]+)\s+seconds required to read solution file", log)
        if solfile_match:
            result["read_solution_file"] = float(solfile_match.group(1))

        # Extract postsolve time
        postsolve_match = re.search(r"([\d\.]+)\s+seconds required for postsolve", log)
        if postsolve_match:
            result["postsolve"] = float(postsolve_match.group(1))

        # Extract info from "Explored 237622 nodes (6971457 simplex iterations) in 1026.32 seconds (1000.48 work units)"
        explored_match = re.search(
            r"Explored\s+(\d+)\s+nodes\s+\((\d+)\s+simplex iterations\)\s+in\s+([\d\.]+)\s+seconds\s+\(([\d\.]+)\s+work units\)",
            log,
        )
        if explored_match:
            result["explored_nodes"] = int(explored_match.group(1))
            result["simplex_iterations"] = int(explored_match.group(2))
            result["explored_time"] = float(explored_match.group(3))
            result["work_units"] = float(explored_match.group(4))

        return result

    def apply_lazy_constraints(self, model, main_constraints: List[str]):
        """
        Apply lazy constraints to the model.
        """
        # Loop through the model's constraints, if they are not in the main_constraints list then apply lazy constraints
        for constraint in model.component_objects(pyo.Constraint):
            if constraint.name not in main_constraints:
                # get the attribute of the constraint
                constraint_attr = getattr(model, constraint.name)
                for idx in constraint_attr:
                    self.solver.set_linear_constraint_attr(
                        constraint_attr[idx], "Lazy", self.lazy_value
                    )

        # TODO: update the model to avoid error(QConstr has not yet been added to the model) like model.update in gorubi for now, only give some time, it may work!!!!!
        time.sleep(3)
        self.solver.update()
        self.solver._solver_model.update()
        self.solver.update()
        time.sleep(1)
