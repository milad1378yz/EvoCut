from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classes.cut_verifier import CutVerifier
from src.cuts.reference_cuts import REFERENCE_CUTS

import pyomo.environ as pyo


@dataclass(frozen=True)
class InstanceRecord:
    instance_id: str
    model_member: str
    meta_member: str
    is_optimal: bool


@dataclass(frozen=True)
class TsplibRecord:
    instance_id: str
    n: int
    c: Dict[Tuple[int, int], float]
    tour: List[int]
    expected_obj: Optional[float]


class Reader:
    def list_members(self) -> List[str]:
        raise NotImplementedError

    def read_bytes(self, member: str) -> bytes:
        raise NotImplementedError

    def close(self) -> None:
        return


class ZipReader(Reader):
    def __init__(self, path: str):
        import zipfile

        self._zf = zipfile.ZipFile(path)

    def list_members(self) -> List[str]:
        return self._zf.namelist()

    def read_bytes(self, member: str) -> bytes:
        return self._zf.read(member)

    def close(self) -> None:
        self._zf.close()


class DirReader(Reader):
    def __init__(self, path: str):
        self._root = os.path.abspath(path)

    def list_members(self) -> List[str]:
        out: List[str] = []
        for root, _dirs, files in os.walk(self._root):
            for fn in files:
                full = os.path.join(root, fn)
                out.append(os.path.relpath(full, self._root))
        return out

    def read_bytes(self, member: str) -> bytes:
        with open(os.path.join(self._root, member), "rb") as f:
            return f.read()


class BsdtarReader(Reader):
    def __init__(self, path: str):
        self._path = path
        if not shutil.which("bsdtar"):
            raise RuntimeError("bsdtar not found; cannot read .rar/.tar archives")

    def list_members(self) -> List[str]:
        out = subprocess.check_output(["bsdtar", "-tf", self._path], text=True)
        return [line.strip() for line in out.splitlines() if line.strip()]

    def read_bytes(self, member: str) -> bytes:
        return subprocess.check_output(["bsdtar", "-xOf", self._path, member])


def open_reader(path: str) -> Reader:
    if os.path.isdir(path):
        return DirReader(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".zip":
        return ZipReader(path)
    if ext in {".rar", ".tar", ".tgz", ".tar.gz"}:
        return BsdtarReader(path)
    raise ValueError(f"Unsupported results path: {path}")


def infer_problem_name(results_path: str, members: List[str]) -> Optional[str]:
    name = os.path.basename(results_path).lower()
    for p in REFERENCE_CUTS:
        if p in name:
            return p

    top_dirs = {m.split("/", 1)[0].lower() for m in members if "/" in m}
    for p in REFERENCE_CUTS:
        if p in top_dirs:
            return p

    return None


def _download_to_path(url: str, output_path: str, *, timeout_s: int = 60) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tmp_path = output_path + ".part"
    with requests.get(url, stream=True, timeout=timeout_s) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    os.replace(tmp_path, output_path)


def _find_tsplib_tsp_dir(root: str) -> Optional[str]:
    for dirpath, _dirs, files in os.walk(root):
        if os.path.basename(dirpath) != "tsp":
            continue
        parent = os.path.basename(os.path.dirname(dirpath))
        if parent != "TSPLIB95":
            continue
        if any(fn.endswith(".tsp") for fn in files):
            return dirpath
    return None


def ensure_tsplib_tsp_dir(
    cache_dir: str,
    *,
    zip_url: str = "https://github.com/pdrozdowski/TSPLib.Net/archive/refs/heads/master.zip",
) -> str:
    """
    Ensure TSPLIB instances + optimal tours are present locally.

    Uses the TSPLib.Net GitHub mirror (contains `TSPLIB95/tsp/*.tsp` and `*.opt.tour`).
    Returns the local path to the `TSPLIB95/tsp` directory.
    """
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    tsp_dir = _find_tsplib_tsp_dir(cache_dir)
    if tsp_dir is not None:
        return tsp_dir

    zip_path = os.path.join(cache_dir, "TSPLib.Net-master.zip")
    if not os.path.exists(zip_path):
        _download_to_path(zip_url, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)

    tsp_dir = _find_tsplib_tsp_dir(cache_dir)
    if tsp_dir is None:
        raise RuntimeError(f"Could not locate TSPLIB95/tsp under {cache_dir}")

    return tsp_dir


def _parse_tsplib_best_solutions(path: str) -> Dict[str, float]:
    best: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            name, val = line.split(":", 1)
            name = name.strip()
            val = val.strip()
            try:
                best[name] = float(val)
            except ValueError:
                continue
    return best


def _read_tsplib_dimension(tsp_path: str) -> Optional[int]:
    try:
        with open(tsp_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                u = line.strip().upper()
                if not u.startswith("DIMENSION"):
                    continue
                parts = u.replace(":", " ").split()
                for tok in reversed(parts):
                    if tok.isdigit():
                        return int(tok)
    except OSError:
        return None
    return None


def _normalize_tour(tour: List[int], *, n: int, depot: int = 1) -> List[int]:
    tour = [int(v) for v in tour if int(v) > 0]
    if len(tour) == n + 1 and tour[0] == tour[-1]:
        tour = tour[:-1]
    if len(tour) != n:
        raise ValueError(f"Tour length mismatch: got {len(tour)} nodes, expected {n}")
    if len(set(tour)) != n:
        raise ValueError("Tour has duplicate nodes")
    if depot not in tour:
        raise ValueError(f"Depot node {depot} not present in tour")
    idx = tour.index(depot)
    return tour[idx:] + tour[:idx]


def _tour_cost_from_c(c: Dict[Tuple[int, int], float], tour: List[int]) -> float:
    total = 0.0
    n = len(tour)
    for k in range(n):
        i = tour[k]
        j = tour[(k + 1) % n]
        total += float(c[i, j])
    return total


def _load_tsplib_record(
    *,
    instance_id: str,
    tsp_path: str,
    tour_path: str,
    expected_obj: Optional[float],
    validate_obj: bool,
) -> TsplibRecord:
    import tsplib95

    tsp_problem = tsplib95.load(tsp_path)
    tour_problem = tsplib95.load(tour_path)

    raw_nodes = sorted(int(v) for v in tsp_problem.get_nodes())
    n = len(raw_nodes)
    old_to_new = {old: idx + 1 for idx, old in enumerate(raw_nodes)}

    c: Dict[Tuple[int, int], float] = {}
    for i_old, i_new in old_to_new.items():
        for j_old, j_new in old_to_new.items():
            if i_new == j_new:
                continue
            c[(i_new, j_new)] = float(tsp_problem.get_weight(i_old, j_old))

    tours = list(tour_problem.tours)
    if not tours:
        raise ValueError("No TOUR_SECTION found")
    tour0 = tours[0]
    mapped_tour: List[int] = []
    for node in tour0:
        node_int = int(node)
        if node_int < 0:
            continue
        if node_int not in old_to_new:
            raise ValueError(f"Tour contains unknown node id {node_int}")
        mapped_tour.append(old_to_new[node_int])
    mapped_tour = _normalize_tour(mapped_tour, n=n, depot=1)

    if validate_obj and expected_obj is not None:
        obj = _tour_cost_from_c(c, mapped_tour)
        if abs(obj - expected_obj) > 1e-6:
            raise ValueError(
                f"Tour objective mismatch: expected {expected_obj}, got {obj} (instance={instance_id})"
            )

    return TsplibRecord(
        instance_id=instance_id,
        n=n,
        c=c,
        tour=mapped_tour,
        expected_obj=expected_obj,
    )


def collect_tsplib_tsp_records(
    *,
    cache_dir: str,
    zip_url: str,
    min_n: int,
    max_n: int,
    validate_obj: bool,
    max_instances: Optional[int] = None,
    instances: Optional[List[str]] = None,
) -> List[TsplibRecord]:
    tsp_dir = ensure_tsplib_tsp_dir(cache_dir, zip_url=zip_url)

    best_path = os.path.join(tsp_dir, "bestSolutions.txt")
    best = _parse_tsplib_best_solutions(best_path) if os.path.exists(best_path) else {}

    wanted = {s.strip() for s in instances} if instances else None

    candidates: List[Tuple[int, str, str, str]] = []
    for fn in sorted(os.listdir(tsp_dir)):
        if not fn.endswith(".tsp"):
            continue
        instance_id = fn[: -len(".tsp")]
        if wanted is not None and instance_id not in wanted:
            continue

        tsp_path = os.path.join(tsp_dir, fn)
        n = _read_tsplib_dimension(tsp_path)
        if n is None or n < min_n or n > max_n:
            continue

        tour_path = None
        for ext in (".opt.tour", ".tour"):
            cand = os.path.join(tsp_dir, instance_id + ext)
            if os.path.exists(cand):
                tour_path = cand
                break
        if tour_path is None:
            continue

        candidates.append((n, instance_id, tsp_path, tour_path))

    candidates.sort(key=lambda t: (t[0], t[1]))

    records: List[TsplibRecord] = []
    for _n, instance_id, tsp_path, tour_path in candidates:
        if max_instances is not None and len(records) >= max_instances:
            break
        try:
            rec = _load_tsplib_record(
                instance_id=instance_id,
                tsp_path=tsp_path,
                tour_path=tour_path,
                expected_obj=best.get(instance_id),
                validate_obj=validate_obj,
            )
        except Exception:
            continue

        records.append(rec)

    records.sort(key=lambda r: (r.n, r.instance_id))
    return records


def create_tsp_mtz_model(n: int, c: Dict[Tuple[int, int], float]):
    model = pyo.ConcreteModel()
    model.N = pyo.RangeSet(1, n)
    model.A = pyo.Set(initialize=[(i, j) for i in model.N for j in model.N if i != j])

    model.x = pyo.Var(model.A, domain=pyo.Binary)
    model.u = pyo.Var(model.N, domain=pyo.NonNegativeIntegers)

    model.obj = pyo.Objective(
        expr=sum(c[i, j] * model.x[i, j] for (i, j) in model.A), sense=pyo.minimize
    )

    model.outgoing_arc = pyo.Constraint(
        model.N, rule=lambda m, i: sum(m.x[i, j] for j in m.N if j != i) == 1
    )
    model.incoming_arc = pyo.Constraint(
        model.N, rule=lambda m, j: sum(m.x[i, j] for i in m.N if i != j) == 1
    )

    def mtz_rule(m, i, j):
        if i != j and i != 1 and j != 1:
            return m.u[i] - m.u[j] + n * m.x[i, j] <= n - 1
        return pyo.Constraint.Skip

    model.subtour_elimination = pyo.Constraint(model.N, model.N, rule=mtz_rule)
    model.u_lower = pyo.Constraint(
        model.N, rule=lambda m, i: m.u[i] >= 2 if i != 1 else pyo.Constraint.Skip
    )
    model.u_upper = pyo.Constraint(
        model.N, rule=lambda m, i: m.u[i] <= n if i != 1 else pyo.Constraint.Skip
    )
    model.u_fix = pyo.Constraint(expr=model.u[1] == 1)
    return model


def set_tsp_solution_from_tour(model, tour: List[int]) -> None:
    n = len(tour)
    if n == 0:
        raise ValueError("Empty tour")
    if tour[0] != 1:
        tour = _normalize_tour(tour, n=n, depot=1)

    pos = {node: idx + 1 for idx, node in enumerate(tour)}
    for node in model.N:
        model.u[int(node)].value = int(pos[int(node)])

    edges = {(tour[k], tour[(k + 1) % n]) for k in range(n)}
    for (i, j) in model.A:
        model.x[i, j].value = 1 if (i, j) in edges else 0


def _parse_meta_is_optimal(meta: Dict) -> bool:
    if "solve_status" in meta:
        return str(meta.get("solve_status", "")).lower() == "optimal"
    if "is_optimal" in meta:
        return bool(meta.get("is_optimal"))
    if "solver_verified" in meta and "validated" in meta:
        return bool(meta.get("validated")) and bool(meta.get("solver_verified"))
    return False


def collect_instances(reader: Reader) -> List[InstanceRecord]:
    members = reader.list_members()
    member_set = set(members)

    records: List[InstanceRecord] = []
    for model_member in members:
        if not model_member.endswith("_model.pkl"):
            continue

        base = model_member[: -len("_model.pkl")]
        meta_member = None
        for cand in (base + "_meta.json", base + "_metadata.json"):
            if cand in member_set:
                meta_member = cand
                break
        if meta_member is None:
            continue

        try:
            meta = json.loads(reader.read_bytes(meta_member))
        except Exception:
            continue

        is_optimal = _parse_meta_is_optimal(meta)
        instance_id = meta.get("instance_name") or meta.get("name") or os.path.basename(base)
        records.append(
            InstanceRecord(
                instance_id=str(instance_id),
                model_member=model_member,
                meta_member=meta_member,
                is_optimal=is_optimal,
            )
        )

    records.sort(key=lambda r: r.instance_id)
    return records


def apply_reference_cut(problem_name: str, model) -> None:
    fn = REFERENCE_CUTS[problem_name]
    fn(model)


def _constraint_is_satisfied(c, *, tol: float) -> bool:
    body = float(pyo.value(c.body))

    if c.lower is not None:
        lb = float(pyo.value(c.lower))
        if body < lb - tol:
            return False
    if c.upper is not None:
        ub = float(pyo.value(c.upper))
        if body > ub + tol:
            return False
    return True


def _set_jssp_z_values(model) -> None:
    pred = {u: 0 for u in model.O}
    succ = {u: 0 for u in model.O}

    for (j1, k1, j2, k2) in model.Pairs:
        u = (j1, k1)
        v = (j2, k2)
        y_val = int(round(float(pyo.value(model.y[j1, k1, j2, k2]))))
        if y_val == 1:
            succ[u] += 1
            pred[v] += 1
        else:
            succ[v] += 1
            pred[u] += 1

    for u in model.O:
        model.z_first[u].value = 1.0 if pred[u] == 0 else 0.0
        model.z_last[u].value = 1.0 if succ[u] == 0 else 0.0


def _check_added_constraints_eval(model, added_components: Iterable[str], *, tol: float) -> bool:
    for cname in added_components:
        comp = getattr(model, cname)
        if comp.is_indexed():
            for c in comp.values():
                if not _constraint_is_satisfied(c, tol=tol):
                    return False
        else:
            if not _constraint_is_satisfied(comp, tol=tol):
                return False
    return True


def check_preservation_for_archive(
    results_path: str,
    *,
    problem_name: str,
    method: str,
    solver: Optional["Solver"],
    max_instances: Optional[int] = None,
    tol: float = 1e-6,
) -> Tuple[int, int, List[str]]:
    reader = open_reader(results_path)
    try:
        records = collect_instances(reader)
        optimal_records = [r for r in records if r.is_optimal]
        if max_instances is not None:
            optimal_records = optimal_records[:max_instances]

        kept = 0
        failed: List[str] = []
        for rec in tqdm(optimal_records, desc=f"Optimal solution preservation: {problem_name}"):
            base_model = pickle.loads(reader.read_bytes(rec.model_member))
            cut_model = base_model.clone()

            before_constraints = {
                c.name for c in cut_model.component_objects(pyo.Constraint, active=True)
            }
            apply_reference_cut(problem_name, cut_model)
            after_constraints = {
                c.name for c in cut_model.component_objects(pyo.Constraint, active=True)
            }
            added_constraints = sorted(after_constraints - before_constraints)

            if method == "eval":
                if problem_name == "jssp":
                    _set_jssp_z_values(cut_model)
                ok = _check_added_constraints_eval(cut_model, added_constraints, tol=tol)
            else:
                if solver is None:
                    raise ValueError("solver must be provided when method='solve'")
                ok = CutVerifier.check_feasibility_with_fixed_vars(
                    solver=solver, model1=base_model, model2=cut_model
                )
            if ok:
                kept += 1
            else:
                failed.append(rec.instance_id)

        return kept, len(optimal_records), failed
    finally:
        reader.close()


def check_preservation_for_tsplib(
    *,
    cache_dir: str,
    zip_url: str,
    method: str,
    solver: Optional["Solver"],
    max_instances: Optional[int] = None,
    tol: float = 1e-6,
    min_n: int = 20,
    max_n: int = 250,
    validate_obj: bool = False,
    instances: Optional[List[str]] = None,
) -> Tuple[int, int, List[str]]:
    records = collect_tsplib_tsp_records(
        cache_dir=cache_dir,
        zip_url=zip_url,
        min_n=min_n,
        max_n=max_n,
        validate_obj=validate_obj,
        max_instances=max_instances,
        instances=instances,
    )

    kept = 0
    failed: List[str] = []
    for rec in tqdm(records, desc="Optimal solution preservation: tsp(tsplib)"):
        base_model = create_tsp_mtz_model(rec.n, rec.c)
        set_tsp_solution_from_tour(base_model, rec.tour)

        cut_model = base_model.clone()
        before_constraints = {c.name for c in cut_model.component_objects(pyo.Constraint, active=True)}
        apply_reference_cut("tsp", cut_model)
        after_constraints = {c.name for c in cut_model.component_objects(pyo.Constraint, active=True)}
        added_constraints = sorted(after_constraints - before_constraints)

        if method == "eval":
            # ensure values are present on the cut_model as well
            set_tsp_solution_from_tour(cut_model, rec.tour)
            ok = _check_added_constraints_eval(cut_model, added_constraints, tol=tol)
        else:
            if solver is None:
                raise ValueError("solver must be provided when method='solve'")
            ok = CutVerifier.check_feasibility_with_fixed_vars(
                solver=solver, model1=base_model, model2=cut_model
            )

        if ok:
            kept += 1
        else:
            failed.append(rec.instance_id)

    return kept, len(records), failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Optimal Solution Preservation rate from saved solved models.",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["results", "tsplib"],
        default="results",
        help="Data source: 'results' (archives of solved models) or 'tsplib' (download TSPLIB opt tours).",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="results/gurobi",
        help="Path to a results archive (.zip/.rar) or a directory containing archives.",
    )
    parser.add_argument(
        "--problem_name",
        type=str,
        default="auto",
        help="Problem name (cwlp/mcnd/jssp/tsp) or 'auto'. If results_path is a directory, this is ignored.",
    )
    parser.add_argument(
        "--solver_name",
        type=str,
        default="appsi_highs",
        help="Pyomo solver name (e.g., gurobi_persistent, gurobi, appsi_highs).",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=60,
        help="Per-instance feasibility solve time limit (seconds).",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Optional cap on number of optimal instances to check per archive.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["solve", "eval"],
        default="solve",
        help="Feasibility method: 'solve' fixes vars and solves; 'eval' checks added constraints directly.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Feasibility tolerance for --method eval.",
    )
    parser.add_argument(
        "--tsplib_cache_dir",
        type=str,
        default="data/tsplib_cache",
        help="Cache directory for TSPLIB instances/tours when --source tsplib.",
    )
    parser.add_argument(
        "--tsplib_zip_url",
        type=str,
        default="https://github.com/pdrozdowski/TSPLib.Net/archive/refs/heads/master.zip",
        help="ZIP URL for a TSPLIB mirror containing TSPLIB95/tsp/*.tsp and *.opt.tour.",
    )
    parser.add_argument(
        "--tsplib_min_n",
        type=int,
        default=20,
        help="Minimum number of nodes (inclusive) for TSPLIB TSP instances.",
    )
    parser.add_argument(
        "--tsplib_max_n",
        type=int,
        default=250,
        help="Maximum number of nodes (inclusive) for TSPLIB TSP instances.",
    )
    parser.add_argument(
        "--tsplib_instances",
        type=str,
        default="",
        help="Comma-separated TSPLIB instance names to check (e.g., 'att48,eil51'). Empty = all with tours.",
    )
    parser.add_argument(
        "--tsplib_validate_obj",
        action="store_true",
        help="Validate that each parsed tour matches TSPLIB's bestSolutions objective (if available).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    solver = None
    if args.method == "solve":
        from src.classes.solver import Solver

        solver = Solver(solver_name=args.solver_name, time_limit=args.time_limit)

    if args.source == "tsplib":
        if args.problem_name.lower() not in {"tsp", "auto"}:
            raise SystemExit("--source tsplib only supports --problem_name tsp")

        instances = None
        if args.tsplib_instances.strip():
            instances = [s.strip() for s in args.tsplib_instances.split(",") if s.strip()]

        kept, total, failed = check_preservation_for_tsplib(
            cache_dir=args.tsplib_cache_dir,
            zip_url=args.tsplib_zip_url,
            method=args.method,
            solver=solver,
            max_instances=args.max_instances,
            tol=args.tol,
            min_n=args.tsplib_min_n,
            max_n=args.tsplib_max_n,
            validate_obj=args.tsplib_validate_obj,
            instances=instances,
        )
        pct = 0.0 if total == 0 else (100.0 * kept / total)
        print(f"tsp(tsplib): {kept}/{total} kept optimal ({pct:.2f}%)")
        if failed:
            print(f"Failures ({len(failed)}): {', '.join(failed[:50])}")
        return

    if os.path.isdir(args.results_path):
        archives = [
            os.path.join(args.results_path, f)
            for f in os.listdir(args.results_path)
            if os.path.isfile(os.path.join(args.results_path, f))
            and os.path.splitext(f)[1].lower() in {".zip", ".rar"}
        ]
        archives.sort()
        if not archives:
            raise SystemExit(f"No .zip/.rar archives found under {args.results_path}")

        summary = []
        for ap in archives:
            reader = open_reader(ap)
            members = reader.list_members()
            reader.close()
            problem = infer_problem_name(ap, members)
            if not problem:
                print(f"[skip] cannot infer problem name for {ap}")
                continue
            if problem not in REFERENCE_CUTS:
                print(f"[skip] no reference cut registered for problem={problem}")
                continue

            kept, total, failed = check_preservation_for_archive(
                ap,
                problem_name=problem,
                method=args.method,
                solver=solver,
                max_instances=args.max_instances,
                tol=args.tol,
            )
            pct = 0.0 if total == 0 else (100.0 * kept / total)
            summary.append((problem, kept, total, pct, failed))

        print("\n" + "=" * 72)
        for problem, kept, total, pct, failed in summary:
            print(f"{problem}: {kept}/{total} kept optimal ({pct:.2f}%)")
            if failed:
                print(f"  first failures: {', '.join(failed[:10])}")
        print("=" * 72 + "\n")
        return

    # single archive
    reader = open_reader(args.results_path)
    members = reader.list_members()
    reader.close()

    if args.problem_name == "auto":
        problem = infer_problem_name(args.results_path, members)
        if not problem:
            raise SystemExit(
                "Cannot infer --problem_name from results_path; pass --problem_name explicitly."
            )
    else:
        problem = args.problem_name.lower()

    if problem not in REFERENCE_CUTS:
        raise SystemExit(f"Unsupported problem_name={problem}. Known: {sorted(REFERENCE_CUTS)}")

    kept, total, failed = check_preservation_for_archive(
        args.results_path,
        problem_name=problem,
        method=args.method,
        solver=solver,
        max_instances=args.max_instances,
        tol=args.tol,
    )

    pct = 0.0 if total == 0 else (100.0 * kept / total)
    print(f"{problem}: {kept}/{total} kept optimal ({pct:.2f}%)")
    if failed:
        print(f"Failures ({len(failed)}): {', '.join(failed[:50])}")


if __name__ == "__main__":
    main()
