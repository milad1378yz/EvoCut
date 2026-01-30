import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx


_DEFAULT_TSP_N_VALUES = [
    21,
    22,
    24,
    26,
    29,
    42,
    48,
    51,
    52,
    58,
    70,
    76,
    96,
    99,
    100,
    101,
    105,
    107,
    120,
    124,
    127,
    130,
    136,
    137,
    144,
    150,
    152,
    159,
    175,
    180,
    195,
    198,
    200,
    202,
    225,
    226,
    229,
]

_DEFAULT_TSP_N_WEIGHTS = [
    1,
    1,
    1,
    1,
    2,
    2,
    3,
    1,
    1,
    1,
    1,
    2,
    1,
    1,
    6,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    3,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    1,
    2,
    1,
    1,
]

_DEFAULT_JSSP_SIZES = [
    (15, 15),
    (20, 15),
    (20, 20),
    (30, 15),
    (30, 20),
    (50, 15),
    (50, 20),
    (100, 20),
]


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _normalize_weights(weights: Sequence[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return arr / total


def _sample_from(values: Sequence[int], rng: np.random.Generator) -> int:
    return int(rng.choice(values))


def _safe_int(x: object, default: int) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _extract_reference_args(reference: Optional[List[Dict]], key: str) -> List:
    if not reference:
        return []
    extracted: List = []
    for inst in reference:
        if not isinstance(inst, dict):
            continue
        args = inst.get("args", {})
        if isinstance(args, dict) and key in args:
            extracted.append(args[key])
    return extracted


def generate_random_tsp_instances(
    num_instances: int,
    seed: Optional[int] = None,
    reference: Optional[List[Dict]] = None,
    n_range: Tuple[int, int] = (20, 250),
    dims: int = 2,
    coord_range: Tuple[float, float] = (0.0, 1000.0),
    integer_coords: bool = True,
) -> List[Dict]:
    """
    Generate random Euclidean TSP instances.

    If reference is provided, sample node counts from it to mimic its size distribution.
    Otherwise, sample from a TSPLIB-inspired size distribution.
    """
    rng = _rng(seed)

    ref_ns = [
        _safe_int(n, 0)
        for n in _extract_reference_args(reference, "n")
        if _safe_int(n, 0) > 0
    ]

    instances: List[Dict] = []
    lo, hi = coord_range
    lo_int = int(math.floor(lo))
    hi_int = int(math.ceil(hi))

    if ref_ns:
        n_choices = ref_ns
        use_weights = None
    else:
        n_choices = _DEFAULT_TSP_N_VALUES
        use_weights = _DEFAULT_TSP_N_WEIGHTS

    for idx in range(num_instances):
        if use_weights is None:
            n_nodes = _sample_from(n_choices, rng)
        else:
            n_nodes = int(rng.choice(n_choices, p=_normalize_weights(use_weights)))

        if not ref_ns and (n_nodes < n_range[0] or n_nodes > n_range[1]):
            n_nodes = int(rng.integers(n_range[0], n_range[1] + 1))

        nodes = list(range(1, n_nodes + 1))
        coords: Dict[int, List[float]] = {}
        for node_id in nodes:
            if integer_coords:
                coords[node_id] = [
                    float(rng.integers(lo_int, hi_int + 1)) for _ in range(dims)
                ]
            else:
                coords[node_id] = [float(rng.uniform(lo, hi)) for _ in range(dims)]

        c: Dict[Tuple[int, int], float] = {}
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                c[(i, j)] = float(math.dist(coords[i], coords[j]))

        instances.append(
            {
                "name": f"tsp_{idx}_{n_nodes}",
                "NODES": nodes,
                "TOURS": None,
                "coords": coords,
                "args": {"n": n_nodes, "c": c},
            }
        )

    return instances


def generate_random_jssp_instances(
    num_instances: int,
    seed: Optional[int] = None,
    reference: Optional[List[Dict]] = None,
    size_choices: Optional[List[Tuple[int, int]]] = None,
    ptime_range: Tuple[int, int] = (1, 99),
) -> List[Dict]:
    """
    Generate random JSSP instances with Taillard-like timing ranges.

    If reference is provided, sample (n_jobs, n_machines) and processing times from it.
    """
    rng = _rng(seed)
    ref_times = _extract_reference_args(reference, "times")
    ref_machines = _extract_reference_args(reference, "machines")

    ref_sizes: List[Tuple[int, int]] = []
    if ref_times:
        for times in ref_times:
            if isinstance(times, list) and times and isinstance(times[0], list):
                n_jobs = len(times)
                n_machines = len(times[0])
                if n_jobs > 0 and n_machines > 0:
                    ref_sizes.append((n_jobs, n_machines))
    if not ref_sizes:
        for times, machines in zip(ref_times, ref_machines):
            if isinstance(times, list) and times:
                n_jobs = len(times)
                n_machines = len(times[0]) if isinstance(times[0], list) else 0
                if n_jobs > 0 and n_machines > 0:
                    ref_sizes.append((n_jobs, n_machines))

    size_pool = ref_sizes or size_choices or _DEFAULT_JSSP_SIZES

    time_pool: Optional[np.ndarray] = None
    if ref_times:
        flat_times = [
            int(val)
            for row in ref_times
            for sub in row
            if isinstance(sub, list)
            for val in sub
        ]
        if flat_times:
            time_pool = np.asarray(flat_times, dtype=int)

    instances: List[Dict] = []
    for idx in range(num_instances):
        n_jobs, n_machines = size_pool[int(rng.integers(0, len(size_pool)))]

        if time_pool is None:
            times = rng.integers(ptime_range[0], ptime_range[1] + 1, size=(n_jobs, n_machines))
        else:
            times = rng.choice(time_pool, size=(n_jobs, n_machines), replace=True)

        machines = np.vstack([rng.permutation(n_machines) for _ in range(n_jobs)])

        instances.append(
            {
                "name": f"jssp_{idx}_{n_jobs}x{n_machines}",
                "args": {
                    "n_jobs": n_jobs,
                    "n_machines": n_machines,
                    "times": times.tolist(),
                    "machines": machines.tolist(),
                },
            }
        )

    return instances


def _connectivity_and_capacity_ok(
    nodes: List[int],
    arcs: List[Tuple[int, int]],
    capacities: Dict[Tuple[int, int], int],
    demand: Dict[int, int],
    origin: Dict[int, int],
    destination: Dict[int, int],
) -> bool:
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for i, j in arcs:
        g.add_edge(i, j, capacity=capacities[(i, j)])

    for k in demand:
        s, t = origin[k], destination[k]
        if not nx.has_path(g, s, t):
            return False
        if nx.maximum_flow_value(g, s, t, capacity="capacity") < demand[k]:
            return False
    return True


def _sample_arcs(
    rng: np.random.Generator, n_nodes: int, target_arcs: int
) -> List[Tuple[int, int]]:
    arcs: set[Tuple[int, int]] = set()
    while len(arcs) < target_arcs:
        i = int(rng.integers(1, n_nodes + 1))
        j = int(rng.integers(1, n_nodes + 1))
        if i == j:
            continue
        arcs.add((i, j))
    return list(arcs)


def generate_random_mcnd_instances(
    num_instances: int,
    seed: Optional[int] = None,
    reference: Optional[List[Dict]] = None,
    node_range: Tuple[int, int] = (20, 200),
    commodity_range: Tuple[int, int] = (2, 20),
    arc_density_range: Tuple[float, float] = (0.15, 0.35),
    capacity_range: Tuple[int, int] = (20, 200),
    fixed_cost_range: Tuple[int, int] = (50, 200),
    variable_cost_range: Tuple[int, int] = (5, 50),
    demand_range: Tuple[int, int] = (5, 30),
    feasibility_checks: bool = True,
    max_attempts: int = 50,
) -> List[Dict]:
    """
    Generate random MCND instances. If reference is provided, sizes and value
    ranges are inferred from it to match the benchmark distribution.
    """
    rng = _rng(seed)

    ref_nodes = _extract_reference_args(reference, "N")
    ref_arcs = _extract_reference_args(reference, "A")
    ref_commodities = _extract_reference_args(reference, "K")
    ref_caps = _extract_reference_args(reference, "u")
    ref_fixed = _extract_reference_args(reference, "f")
    ref_var = _extract_reference_args(reference, "c")
    ref_dem = _extract_reference_args(reference, "d")

    ref_node_counts = [len(n) for n in ref_nodes if isinstance(n, list) and len(n) >= 2]
    ref_arc_counts = [len(a) for a in ref_arcs if isinstance(a, list) and len(a) >= 1]
    ref_comm_counts = [len(k) for k in ref_commodities if isinstance(k, list) and len(k) >= 1]

    ref_cap_vals = [val for d in ref_caps if isinstance(d, dict) for val in d.values()]
    ref_fix_vals = [val for d in ref_fixed if isinstance(d, dict) for val in d.values()]
    ref_var_vals = [val for d in ref_var if isinstance(d, dict) for val in d.values()]
    ref_dem_vals = [val for d in ref_dem if isinstance(d, dict) for val in d.values()]

    instances: List[Dict] = []

    for idx in range(num_instances):
        if ref_node_counts:
            n_nodes = int(rng.choice(ref_node_counts))
        else:
            n_nodes = int(rng.integers(node_range[0], node_range[1] + 1))

        if ref_comm_counts:
            n_comms = int(rng.choice(ref_comm_counts))
        else:
            n_comms = int(rng.integers(commodity_range[0], commodity_range[1] + 1))

        if ref_arc_counts:
            density = float(rng.choice(ref_arc_counts)) / float(n_nodes * (n_nodes - 1))
            density = max(0.05, min(density, 0.9))
        else:
            density = float(rng.uniform(arc_density_range[0], arc_density_range[1]))

        target_arcs = max(int(density * n_nodes * (n_nodes - 1)), n_nodes)

        attempts = 0
        while True:
            attempts += 1

            nodes = list(range(1, n_nodes + 1))
            arcs = _sample_arcs(rng, n_nodes, target_arcs)

            cap_vals = ref_cap_vals or list(range(capacity_range[0], capacity_range[1] + 1))
            fix_vals = ref_fix_vals or list(range(fixed_cost_range[0], fixed_cost_range[1] + 1))
            var_vals = ref_var_vals or list(range(variable_cost_range[0], variable_cost_range[1] + 1))
            dem_vals = ref_dem_vals or list(range(demand_range[0], demand_range[1] + 1))

            capacity = {(i, j): int(rng.choice(cap_vals)) for (i, j) in arcs}
            fixed_cost = {(i, j): int(rng.choice(fix_vals)) for (i, j) in arcs}
            variable_cost = {(i, j): int(rng.choice(var_vals)) for (i, j) in arcs}

            commodities = list(range(1, n_comms + 1))
            origin: Dict[int, int] = {}
            destination: Dict[int, int] = {}
            demand: Dict[int, int] = {}
            for k in commodities:
                o = int(rng.integers(1, n_nodes + 1))
                d = int(rng.integers(1, n_nodes + 1))
                while d == o:
                    d = int(rng.integers(1, n_nodes + 1))
                origin[k] = o
                destination[k] = d
                demand[k] = int(rng.choice(dem_vals))

            if not feasibility_checks:
                break
            if _connectivity_and_capacity_ok(nodes, arcs, capacity, demand, origin, destination):
                break
            if attempts >= max_attempts:
                break

        instances.append(
            {
                "name": f"mcnd_{idx}",
                "args": {
                    "N": nodes,
                    "A": arcs,
                    "K": commodities,
                    "c": variable_cost,
                    "f": fixed_cost,
                    "u": capacity,
                    "d": demand,
                    "O": origin,
                    "D": destination,
                },
            }
        )

    return instances


def generate_random_cwlp_instances(
    num_instances: int,
    seed: Optional[int] = None,
    reference: Optional[List[Dict]] = None,
    customer_range: Tuple[int, int] = (50, 1000),
    site_range: Tuple[int, int] = (10, 100),
    demand_range: Tuple[int, int] = (5, 100),
    capacity_factor_range: Tuple[float, float] = (1.05, 1.35),
    capacity_spread: float = 0.1,
    fixed_cost_range: Tuple[int, int] = (1000, 20000),
    unit_cost_range: Tuple[float, float] = (1.0, 10.0),
) -> List[Dict]:
    """
    Generate random CWLP instances. If reference is provided, sample sizes and
    value ranges from it to mimic the benchmark distribution.
    """
    rng = _rng(seed)

    ref_I = _extract_reference_args(reference, "I")
    ref_J = _extract_reference_args(reference, "J")
    ref_d = _extract_reference_args(reference, "d")
    ref_u = _extract_reference_args(reference, "u")
    ref_f = _extract_reference_args(reference, "f")
    ref_c = _extract_reference_args(reference, "c")

    n_customers_list = [len(i) for i in ref_I if isinstance(i, list) and i]
    n_sites_list = [len(j) for j in ref_J if isinstance(j, list) and j]
    demand_vals = [val for d in ref_d if isinstance(d, dict) for val in d.values()]
    capacity_vals = [val for u in ref_u if isinstance(u, dict) for val in u.values()]
    fixed_vals = [val for f in ref_f if isinstance(f, dict) for val in f.values()]

    unit_cost_vals: List[float] = []
    if ref_c and ref_d:
        for c_map, d_map in zip(ref_c, ref_d):
            if not isinstance(c_map, dict) or not isinstance(d_map, dict):
                continue
            keys = list(c_map.keys())
            if not keys:
                continue
            sample_count = min(200, len(keys))
            sample_idx = rng.choice(len(keys), size=sample_count, replace=False)
            for idx in sample_idx:
                i, j = keys[int(idx)]
                d_i = float(d_map.get(i, 0))
                if d_i <= 0:
                    continue
                unit_cost_vals.append(float(c_map[(i, j)]) / d_i)

    cap_factors: List[float] = []
    for d_map, u_map in zip(ref_d, ref_u):
        if not isinstance(d_map, dict) or not isinstance(u_map, dict):
            continue
        total_d = sum(float(v) for v in d_map.values())
        total_u = sum(float(v) for v in u_map.values())
        if total_d > 0:
            cap_factors.append(total_u / total_d)

    instances: List[Dict] = []
    for idx in range(num_instances):
        n_customers = (
            int(rng.choice(n_customers_list))
            if n_customers_list
            else int(rng.integers(customer_range[0], customer_range[1] + 1))
        )
        n_sites = (
            int(rng.choice(n_sites_list))
            if n_sites_list
            else int(rng.integers(site_range[0], site_range[1] + 1))
        )

        I = list(range(1, n_customers + 1))
        J = list(range(1, n_sites + 1))

        if demand_vals:
            d = {i: int(rng.choice(demand_vals)) for i in I}
        else:
            d = {i: int(rng.integers(demand_range[0], demand_range[1] + 1)) for i in I}

        total_demand = sum(d.values())
        if cap_factors:
            cap_factor = float(rng.choice(cap_factors))
        else:
            cap_factor = float(rng.uniform(capacity_factor_range[0], capacity_factor_range[1]))
        cap_factor = max(1.0, cap_factor)
        avg_capacity = int(math.ceil(total_demand * cap_factor / max(1, n_sites)))

        if capacity_vals:
            u = {j: int(max(1, rng.choice(capacity_vals))) for j in J}
        else:
            u = {
                j: int(avg_capacity * rng.uniform(1 - capacity_spread, 1 + capacity_spread))
                for j in J
            }

        if fixed_vals:
            f = {j: int(rng.choice(fixed_vals)) for j in J}
        else:
            f = {j: int(rng.integers(fixed_cost_range[0], fixed_cost_range[1] + 1)) for j in J}

        if unit_cost_vals:
            unit_costs = unit_cost_vals
            c = {(i, j): float(rng.choice(unit_costs)) * d[i] for i in I for j in J}
        else:
            c = {
                (i, j): float(rng.uniform(unit_cost_range[0], unit_cost_range[1])) * d[i]
                for i in I
                for j in J
            }

        total_capacity = sum(u.values())
        if total_capacity < total_demand:
            scale = total_demand / max(1.0, total_capacity)
            u = {j: int(math.ceil(val * scale)) for j, val in u.items()}
            total_capacity = sum(u.values())

        instances.append(
            {
                "name": f"cwlp_{idx}_{n_customers}x{n_sites}",
                "args": {"I": I, "J": J, "d": d, "u": u, "f": f, "c": c},
                "meta": {
                    "n_customers": n_customers,
                    "n_sites": n_sites,
                    "total_demand": total_demand,
                    "total_capacity": total_capacity,
                    "capacity_factor": total_capacity / max(1.0, total_demand),
                },
            }
        )

    return instances


def generate_random_mclp_instances(
    num_instances: int,
    seed: Optional[int] = None,
    reference: Optional[List[Dict]] = None,
    demand_range: Tuple[int, int] = (200, 800),
    site_range: Tuple[int, int] = (200, 800),
    p_fraction_range: Tuple[float, float] = (0.01, 0.05),
    radius_range: Tuple[float, float] = (0.02, 0.1),
    weight_range: Tuple[int, int] = (50, 500),
) -> List[Dict]:
    rng = _rng(seed)

    ref_meta = _extract_reference_args(reference, "I")
    ref_sites = _extract_reference_args(reference, "J")
    ref_P = _extract_reference_args(reference, "P")
    ref_meta_blocks: List[Dict] = []
    if reference:
        for inst in reference:
            if isinstance(inst, dict) and isinstance(inst.get("meta"), dict):
                ref_meta_blocks.append(inst["meta"])

    n_demand_list = [len(i) for i in ref_meta if isinstance(i, list) and i]
    n_sites_list = [len(j) for j in ref_sites if isinstance(j, list) and j]

    radius_list: List[float] = []
    for meta in ref_meta_blocks:
        if isinstance(meta, dict) and "radius" in meta:
            try:
                radius_list.append(float(meta["radius"]))
            except (TypeError, ValueError):
                pass

    p_list = [int(p) for p in ref_P if isinstance(p, int) and p > 0]

    instances: List[Dict] = []
    for idx in range(num_instances):
        n_demand = (
            int(rng.choice(n_demand_list))
            if n_demand_list
            else int(rng.integers(demand_range[0], demand_range[1] + 1))
        )
        n_sites = (
            int(rng.choice(n_sites_list))
            if n_sites_list
            else int(rng.integers(site_range[0], site_range[1] + 1))
        )

        if p_list:
            P = int(rng.choice(p_list))
        else:
            p_frac = float(rng.uniform(p_fraction_range[0], p_fraction_range[1]))
            P = max(1, int(math.ceil(p_frac * n_sites)))

        if radius_list:
            radius = float(rng.choice(radius_list))
        else:
            radius = float(rng.uniform(radius_range[0], radius_range[1]))

        coords_d = rng.random((n_demand, 2))
        coords_s = rng.random((n_sites, 2))

        a = {i + 1: int(rng.integers(weight_range[0], weight_range[1] + 1)) for i in range(n_demand)}

        Ni: Dict[int, List[int]] = {}
        rad2 = radius * radius
        for i, d_pt in enumerate(coords_d, start=1):
            sq_dists = np.sum((coords_s - d_pt) ** 2, axis=1)
            coverers = np.nonzero(sq_dists <= rad2)[0] + 1
            if coverers.size == 0:
                coverers = np.array([int(rng.integers(1, n_sites + 1))])
            Ni[i] = coverers.tolist()

        I = list(range(1, n_demand + 1))
        J = list(range(1, n_sites + 1))

        instances.append(
            {
                "name": f"mclp_{idx}_{n_demand}x{n_sites}",
                "args": {"I": I, "J": J, "Ni": Ni, "a": a, "P": P},
                "meta": {"n_demand": n_demand, "n_sites": n_sites, "radius": radius},
            }
        )

    return instances


def generate_random_2dbp_instances(
    num_instances: int,
    seed: Optional[int] = None,
    reference: Optional[List[Dict]] = None,
    n_range: Tuple[int, int] = (10, 30),
    bin_dim: Tuple[float, float] = (15.0, 15.0),
    item_dim_range: Tuple[float, float] = (2.0, 10.0),
) -> List[Dict]:
    rng = _rng(seed)

    ref_n = _extract_reference_args(reference, "n")
    ref_W = _extract_reference_args(reference, "W")
    ref_H = _extract_reference_args(reference, "H")
    ref_w = _extract_reference_args(reference, "w")
    ref_h = _extract_reference_args(reference, "h")

    n_list = [int(n) for n in ref_n if isinstance(n, int) and n > 0]
    bin_sizes = [
        (float(w), float(h))
        for w, h in zip(ref_W, ref_H)
        if isinstance(w, (int, float)) and isinstance(h, (int, float))
    ]

    width_vals = [val for d in ref_w if isinstance(d, dict) for val in d.values()]
    height_vals = [val for d in ref_h if isinstance(d, dict) for val in d.values()]

    instances: List[Dict] = []
    for idx in range(num_instances):
        n_items = (
            int(rng.choice(n_list)) if n_list else int(rng.integers(n_range[0], n_range[1] + 1))
        )
        if bin_sizes:
            W, H = bin_sizes[int(rng.integers(0, len(bin_sizes)))]
        else:
            W, H = bin_dim

        items: List[Tuple[float, float]] = []
        for _ in range(n_items):
            if height_vals:
                height_i = float(rng.choice(height_vals))
            else:
                height_i = float(rng.uniform(item_dim_range[0], min(item_dim_range[1], H)))

            if width_vals:
                width_i = float(rng.choice(width_vals))
            else:
                width_i = float(rng.uniform(item_dim_range[0], min(item_dim_range[1], W)))

            height_i = min(height_i, H)
            width_i = min(width_i, W)
            items.append((height_i, width_i))

        items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
        h = {i + 1: items_sorted[i][0] for i in range(n_items)}
        w = {i + 1: items_sorted[i][1] for i in range(n_items)}

        instances.append(
            {
                "name": f"2dbp_{idx}_{n_items}",
                "args": {"n": n_items, "W": float(W), "H": float(H), "h": h, "w": w},
            }
        )

    return instances


def generate_random_pdptw_instances(
    num_instances: int,
    seed: Optional[int] = None,
    n_range: Tuple[int, int] = (10, 40),
    k_range: Tuple[int, int] = (2, 8),
    coord_range: Tuple[float, float] = (0.0, 100.0),
    service_time_range: Tuple[float, float] = (5.0, 20.0),
    window_range: Tuple[float, float] = (30.0, 120.0),
    vehicle_ready_range: Tuple[float, float] = (0.0, 20.0),
) -> List[Dict]:
    rng = _rng(seed)

    instances: List[Dict] = []
    for idx in range(num_instances):
        N = int(rng.integers(n_range[0], n_range[1] + 1))
        K = int(rng.integers(k_range[0], k_range[1] + 1))

        coord_lo, coord_hi = coord_range
        pickup = {
            i: (
                float(rng.uniform(coord_lo, coord_hi)),
                float(rng.uniform(coord_lo, coord_hi)),
            )
            for i in range(1, N + 1)
        }
        delivery = {
            i: (
                float(rng.uniform(coord_lo, coord_hi)),
                float(rng.uniform(coord_lo, coord_hi)),
            )
            for i in range(1, N + 1)
        }
        depot = {
            k: (
                float(rng.uniform(coord_lo, coord_hi)),
                float(rng.uniform(coord_lo, coord_hi)),
            )
            for k in range(1, K + 1)
        }

        def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
            return float(math.hypot(a[0] - b[0], a[1] - b[1]))

        d0i: Dict[Tuple[int, int], float] = {}
        diH: Dict[Tuple[int, int], float] = {}
        for k in range(1, K + 1):
            for i in range(1, N + 1):
                d0i[(k, i)] = dist(depot[k], pickup[i])
                diH[(k, i)] = dist(delivery[i], depot[k])

        dij: Dict[Tuple[int, int], float] = {}
        service_times = {
            i: float(rng.uniform(service_time_range[0], service_time_range[1]))
            for i in range(1, N + 1)
        }
        for i in range(1, N + 1):
            dij[(i, i)] = dist(pickup[i], delivery[i]) + service_times[i]
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if i == j:
                    continue
                dij[(i, j)] = dist(delivery[i], pickup[j])

        v = {k: float(rng.uniform(vehicle_ready_range[0], vehicle_ready_range[1])) for k in range(1, K + 1)}

        horizon = max(dij.values()) * 4.0 if dij else 100.0
        tau_earliest: Dict[int, float] = {}
        tau_latest: Dict[int, float] = {}
        for i in range(1, N + 1):
            earliest = float(rng.uniform(0.0, horizon * 0.5))
            window = float(rng.uniform(window_range[0], window_range[1]))
            tau_earliest[i] = earliest
            tau_latest[i] = earliest + window

        instances.append(
            {
                "name": f"pdptw_{idx}_N{N}_K{K}",
                "args": {
                    "K": K,
                    "N": N,
                    "d0i": d0i,
                    "dij": dij,
                    "diH": diH,
                    "v": v,
                    "tau_earliest": tau_earliest,
                    "tau_latest": tau_latest,
                },
            }
        )

    return instances



__all__ = [
    "generate_random_tsp_instances",
    "generate_random_jssp_instances",
    "generate_random_mcnd_instances",
    "generate_random_cwlp_instances",
    "generate_random_mclp_instances",
    "generate_random_2dbp_instances",
    "generate_random_pdptw_instances",
]
