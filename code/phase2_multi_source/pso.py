"""Particle swarm optimization (PSO) wrapped around the MILP.

For each candidate source, the PSO searches the installed power and the gas
share; every particle is scored by solving the MILP of milp.py.
"""
import numpy as np
import multiprocessing
from milp import optimize_for_given_source, ENR_sourceMaxCap, cGaz, cMax, number_of_nodes


#  Helper computations

Q_DELIVERED_BY_SOURCE = [
    17131436.0,
    17493313.0,
    17203100.0,
    17440403.0,
    17313950.0,
    17083321.0,
    16882083.0,
    16793297.0,
    17459270.0,
    16473201.0,
]


def calculate_pgaz_posteriori(P_installed, source_id):
    Q_deliverd = Q_DELIVERED_BY_SOURCE[source_id]
    if P_installed <= ENR_sourceMaxCap:
        return 0
    else:
        c = (-(10**-7)*ENR_sourceMaxCap -(10**-10)*Q_deliverd)/(1-(10**-7)*ENR_sourceMaxCap -(10**-10)*Q_deliverd)
        a = (10**-7)*(1-c)
        b = (10**-10)*(1-c)
        result = min(1, a*P_installed+b*Q_deliverd+c)
        return result


def calculate_error(P_installed, Q_deliverd, gaz_part_i, source_id):
    Pgaz = calculate_pgaz_posteriori(P_installed, source_id)
    new = Pgaz*Q_deliverd*cGaz
    previous = gaz_part_i*Q_deliverd*cGaz
    return abs(new-previous)


def precalculate_score(X, source_id, A=1, B=100):
    result_for_given_power = optimize_for_given_source(source_id, X[0], X[1])
    result_for_given_power["error"] = calculate_error(X[0], result_for_given_power["Q_deliverd"], X[1], source_id)
    return A*result_for_given_power["total_cost"] + B*result_for_given_power["error"]


#  Top-level evaluation function (required by multiprocessing)

def evaluate_particle(args):
    """Score one particle. Must be top-level so that it can be pickled."""
    X, source_id = args
    return precalculate_score(X, source_id)


#  Parallel PSO loop

def run_parallel_pso(source_id, min_values, max_values,
                     swarm_size=4, iterations=5,
                     w=0.9, c1=2.0, c2=2.0, seed=42,
                     n_cores=4):
    """
    PSO with the particles evaluated in parallel through multiprocessing.Pool.

    Parameters
    ----------
    source_id   : int    index of the source (0-9)
    min_values  : tuple  lower bounds (P_installed_min, gaz_part_min)
    max_values  : tuple  upper bounds (P_installed_max, gaz_part_max)
    swarm_size  : int    number of particles (ideally = n_cores)
    iterations  : int    number of PSO iterations
    w           : float  inertia
    c1, c2      : float  cognitive / social coefficients
    seed        : int    random seed for reproducibility
    n_cores     : int    number of parallel processes

    Returns
    -------
    dict with best_position (P_installed, gaz_part_i) and best_score
    """
    rng = np.random.default_rng(seed)
    n_dims = len(min_values)
    min_v = np.array(min_values, dtype=float)
    max_v = np.array(max_values, dtype=float)

    # Initial positions and velocities
    positions  = rng.uniform(min_v, max_v, size=(swarm_size, n_dims))
    velocities = np.zeros((swarm_size, n_dims))

    # Initial evaluation (parallel)
    print(f"[PSO] Initialization - evaluating {swarm_size} particles on {n_cores} cores...")
    with multiprocessing.Pool(processes=n_cores) as pool:
        scores = pool.map(evaluate_particle,
                          [(positions[i], source_id) for i in range(swarm_size)])

    personal_best_pos    = positions.copy()
    personal_best_scores = list(scores)

    global_best_idx   = int(np.argmin(personal_best_scores))
    global_best_pos   = personal_best_pos[global_best_idx].copy()
    global_best_score = personal_best_scores[global_best_idx]

    print(f"[PSO] Initial best score: {global_best_score:.4f}")

    # PSO loop
    for iteration in range(iterations):
        print(f"\n[PSO] Iteration {iteration+1}/{iterations}")

        r1 = rng.random((swarm_size, n_dims))
        r2 = rng.random((swarm_size, n_dims))

        # Update the velocities
        velocities = (
            w * velocities
            + c1 * r1 * (personal_best_pos - positions)
            + c2 * r2 * (global_best_pos   - positions)
        )

        # Update the positions and clamp them to the bounds
        positions = np.clip(positions + velocities, min_v, max_v)

        # Parallel evaluation
        with multiprocessing.Pool(processes=n_cores) as pool:
            scores = pool.map(evaluate_particle,
                              [(positions[i], source_id) for i in range(swarm_size)])

        # Update the personal bests and the global best
        for i in range(swarm_size):
            if scores[i] < personal_best_scores[i]:
                personal_best_scores[i] = scores[i]
                personal_best_pos[i]    = positions[i].copy()

        new_best_idx = int(np.argmin(personal_best_scores))
        if personal_best_scores[new_best_idx] < global_best_score:
            global_best_score = personal_best_scores[new_best_idx]
            global_best_pos   = personal_best_pos[new_best_idx].copy()

        print(f"[PSO] Best score: {global_best_score:.4f} "
              f"| P_installed={global_best_pos[0]:.2f}, gaz_part={global_best_pos[1]:.4f}")

    return {
        "best_position" : global_best_pos.tolist(),
        "best_score"    : global_best_score,
        "P_installed"   : global_best_pos[0],
        "gaz_part_i"    : global_best_pos[1],
    }


#  Entry point

if __name__ == "__main__":
    all_results = []

    for source_id in range(10):
        print("\n" + "="*50)
        print(f"PSO SOURCE S{source_id + 1} (source_id={source_id})")
        print("="*50)

        maxConsumption = sum(cMax[(number_of_nodes + source_id, i)]
                             for i in range(number_of_nodes))

        result = run_parallel_pso(
            source_id  = source_id,
            min_values = (0, 0),
            max_values = (maxConsumption, 1),
            swarm_size = 4,
            iterations = 5,
            w          = 0.9,
            c1         = 2.0,
            c2         = 2.0,
            seed       = 42,
            n_cores    = 4,
        )
        result["source_id"] = source_id
        all_results.append(result)

    result = min(all_results, key=lambda r: r["best_score"])

    final_result = optimize_for_given_source(
        result["source_id"],
        result["P_installed"],
        result["gaz_part_i"],
    )

    print("\n" + "="*50)
    print("PSO RESULT")
    print("="*50)
    print(f"Best source          : S{result['source_id'] + 1} (source_id={result['source_id']})")
    print(f"Best P_installed     : {result['P_installed']:.2f} kW")
    print(f"Best gas share       : {result['gaz_part_i']:.4f}")
    print(f"Minimum score        : {result['best_score']:.4f}")
    print(f"Final MILP status    : {final_result['status']}")
    print(f"Final MILP cost      : {final_result['total_cost']:.4f}")
