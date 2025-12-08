import matplotlib.pyplot as plt
from tsp_parse import parse_tsp_df
from utils import (
    calculate_fitness,
    greedy_nearest_neighbor,
    create_population,
    new_epoch,
)


# helper for tasks 18 and 22:
# use GA (with tuned parameters) to find best solution for a given instance
def run_ga_best(
    df,
    epochs=200,
    population_size=80,
    tournament_size=7,
    crossover_prob=0.8,
    mutation_prob=0.25,
):
    """Run GA with optimized defaults. Returns (best_fitness, best_route)."""
    city_ids = df["id"].tolist()

    # greedy baseline inside: related to tasks 7–8
    best_greedy = min(
        (greedy_nearest_neighbor(df, start_id) for start_id in city_ids),
        key=lambda route: calculate_fitness(df, route),
    )

    # initial population with greedy solution included – task 12
    population = create_population(df, population_size, greedy_solutions=[best_greedy])
    best_route = best_greedy
    best_fitness = calculate_fitness(df, best_greedy)

    # GA epochs – task 18 (repeating new_epoch and tracking best)
    for _ in range(epochs):
        population, candidate = new_epoch(
            df, population, tournament_size, crossover_prob, mutation_prob
        )
        candidate_fitness = calculate_fitness(df, candidate)
        if candidate_fitness < best_fitness:
            best_fitness = candidate_fitness
            best_route = candidate

    return best_fitness, best_route


# route visualization for final report:
# connected with "graphical results" / extra visualisation beyond tasks 19–20
def plot_route(df, route, title):
    """Plot a TSP route visualization."""
    coords = {int(row.id): (float(row.x), float(row.y)) for _, row in df.iterrows()}

    xs = [coords[city][0] for city in route] + [coords[route[0]][0]]
    ys = [coords[city][1] for city in route] + [coords[route[0]][1]]

    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


# main experiment / solving script:
# - task 18: run GA on berlin11 and berlin52
# - task 22: check which additional files can be solved (kroA100, kroA150)
# - used for final report summary + route plots
def main():
    tsp_files = [
        ("berlin11_modified.tsp", "Berlin11"),
        ("berlin52.tsp", "Berlin52"),
        ("kroA100.tsp", "KroA100"),
        ("kroA150.tsp", "KroA150"),
    ]

    results = {}
    for filename, name in tsp_files:
        print(f"Running GA on {name}...")
        df = parse_tsp_df(f"./tsps/{filename}")
        fitness, route = run_ga_best(df)
        results[name] = (fitness, route)
        print(f"Best for {name}: {fitness:.2f}")

    print("\n===== FINAL RESULTS =====")
    for name, (fitness, _) in results.items():
        print(f"{name}: {fitness:.2f}")

    print("\nPlotting best routes...")
    # visualisation for selected instances (linked to report)
    viz_names = ["Berlin52", "KroA100"]

    for name in viz_names:
        # note: this assumes filenames like "berlin52.tsp", "kroa100.tsp"
        # adjust if your actual filenames differ in case/spelling
        df = parse_tsp_df(f"./tsps/{name.lower()}.tsp")
        fitness, route = results[name]
        plot_route(df, route, f"Best Route — {name} (score={fitness:.2f})")


if __name__ == "__main__":
    main()
