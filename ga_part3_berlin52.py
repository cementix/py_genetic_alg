import random
import statistics as stats

from tsp_parse import parse_tsp_df
from utils import (
    calculate_fitness,
    greedy_nearest_neighbor,
    create_population,
    new_epoch,
)


# note: create_random_solution is already implemented in utils (task 4),
# here it is duplicated for local use.
# task 4 / 9 / 10 – random solution generator for random search
def create_random_solution(df):
    """Create a random permutation of cities."""
    cities = df["id"].tolist()
    random.shuffle(cities)
    return cities


# this is a helper for experiments: GA run with fixed params and greedy seed
# mainly used for task 18 (GA over epochs) and later report experiments
def run_ga_once(
    df,
    greedy_route,
    population_size,
    epochs,
    tournament_size,
    crossover_prob,
    mutation_prob,
):
    """Run GA once and return best fitness found."""
    population = create_population(df, population_size, greedy_solutions=[greedy_route])
    best_solution = min(population, key=lambda sol: calculate_fitness(df, sol))
    best_fitness = calculate_fitness(df, best_solution)

    for _ in range(epochs):
        population, epoch_best = new_epoch(
            df, population, tournament_size, crossover_prob, mutation_prob
        )
        epoch_fitness = calculate_fitness(df, epoch_best)
        if epoch_fitness < best_fitness:
            best_fitness = epoch_fitness
            best_solution = epoch_best

    return best_fitness


# task 7 / 8 – greedy algorithm from all starting cities and best greedy baseline
def find_best_greedy(df):
    """Find best greedy solution from all starting cities."""
    city_ids = df["id"].tolist()
    scores = []
    best_route = None
    best_score = None

    for start_id in city_ids:
        route = greedy_nearest_neighbor(df, start_id)
        score = calculate_fitness(df, route)
        scores.append(score)
        if best_score is None or score < best_score:
            best_score = score
            best_route = route

    return best_route, best_score, scores


# task 9 / 10 – random search baseline (many random solutions)
def run_random_search(df, num_runs):
    """Run random search and return best score and all scores."""
    scores = []
    best_score = None

    for _ in range(num_runs):
        solution = create_random_solution(df)
        score = calculate_fitness(df, solution)
        scores.append(score)
        if best_score is None or score < best_score:
            best_score = score

    return best_score, scores


# task 21 – printing statistics used in the final report
def print_statistics(name, scores, best_score=None):
    """Print formatted statistics for a set of scores."""
    mean = stats.mean(scores)
    std = stats.stdev(scores)
    variance = stats.variance(scores)
    min_score = min(scores)
    max_score = max(scores)

    print(f"\n{name} statistics:")
    print(f"mean       = {mean:.2f}")
    print(f"std dev    = {std:.2f}")
    print(f"variance   = {variance:.2f}")
    print(f"min        = {min_score:.2f}")
    print(f"max        = {max_score:.2f}")
    if best_score is not None:
        print(f"best       = {best_score:.2f}")


# main experiment runner for:
# - task 8 / 10: greedy on berlin52
# - task 9 / 10: random search on berlin52
# - task 18 / 21: GA runs + statistics comparison
def main():
    print("starting part 3 script...", flush=True)

    print("loading berlin52...", flush=True)
    df = parse_tsp_df("./tsps/berlin52.tsp")
    print(f"berlin52 loaded, cities: {len(df)}", flush=True)

    # greedy baseline (tasks 7–8 / 10)
    print("running greedy from all starting cities...", flush=True)
    best_greedy_route, best_greedy_score, greedy_scores = find_best_greedy(df)
    greedy_best5 = sorted(greedy_scores)[:5]
    print("finished greedy.", flush=True)

    # random search baseline (tasks 9–10)
    print("running random search (1000 runs)...", flush=True)
    best_random_score, random_scores = run_random_search(df, 1000)
    print("finished random search.", flush=True)

    # GA experiments (task 18 + report)
    print("running GA (10 runs)...", flush=True)
    ga_params = {"tournament_size": 7, "crossover_prob": 0.8, "mutation_prob": 0.25}
    ga_scores = []

    for i in range(10):
        score = run_ga_once(
            df,
            best_greedy_route,
            population_size=50,
            epochs=150,
            tournament_size=ga_params["tournament_size"],
            crossover_prob=ga_params["crossover_prob"],
            mutation_prob=ga_params["mutation_prob"],
        )
        ga_scores.append(score)
        print(f"ga run {i + 1}/10 done, best = {score:.2f}", flush=True)

    print("finished GA.", flush=True)

    print("\n=============== PART 3 RESULTS (BERLIN52) ===============")

    print("\nGenetic Algorithm (10 runs)")
    print("run\tfitness")
    for i, score in enumerate(ga_scores, start=1):
        print(f"{i}\t{score:.2f}")
    print_statistics("GA", ga_scores)

    print("\nGreedy Algorithm (all starting cities)")
    print("best 5 results:")
    for i, score in enumerate(greedy_best5, start=1):
        print(f"{i}\t{score:.2f}")
    print_statistics("Greedy", greedy_scores)

    print("\nRandom Search (1000 runs)")
    print_statistics("Random", random_scores, best_random_score)

    print("\n=================== DONE ===================\n")


if __name__ == "__main__":
    main()
