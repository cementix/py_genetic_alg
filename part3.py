from tsp_parse import parse_tsp_df
from utils import (
    find_best_greedy,
    print_statistics,
    run_ga_once,
    run_random_search,
)


def main():
    df = parse_tsp_df("./tsps/berlin52.tsp")

    best_greedy_route, best_greedy_score, greedy_scores = find_best_greedy(df)
    best_random_score, random_scores = run_random_search(df, 1000)

    ga_params = {"tournament_size": 7, "crossover_prob": 0.8, "mutation_prob": 0.25}
    ga_scores = [
        run_ga_once(
            df,
            best_greedy_route,
            population_size=50,
            epochs=150,
            tournament_size=ga_params["tournament_size"],
            crossover_prob=ga_params["crossover_prob"],
            mutation_prob=ga_params["mutation_prob"],
        )
        for _ in range(10)
    ]

    print("\n=============== PART 3 RESULTS (BERLIN52) ===============")

    print("\nGenetic Algorithm (10 runs)")
    print("run\tfitness")
    for i, score in enumerate(ga_scores, start=1):
        print(f"{i}\t{score:.2f}")
    print_statistics("GA", ga_scores)

    print("\nGreedy Algorithm (all starting cities)")
    print("best 5 results:")
    for i, score in enumerate(sorted(greedy_scores)[:5], start=1):
        print(f"{i}\t{score:.2f}")
    print_statistics("Greedy", greedy_scores)

    print("\nRandom Search (1000 runs)")
    print_statistics("Random", random_scores, best_random_score)

    print("\n=================== DONE ===================\n")


if __name__ == "__main__":
    main()
