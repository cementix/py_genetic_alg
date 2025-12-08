import matplotlib.pyplot as plt

from tsp_parse import parse_tsp_df
from utils import (
    find_best_greedy,
    create_population,
    new_epoch,
    ordered_crossover,
    inversion_mutation,
    plot_fitness_history,
    plot_parameter_comparison,
    print_statistics,
    run_ga_once,
    run_ga_with_history,
    run_random_search,
    tournament_selection,
)


def main():
    # ---------- data loading (task 1: parser usage) ----------
    df_berlin52 = parse_tsp_df("./tsps/berlin52.tsp")
    df_berlin11 = parse_tsp_df("./tsps/berlin11_modified.tsp")

    # print("berlin52 rows:", len(df_berlin52))
    # print("berlin11 rows:", len(df_berlin11))

    # ---------- baselines: greedy + random (tasks 7–10, code only here) ----------

    # task 8 / 10: greedy for all starting cities
    # task 9 / 10: random search comparisons

    # berlin11: greedy and random
    best_route_11, best_score_11, greedy_scores_11 = find_best_greedy(df_berlin11)
    _, best_random_score_11, random_scores_11 = run_random_search(df_berlin11, 100)

    # optional prints for berlin11 baselines
    # print("\nberlin11 greedy best:", best_score_11)
    # print("berlin11 random best:", best_random_score_11)

    # berlin52: greedy and random (1000 runs for report statistics)
    best_route_52, best_score_52, greedy_scores_52 = find_best_greedy(df_berlin52)
    _, best_random_score_52, random_scores_52 = run_random_search(df_berlin52, 1000)

    # optional prints for berlin52 baselines
    # print("\nberlin52 greedy best:", best_score_52)
    # print("berlin52 random best (1000 runs):", best_random_score_52)

    # ---------- initial populations (tasks 11–12, code only) ----------

    # task 11: population = set of solutions
    # task 12: function to create initial population (with greedy individuals)
    pop11 = create_population(df_berlin11, 50, greedy_solutions=[best_route_11])
    pop52 = create_population(df_berlin52, 50, greedy_solutions=[best_route_52])

    # optional population info prints (task 13)
    # population_info(df_berlin11, pop11)
    # population_info(df_berlin52, pop52)

    # ---------- selection / crossover / mutation demo (tasks 14–16, code only) ----------

    # task 14: selection function (tournament here)
    selected_11_a = tournament_selection(df_berlin11, pop11, tournament_size=3)
    selected_11_b = tournament_selection(df_berlin11, pop11, tournament_size=3)
    selected_52_a = tournament_selection(df_berlin52, pop52, tournament_size=3)
    selected_52_b = tournament_selection(df_berlin52, pop52, tournament_size=3)

    # task 15: crossover (ordered crossover)
    child_11 = ordered_crossover(selected_11_a, selected_11_b)
    child_52 = ordered_crossover(selected_52_a, selected_52_b)

    # task 16: mutation (inversion with probability)
    inversion_mutation(child_11, mutation_prob=0.2)
    inversion_mutation(child_52, mutation_prob=0.2)

    # optional prints for single GA operators
    # print("selected_11_a:", selected_11_a)
    # print("selected_11_b:", selected_11_b)
    # print("child_11:", child_11)

    # ---------- single epoch demo (task 17, code only) ----------

    # task 17: new epoch = selection → crossover → mutation → new population
    new_pop11, best_11 = new_epoch(df_berlin11, pop11, 3, 0.8, 0.2)
    new_pop52, best_52 = new_epoch(df_berlin52, pop52, 3, 0.8, 0.2)

    # optional prints for single epoch demo
    # print("epoch 1 berlin11 best fitness:", calculate_fitness(df_berlin11, best_11))
    # print("epoch 1 berlin52 best fitness:", calculate_fitness(df_berlin52, best_52))

    # ---------- part 2: base GA run + evolution plots (tasks 18–19) ----------

    # task 18: repeat epochs and track best solution over time
    # task 19: graphical form – best score as a function of epoch
    epochs = 150
    history_11, _ = run_ga_with_history(df_berlin11, pop11, epochs, 3, 0.8, 0.2)
    history_52, _ = run_ga_with_history(df_berlin52, pop52, epochs, 3, 0.8, 0.2)

    # plotting code for part 2 (task 19 – no prints)
    plot_fitness_history(history_11, "berlin11: best fitness per epoch")
    plot_fitness_history(history_52, "berlin52: best fitness per epoch")
    plt.show()

    # ---------- part 2: parameter comparison (task 20) ----------

    # task 20: code to compare quality for different initial parameters
    experiments = [
        {"tournament_size": 2, "crossover_prob": 0.8, "mutation_prob": 0.3},
        {"tournament_size": 3, "crossover_prob": 0.8, "mutation_prob": 0.3},
        {"tournament_size": 5, "crossover_prob": 0.8, "mutation_prob": 0.3},
        {"tournament_size": 7, "crossover_prob": 0.8, "mutation_prob": 0.3},
    ]

    # plotting code for comparison (task 20 – no prints)
    plot_parameter_comparison(
        df_berlin11, pop11, experiments, 150, "berlin11 – GA parameter comparison"
    )
    plot_parameter_comparison(
        df_berlin52, pop52, experiments, 150, "berlin52 – GA parameter comparison"
    )
    plt.show()

    # ---------- part 3: GA vs greedy vs random on berlin52 (code side) ----------

    # this block is used for report (tasks 18, 21, 22: comparison and discussion)
    best_params = {"tournament_size": 7, "crossover_prob": 0.8, "mutation_prob": 0.25}
    ga_scores_52 = []
    ga_routes_52 = []

    # run GA 10 times with chosen best parameters
    for _ in range(10):
        score, route = run_ga_once(
            df_berlin52,
            best_route_52,
            population_size=50,
            epochs=150,
            tournament_size=best_params["tournament_size"],
            crossover_prob=best_params["crossover_prob"],
            mutation_prob=best_params["mutation_prob"],
        )
        ga_scores_52.append(score)
        ga_routes_52.append(route)

    # ---------- part 3: prints for report (tasks 21–22: reporting and analysis) ----------

    print("\n================ part 3: comparison on berlin52 ================")

    print("\nGA (10 runs, Ts=7, Pc=0.8, Pm=0.25)")
    print("run\tbest_fitness")
    for i, score in enumerate(ga_scores_52, start=1):
        print(f"{i}\t{score:.2f}")
    print_statistics("GA", ga_scores_52)

    print("\nGreedy (all starting cities, berlin52)")
    print("best 5 results:")
    print("rank\tfitness")
    for i, score in enumerate(sorted(greedy_scores_52)[:5], start=1):
        print(f"{i}\t{score:.2f}")
    print_statistics("Greedy", greedy_scores_52)

    print("\nRandom search (1000 runs, berlin52)")
    print_statistics("Random", random_scores_52, best_random_score_52)

    print("\n================ end of part 3 ====================\n")


if __name__ == "__main__":
    main()
