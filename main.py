from tsp_parse import parse_tsp_df
from utils import (
    city_distance,
    calculate_fitness,
    info,
    greedy_nearest_neighbor,
    create_population,
    inversion_mutation,
    new_epoch,
    ordered_crossover,
    population_info,
    run_ga,
    tournament_selection,
)
import random
import matplotlib.pyplot as plt


# task 3
def create_random_solution(df):
    cities = df["id"].tolist()
    random.shuffle(cities)
    return cities


def main():
    df1 = parse_tsp_df("./tsps/berlin52.tsp")
    df2 = parse_tsp_df("./tsps/berlin11_modified.tsp")

    city1 = df1.loc[df1.id == 1].iloc[0]
    city2 = df1.loc[df1.id == 2].iloc[0]

    # task 4
    # dist = city_distance(city1, city2)
    # print("distance between city 1 and 2:", dist)

    # task 3
    # solution = create_random_solution(df2)
    ids_11 = df2["id"].tolist()
    # print("random solution:", solution)
    # if sorted(solution) == sorted(ids_11):
    #     print("all cities included, no duplicates")
    # else:
    #     print("something wrong (missing or duplicate ids)")

    # task 5 + task 6
    # test_solution = create_random_solution(df2)
    # fitness = calculate_fitness(df2, test_solution)
    # info(test_solution, fitness)

    # task 7
    start_id_11 = 1
    greedy_solution_11_single = greedy_nearest_neighbor(df2, start_id_11)
    greedy_fitness_11_single = calculate_fitness(df2, greedy_solution_11_single)
    info(greedy_solution_11_single, greedy_fitness_11_single)

    # task 8 (berlin11)
    # print("==== task 8: greedy from every start (berlin11) ====")
    best_start_11 = None
    best_route_11 = None
    best_score_11 = None

    for start in ids_11:
        route = greedy_nearest_neighbor(df2, start)
        score = calculate_fitness(df2, route)
        # print(f"\nstart city: {start}")
        # info(route, score)
        if best_score_11 is None or score < best_score_11:
            best_score_11 = score
            best_start_11 = start
            best_route_11 = route

    # print("\n==== best greedy result for berlin11 (task 8) ====")
    # print(f"best start city: {best_start_11}")
    # info(best_route_11, best_score_11)

    reference_greedy_score_11 = best_score_11
    # print(f"reference greedy score berlin11: {reference_greedy_score_11}")

    # task 9 (berlin11)
    print("\n==== task 9: 100 random solutions (berlin11) ====")
    best_random_score_11 = None
    best_random_solution_11 = None

    for i in range(100):
        sol = create_random_solution(df2)
        score = calculate_fitness(df2, sol)
        print(f"\nrandom #{i + 1} (berlin11)")
        info(sol, score)
        if best_random_score_11 is None or score < best_random_score_11:
            best_random_score_11 = score
            best_random_solution_11 = sol

    print("\n==== summary task 9 (berlin11) ====")
    print("best random solution (berlin11):")
    info(best_random_solution_11, best_random_score_11)
    print(f"greedy reference score berlin11: {reference_greedy_score_11}")
    if best_random_score_11 < reference_greedy_score_11:
        print("random search found better solution than greedy on berlin11.")
    else:
        print("greedy solution is better or equal to best random on berlin11.")

    # task 10 (berlin52)
    ids_52 = df1["id"].tolist()

    # print("\n==== task 10: greedy from every start (berlin52) ====")
    best_start_52 = None
    best_route_52 = None
    best_score_52 = None

    for start in ids_52:
        route = greedy_nearest_neighbor(df1, start)
        score = calculate_fitness(df1, route)
        # print(f"\nstart city: {start}")
        # info(route, score)
        if best_score_52 is None or score < best_score_52:
            best_score_52 = score
            best_start_52 = start
            best_route_52 = route

    # print("\n==== best greedy result for berlin52 (task 10) ====")
    # print(f"best start city: {best_start_52}")
    # info(best_route_52, best_score_52)
    reference_greedy_score_52 = best_score_52
    # print(f"reference greedy score berlin52: {reference_greedy_score_52}")

    # print("\n==== task 10: 100 random solutions (berlin52) ====")
    best_random_score_52 = None
    best_random_solution_52 = None

    for _ in range(100):
        sol = create_random_solution(df1)
        score = calculate_fitness(df1, sol)
        # print(f"\nrandom #{i + 1} (berlin52)")
        # info(sol, score)
        if best_random_score_52 is None or score < best_random_score_52:
            best_random_score_52 = score
            best_random_solution_52 = sol

    # print("\n==== summary task 10 (berlin52) ====")
    # print("best random solution (berlin52):")
    # info(best_random_solution_52, best_random_score_52)
    # print(f"greedy reference score berlin52: {reference_greedy_score_52}")
    # if best_random_score_52 < reference_greedy_score_52:
    #     print("random search found better solution than greedy on berlin52.")
    # else:
    #     print("greedy solution is better or equal to best random on berlin52.")

    # task 11–12 (populations)
    pop11 = create_population(df2, 50, greedy_solutions=[best_route_11])
    pop52 = create_population(df1, 50, greedy_solutions=[best_route_52])
    # print("population 11 example:", pop11[0])
    # print("population 52 example:", pop52[0])

    # task 13

    # population_info(df2, pop11)
    # population_info(df1, pop52)

    # task 14
    selected_11_a = tournament_selection(df2, pop11, tournament_size=3)
    selected_11_b = tournament_selection(df2, pop11, tournament_size=3)
    selected_52_a = tournament_selection(df1, pop52, tournament_size=3)
    selected_52_b = tournament_selection(df1, pop52, tournament_size=3)
    # print("selected 11 a:")
    # info(selected_11_a, calculate_fitness(df2, selected_11_a))
    # print("selected 11 b:")
    # info(selected_11_b, calculate_fitness(df2, selected_11_b))

    child_11 = ordered_crossover(selected_11_a, selected_11_b)
    child_52 = ordered_crossover(selected_52_a, selected_52_b)
    # print("child berlin11:")
    # info(child_11, calculate_fitness(df2, child_11))
    # print("child berlin52:")
    # info(child_52, calculate_fitness(df1, child_52))

    # task 16
    mutated_child_11 = inversion_mutation(child_11, mutation_prob=0.2)
    mutated_child_52 = inversion_mutation(child_52, mutation_prob=0.2)
    # print("mutated child berlin11:")
    # info(mutated_child_11, calculate_fitness(df2, mutated_child_11))
    # print("mutated child berlin52:")
    # info(mutated_child_52, calculate_fitness(df1, mutated_child_52))

    # task 17
    new_pop11, best_11 = new_epoch(
        df2,
        pop11,
        tournament_size=3,
        crossover_prob=0.8,
        mutation_prob=0.2,
    )

    new_pop52, best_52 = new_epoch(
        df1,
        pop52,
        tournament_size=3,
        crossover_prob=0.8,
        mutation_prob=0.2,
    )

    # print("epoch 1 berlin11:")
    # population_info(df2, new_pop11)
    # print("best individual berlin11:")
    # info(best_11, calculate_fitness(df2, best_11))

    # print("epoch 1 berlin52:")
    # population_info(df1, new_pop52)
    # print("best individual berlin52:")
    # info(best_52, calculate_fitness(df1, best_52))

    # task 18
    epochs = 50

    current_pop11 = pop11
    best_global_11 = min(current_pop11, key=lambda sol: calculate_fitness(df2, sol))
    best_global_11_score = calculate_fitness(df2, best_global_11)
    best_history_11 = []

    for epoch in range(epochs):
        current_pop11, best_epoch_11 = new_epoch(
            df2,
            current_pop11,
            tournament_size=3,
            crossover_prob=0.8,
            mutation_prob=0.2,
        )
        score_epoch_11 = calculate_fitness(df2, best_epoch_11)
        if score_epoch_11 < best_global_11_score:
            best_global_11 = best_epoch_11
            best_global_11_score = score_epoch_11
        best_history_11.append(best_global_11_score)
        # print(f"epoch {epoch + 1} berlin11 best:", best_global_11_score)

    current_pop52 = pop52
    best_global_52 = min(current_pop52, key=lambda sol: calculate_fitness(df1, sol))
    best_global_52_score = calculate_fitness(df1, best_global_52)
    best_history_52 = []

    for epoch in range(epochs):
        current_pop52, best_epoch_52 = new_epoch(
            df1,
            current_pop52,
            tournament_size=3,
            crossover_prob=0.8,
            mutation_prob=0.2,
        )
        score_epoch_52 = calculate_fitness(df1, best_epoch_52)
        if score_epoch_52 < best_global_52_score:
            best_global_52 = best_epoch_52
            best_global_52_score = score_epoch_52
        best_history_52.append(best_global_52_score)
        # print(f"epoch {epoch + 1} berlin52 best:", best_global_52_score)

    # task 19
    plt.figure()
    plt.plot(range(1, epochs + 1), best_history_11)
    plt.xlabel("epoch")
    plt.ylabel("best fitness")
    plt.title("berlin11: best fitness per epoch")
    plt.grid(True)

    plt.figure()
    plt.plot(range(1, epochs + 1), best_history_52)
    plt.xlabel("epoch")
    plt.ylabel("best fitness")
    plt.title("berlin52: best fitness per epoch")
    plt.grid(True)

    plt.show()

    # task 20

    # параметры, которые хотим сравнить
    experiments = [
        {"tournament_size": 3, "crossover_prob": 0.8, "mutation_prob": 0.2},
        {"tournament_size": 3, "crossover_prob": 0.9, "mutation_prob": 0.3},
        {"tournament_size": 5, "crossover_prob": 0.8, "mutation_prob": 0.35},
        {"tournament_size": 5, "crossover_prob": 0.95, "mutation_prob": 0.4},
    ]

    epochs = 50

    plt.figure(figsize=(10, 6))

    for params in experiments:
        hist, _, _ = run_ga(
            df2,
            pop11,
            epochs,
            params["tournament_size"],
            params["crossover_prob"],
            params["mutation_prob"],
        )
        label = (
            f"Ts={params['tournament_size']}  "
            f"Pc={params['crossover_prob']}  "
            f"Pm={params['mutation_prob']}"
        )
        plt.plot(hist, label=label)

    plt.xlabel("epoch")
    plt.ylabel("best fitness")
    plt.title("berlin11 – comparison of GA parameter sets")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
