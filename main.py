from tsp_parse import parse_tsp_df
from utils import (
    city_distance,
    calculate_fitness,
    info,
    greedy_nearest_neighbor,
    create_population,
)
import random


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
    # start_id_11 = 1
    # greedy_solution_11_single = greedy_nearest_neighbor(df2, start_id_11)
    # greedy_fitness_11_single = calculate_fitness(df2, greedy_solution_11_single)
    # info(greedy_solution_11_single, greedy_fitness_11_single)

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
    # print("\n==== task 9: 100 random solutions (berlin11) ====")
    # best_random_score_11 = None
    # best_random_solution_11 = None

    # for i in range(100):
    #     sol = create_random_solution(df2)
    #     score = calculate_fitness(df2, sol)
    #     print(f"\nrandom #{i + 1} (berlin11)")
    #     info(sol, score)
    #     if best_random_score_11 is None or score < best_random_score_11:
    #         best_random_score_11 = score
    #         best_random_solution_11 = sol

    # print("\n==== summary task 9 (berlin11) ====")
    # print("best random solution (berlin11):")
    # info(best_random_solution_11, best_random_score_11)
    # print(f"greedy reference score berlin11: {reference_greedy_score_11}")
    # if best_random_score_11 < reference_greedy_score_11:
    #     print("random search found better solution than greedy on berlin11.")
    # else:
    #     print("greedy solution is better or equal to best random on berlin11.")

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


if __name__ == "__main__":
    main()
