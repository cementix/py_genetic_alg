import math
import random


# task 4
def city_distance(city1, city2):
    x1, y1 = city1["x"], city1["y"]
    x2, y2 = city2["x"], city2["y"]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# task 5
def calculate_fitness(df, solution):
    total = 0.0
    city_map = {row.id: row for _, row in df.iterrows()}
    for i in range(len(solution)):
        c1 = city_map[solution[i]]
        c2 = city_map[solution[(i + 1) % len(solution)]]
        total += city_distance(c1, c2)
    return total


# task 6
def info(solution, fitness_value):
    route_str = " ".join(str(c) for c in solution)
    print(f"route: {route_str}")
    print(f"fitness: {fitness_value}")


# task 7
def greedy_nearest_neighbor(df, start_id):
    city_map = {row.id: row for _, row in df.iterrows()}
    unvisited = set(df["id"].tolist())
    route = [start_id]
    unvisited.remove(start_id)
    current = start_id
    while unvisited:
        current_city = city_map[current]
        nxt = min(unvisited, key=lambda cid: city_distance(current_city, city_map[cid]))
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return route


# task 11–12
def create_population(df, size, greedy_solutions=None):
    ids = df["id"].tolist()
    population = []
    if greedy_solutions:
        for r in greedy_solutions:
            population.append(r)
    while len(population) < size:
        s = ids.copy()
        random.shuffle(s)
        population.append(s)
    return population
