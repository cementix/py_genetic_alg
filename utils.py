import math


def city_distance(city1, city2):
    x1, y1 = city1["x"], city1["y"]
    x2, y2 = city2["x"], city2["y"]

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_fitness(df, solution):
    total = 0.0

    city_map = {row.id: row for _, row in df.iterrows()}

    for i in range(len(solution)):
        c1 = city_map[solution[i]]
        c2 = city_map[solution[(i + 1) % len(solution)]]
        total += city_distance(c1, c2)

    return total


def info(solution, fitness_value):
    route_str = " ".join(str(c) for c in solution)
    print(f"route: {route_str}")
    print(f"fitness: {fitness_value}")
