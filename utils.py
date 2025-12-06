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


# task 13
def population_info(df, population):
    fitness_list = [calculate_fitness(df, sol) for sol in population]
    fitness_list_sorted = sorted(fitness_list)

    size = len(population)
    best = fitness_list_sorted[0]
    worst = fitness_list_sorted[-1]

    mid = size // 2
    if size % 2 == 1:
        median = fitness_list_sorted[mid]
    else:
        median = (fitness_list_sorted[mid - 1] + fitness_list_sorted[mid]) / 2

    print(f"population size: {size}")
    print(f"best fitness: {best}")
    print(f"median fitness: {median}")
    print(f"worst fitness: {worst}")


# task 14
def tournament_selection(df, population, tournament_size):
    candidates = random.sample(population, tournament_size)
    best = min(candidates, key=lambda sol: calculate_fitness(df, sol))
    return best


# task 15
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size

    child[a : b + 1] = parent1[a : b + 1]
    used = set(child[a : b + 1])

    p2_idx = (b + 1) % size
    child_idx = (b + 1) % size

    while None in child:
        gene = parent2[p2_idx]
        if gene not in used:
            child[child_idx] = gene
            used.add(gene)
            child_idx = (child_idx + 1) % size
        p2_idx = (p2_idx + 1) % size

    return child


# task 16
def inversion_mutation(individual, mutation_prob):
    if random.random() >= mutation_prob:
        return individual
    size = len(individual)
    i, j = sorted(random.sample(range(size), 2))
    child = individual.copy()
    child[i : j + 1] = reversed(child[i : j + 1])
    return child


# task 17
def new_epoch(df, population, tournament_size, crossover_prob, mutation_prob):
    new_population = []
    best_solution = min(population, key=lambda sol: calculate_fitness(df, sol))

    pop_size = len(population)

    while len(new_population) < pop_size:
        P1 = tournament_selection(df, population, tournament_size)
        P2 = tournament_selection(df, population, tournament_size)

        if random.random() < crossover_prob:
            O1 = ordered_crossover(P1, P2)
        else:
            O1 = P1.copy()

        O1 = inversion_mutation(O1, mutation_prob)

        new_population.append(O1)

        if calculate_fitness(df, O1) < calculate_fitness(df, best_solution):
            best_solution = O1

    return new_population, best_solution


# task 20


def run_ga(
    df, initial_population, epochs, tournament_size, crossover_prob, mutation_prob
):
    pop = initial_population
    best = min(pop, key=lambda s: calculate_fitness(df, s))
    best_score = calculate_fitness(df, best)

    history = []

    for _ in range(epochs):
        pop, best_epoch = new_epoch(
            df,
            pop,
            tournament_size,
            crossover_prob,
            mutation_prob,
        )
        score = calculate_fitness(df, best_epoch)
        if score < best_score:
            best = best_epoch
            best_score = score

        history.append(best_score)

    return history, best, best_score
