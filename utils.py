import math
import random
import matplotlib.pyplot as plt
import copy
import statistics as stats


_CITY_COORDS_CACHE = {}


# task 1a – helper for easy access to city coordinates
def _get_city_coords(df):
    """Return cached map: city_id -> (x, y) for given df."""
    key = id(df)
    if key not in _CITY_COORDS_CACHE:
        _CITY_COORDS_CACHE[key] = {
            int(row.id): (float(row.x), float(row.y)) for _, row in df.iterrows()
        }
    return _CITY_COORDS_CACHE[key]


# task 2 – distance function between two cities
def city_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.hypot(x2 - x1, y2 - y1)


# task 4 – random solution generator (used in tasks 9–10)
def create_random_solution(df):
    """Create a random permutation of cities."""
    cities = df["id"].tolist()
    random.shuffle(cities)
    return cities


# task 5 – fitness function (total route length)
def calculate_fitness(df, solution):
    """Calculate total route distance for a TSP solution."""
    coords = _get_city_coords(df)
    distance = city_distance
    get = coords.get

    prev_id = solution[-1]
    x1, y1 = get(prev_id)
    total = 0.0

    for city_id in solution:
        x2, y2 = get(city_id)
        total += distance(x1, y1, x2, y2)
        x1, y1 = x2, y2

    return total


# task 6 – info function: print route + score
def info(solution, fitness_value):
    """Print solution route and fitness value."""
    route_str = " ".join(str(city) for city in solution)
    print(f"route: {route_str}")
    print(f"fitness: {fitness_value}")


# task 7 – greedy nearest-neighbor algorithm
def greedy_nearest_neighbor(df, start_id):
    """Build a route using nearest neighbor heuristic from start_id."""
    coords = _get_city_coords(df)
    distance = city_distance
    get = coords.get

    unvisited = set(coords.keys())
    route = [start_id]
    unvisited.remove(start_id)
    current = start_id

    while unvisited:
        x1, y1 = get(current)
        nearest_city = min(
            unvisited, key=lambda city_id: distance(x1, y1, *get(city_id))
        )
        route.append(nearest_city)
        unvisited.remove(nearest_city)
        current = nearest_city

    return route


# task 8 – greedy runs for all starting cities (best greedy baseline)
def find_best_greedy(df):
    """
    Run greedy nearest-neighbor from every possible start.
    Return:
        best_route, best_score, all_scores
    """
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


# task 9/10 – random search baseline (100 random runs or more)
def run_random_search(df, num_runs):
    """
    Run random search: many random routes.
    Return best route, best score, all scores.
    """
    scores = []
    best_solution = None
    best_score = None

    for _ in range(num_runs):
        solution = create_random_solution(df)
        score = calculate_fitness(df, solution)
        scores.append(score)

        if best_score is None or score < best_score:
            best_score = score
            best_solution = solution

    return best_solution, best_score, scores


# task 12 – initial population generation (with optional greedy individuals)
def create_population(df, size, greedy_solutions=None):
    """Create initial population of random permutations."""
    city_ids = df["id"].tolist()
    population = [route[:] for route in greedy_solutions] if greedy_solutions else []

    while len(population) < size:
        route = city_ids[:]
        random.shuffle(route)
        population.append(route)

    return population


# task 13 – population info (size, best, median, worst)
def population_info(df, population):
    """Print statistics about population fitness."""
    fitness_values = sorted(calculate_fitness(df, sol) for sol in population)
    size = len(population)

    median_idx = size // 2
    median = (
        fitness_values[median_idx]
        if size % 2 == 1
        else (fitness_values[median_idx - 1] + fitness_values[median_idx]) / 2.0
    )

    print(f"population size: {size}")
    print(f"best fitness: {fitness_values[0]}")
    print(f"median fitness: {median}")
    print(f"worst fitness: {fitness_values[-1]}")


# task 14 – tournament selection
def tournament_selection(df, population, tournament_size):
    """Select best individual from random tournament."""
    candidates = random.sample(population, tournament_size)
    best = min(candidates, key=lambda sol: calculate_fitness(df, sol))
    return best[:]


# task 15 – ordered crossover (OX)
def ordered_crossover(parent1, parent2):
    """Perform ordered crossover (OX) between two parents."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size

    child[start : end + 1] = parent1[start : end + 1]
    used = set(child[start : end + 1])

    parent_idx = (end + 1) % size
    child_idx = (end + 1) % size
    remaining = size - (end - start + 1)

    while remaining > 0:
        gene = parent2[parent_idx]
        if gene not in used:
            child[child_idx] = gene
            used.add(gene)
            child_idx = (child_idx + 1) % size
            remaining -= 1
        parent_idx = (parent_idx + 1) % size

    return child


# task 16 – mutation (inversion)
def inversion_mutation(individual, mutation_prob):
    """Apply inversion mutation with given probability."""
    if random.random() >= mutation_prob:
        return individual
    mutated = individual[:]
    i, j = sorted(random.sample(range(len(mutated)), 2))
    mutated[i : j + 1] = reversed(mutated[i : j + 1])
    return mutated


# task 17 – epoch creation (selection → crossover → mutation)
def new_epoch(df, population, tournament_size, crossover_prob, mutation_prob):
    """Generate new population and return best solution found."""
    fitness_values = [calculate_fitness(df, sol) for sol in population]
    best_idx = min(range(len(population)), key=lambda i: fitness_values[i])
    best_solution = population[best_idx][:]
    best_fitness = fitness_values[best_idx]

    new_population = []
    pop_size = len(population)

    while len(new_population) < pop_size:
        parent1 = tournament_selection(df, population, tournament_size)
        parent2 = tournament_selection(df, population, tournament_size)

        offspring = (
            ordered_crossover(parent1, parent2)
            if random.random() < crossover_prob
            else parent1[:]
        )
        offspring = inversion_mutation(offspring, mutation_prob)
        new_population.append(offspring)

        offspring_fitness = calculate_fitness(df, offspring)
        if offspring_fitness < best_fitness:
            best_fitness = offspring_fitness
            best_solution = offspring

    return new_population, best_solution


# task 18 – full GA loop across epochs + best tracking
def run_ga(
    df, initial_population, epochs, tournament_size, crossover_prob, mutation_prob
):
    """Run genetic algorithm and return history, best solution, and best score."""
    population = initial_population
    fitness_values = [calculate_fitness(df, sol) for sol in population]
    best_idx = min(range(len(population)), key=lambda i: fitness_values[i])
    best_solution = population[best_idx][:]
    best_score = fitness_values[best_idx]
    history = [best_score]

    for _ in range(epochs):
        population, epoch_best = new_epoch(
            df, population, tournament_size, crossover_prob, mutation_prob
        )
        epoch_fitness = calculate_fitness(df, epoch_best)
        if epoch_fitness < best_score:
            best_solution = epoch_best[:]
            best_score = epoch_fitness
        history.append(best_score)

    return history, best_solution, best_score


# additional helper for experiments (not tied to specific task number)
def run_ga_once(
    df,
    greedy_route,
    population_size,
    epochs,
    tournament_size,
    crossover_prob,
    mutation_prob,
):
    """
    Run GA once with given params, seeded with greedy solution.
    """
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

    return best_fitness, best_solution


# task 19 – fitness history for plotting
def run_ga_with_history(
    df, population, epochs, tournament_size, crossover_prob, mutation_prob
):
    """
    Run GA and track best fitness across epochs.
    """
    best_solution = min(population, key=lambda sol: calculate_fitness(df, sol))
    best_fitness = calculate_fitness(df, best_solution)
    history = [best_fitness]

    for _ in range(epochs):
        population, epoch_best = new_epoch(
            df, population, tournament_size, crossover_prob, mutation_prob
        )
        epoch_fitness = calculate_fitness(df, epoch_best)
        if epoch_fitness < best_fitness:
            best_fitness = epoch_fitness
            best_solution = epoch_best
        history.append(best_fitness)

    return history, best_solution


# task 19 – plot: best score vs epoch
def plot_fitness_history(history, title):
    """Plot fitness history over epochs for a single experiment."""
    plt.figure()
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("epoch")
    plt.ylabel("best fitness")
    plt.title(title)
    plt.grid(True)


# task 19/20 – experiment wrapper for plotting and comparisons
def run_single_experiment(df, initial_pop, params, epochs):
    """Run single GA experiment with given parameters."""
    return run_ga(
        df,
        initial_pop,
        epochs,
        params["tournament_size"],
        params["crossover_prob"],
        params["mutation_prob"],
    )


# task 20 – comparison plot for different parameter sets
def plot_parameter_comparison(df, population, experiments, epochs, title):
    """
    Compare GA parameter sets on one plot.
    """
    plt.figure(figsize=(10, 6))

    for params in experiments:
        history, _, _ = run_ga(
            df,
            copy.deepcopy(population),
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
        plt.plot(range(1, epochs + 1), history, label=label)

    plt.xlabel("epoch")
    plt.ylabel("best fitness")
    plt.title(title)
    plt.grid(True)
    plt.legend()


# task 21 – statistics printing (used in final report)
def print_statistics(name, scores, best_score=None):
    """Print formatted statistics."""
    mean = stats.mean(scores)
    std = stats.stdev(scores) if len(scores) > 1 else 0.0
    variance = stats.variance(scores) if len(scores) > 1 else 0.0
    min_score = min(scores)
    max_score = max(scores)

    print(f"\n{name} stats")
    print(f"mean       = {mean:.2f}")
    print(f"std dev    = {std:.2f}")
    print(f"variance   = {variance:.2f}")
    print(f"min        = {min_score:.2f}")
    print(f"max        = {max_score:.2f}")
    if best_score is not None:
        print(f"best       = {best_score:.2f}")
