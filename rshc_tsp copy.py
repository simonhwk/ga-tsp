import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool, cpu_count

# TSP problem : finding the shortest path to visit all the cities exactly once
# Random Search

# Input Data (1000 Coordinate Points and display it with matplotlib)

cities = np.loadtxt('tsp.txt', delimiter=',')

def fitness_evaluation(path):
    return 1 / total_distance(path, cities)

def parallel_fitness_evaluation(population):
    with Pool(cpu_count()) as pool:
        fitnesses = pool.map(fitness_evaluation, population)
    return fitnesses

def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

def total_distance(order, cities):
    ordered_cities = cities[order]
    pairwise_distances = np.linalg.norm(ordered_cities - np.roll(ordered_cities, -1, axis=0), axis=1)
    return pairwise_distances.sum()

def rs_evaluation(i, best_order, best_distance):
    random_order = best_order.copy()
    random.shuffle(random_order)
    current_distance = total_distance(random_order, cities)
    if current_distance < best_distance:
        best_distance = current_distance
        best_order = random_order
    return best_order, best_distance

def rs_tsp(cities, iterations = 1000000):
    best_order = list(range(len(cities)))
    best_distance = total_distance(best_order, cities)
    distance_over_time = [best_distance]
    
    with Pool(cpu_count()) as pool:
        results = pool.starmap(rs_evaluation, [(i, best_order, best_distance) for i in range(iterations)])
    
    for order, dist in results:
        if dist < best_distance:
            best_distance = dist
            best_order = order
        distance_over_time.append(best_distance)
    
    return best_order, best_distance, distance_over_time


def rmhc_evaluation(i, best_order, best_distance):
    mutated_order = swap_mutation(best_order)
    current_distance = total_distance(mutated_order, cities)
    if current_distance < best_distance:
        best_distance = current_distance
        best_order = mutated_order
    return best_order, best_distance

def rmhc_tsp(cities, iterations = 1000000):
    best_order = list(range(len(cities)))
    best_distance = total_distance(best_order, cities)
    distance_over_time = [best_distance]
    
    with Pool(cpu_count()) as pool:
        results = pool.starmap(rmhc_evaluation, [(i, best_order, best_distance) for i in range(iterations)])
    
    for order, dist in results:
        if dist < best_distance:
            best_distance = dist
            best_order = order
        distance_over_time.append(best_distance)
    
    return best_order, best_distance, distance_over_time


# Ordered Crossover (OX)
def ordered_crossover(parent1, parent2):
    start_idx, end_idx = sorted(random.sample(range(len(parent1)), 2))
    subset_parent1 = parent1[start_idx:end_idx]
    offspring = [-1] * len(parent1)
    offspring[start_idx:end_idx] = subset_parent1
    pointer = end_idx
    for city in parent2:
        if city not in subset_parent1:
            if pointer >= len(parent1):
                pointer = 0
            offspring[pointer] = city
            pointer += 1
    return offspring

# Used for RMHC and GA
def swap_mutation(path):
    mutated_path = path.copy()
    idx1, idx2 = random.sample(range(len(path)), 2)
    mutated_path[idx1], mutated_path[idx2] = mutated_path[idx2], mutated_path[idx1]
    return mutated_path


# Tournament Selection

def tournament_selection(population, fitnesses, tournament_size):
    selected = []
    for _ in range(len(population)):
        candidates = random.sample(list(enumerate(fitnesses)), tournament_size)
        win_idx, win_fitness = max(candidates, key = lambda item : item[1])
        selected.append(population[win_idx])

    return selected


# Genetic Algorithm for TSP
def ga_tsp(cities, initial_population = None, pop_size=50, generations=20000,crossover_prob=0.5, mutation_prob=0.1):

    if initial_population:
        population = initial_population
    else:
        population = [list(range(len(cities))) for _ in range(pop_size)]
        for path in population:
            random.shuffle(path)

    best_order = None
    best_distance = float('inf')
    distance_over_time = []

    for _ in range(generations):

        fitnesses = parallel_fitness_evaluation(population)
        total_fitness = sum(fitnesses)
        mating_pool = []

        mating_pool = tournament_selection(population, fitnesses, tournament_size=3)

        # Crossover and Mutation
        new_population = []
        for i in range(0, pop_size, 2):

            # Ordered Crossover, obtain offspring
            parent1, parent2 = mating_pool[i], mating_pool[i+1]
            if random.random() < crossover_prob:
                offspring1 = ordered_crossover(parent1, parent2)
                offspring2 = ordered_crossover(parent2, parent1)
            else:
                offspring1, offspring2 = parent1, parent2

            # Swap Mutation on the offspring
            if random.random() < mutation_prob:
                offspring1 = swap_mutation(offspring1)
            if random.random() < mutation_prob:
                offspring2 = swap_mutation(offspring2)

            new_population.extend([offspring1, offspring2])

        population = new_population

        # Distance Calculation
        for path in population:
            current_distance = total_distance(path, cities)
            # print(f"C Distance: {current_distance}")
            if current_distance < best_distance:
                best_distance = current_distance
                best_order = path
                print(f"B Distance: {best_distance}")

            distance_over_time.append(best_distance)

        # distance_over_time.append(best_distance)

        # Move on to next generation

    return best_order, best_distance, distance_over_time


# Running the genetic algorithm with reduced parameters
ga_best_order, ga_best_distance, ga_distance_over_time = ga_tsp(cities)

rs_best_order, rs_best_distance, rs_distance_over_time = rs_tsp(cities)
rmhc_best_order, rmhc_best_distance, rmhc_distance_over_time = rmhc_tsp(cities)

x_set = [city[0] for city in cities]
y_set = [city[1] for city in cities]


# Define Plot
plt.figure(figsize=(24, 5))

# Plotting for Random Search
plt.subplot(1, 3, 1)
plt.scatter(x_set, y_set, color='orange')
plt.plot([x_set[i] for i in rs_best_order] + [x_set[rs_best_order[0]]], 
         [y_set[i] for i in rs_best_order] + [y_set[rs_best_order[0]]], color='blue')
plt.title(f"Random Search Hill Climbing TSP: Shortest Path Distance = {rs_best_distance:.2f}")
plt.xlabel('x set')
plt.ylabel('y set')

# Plotting for Random Mutation Hill Climber
plt.subplot(1, 3, 2)
plt.scatter(x_set, y_set, color='orange')
plt.plot([x_set[i] for i in rmhc_best_order] + [x_set[rmhc_best_order[0]]], 
         [y_set[i] for i in rmhc_best_order] + [y_set[rmhc_best_order[0]]], color='blue')
plt.title(f"Random Mutation Hill Climbing TSP: Shortest Path Distance = {rmhc_best_distance:.2f}")
plt.xlabel('x set')
plt.ylabel('y set')

# Plotting for Genetic Algorithm
plt.subplot(1, 3, 3)
plt.scatter(x_set, y_set, color='orange')
plt.plot([x_set[i] for i in ga_best_order] + [x_set[ga_best_order[0]]],
         [y_set[i] for i in ga_best_order] + [y_set[ga_best_order[0]]], color='blue')
plt.title(f"Genetic Algorithm TSP: Shortest Path Distance = {ga_best_distance:.2f}")
plt.xlabel('x set')
plt.ylabel('y set')

plt.tight_layout()
plt.show()

# Distance comparison plot
plt.figure(figsize=(12, 5))
plt.plot(rs_distance_over_time, label='Random Search', color='green')
plt.plot(rmhc_distance_over_time, label='Random Mutation Hill Climbing', color='orange')
plt.plot(ga_distance_over_time, label='GA (Tournament)', color='blue')
# plt.plot(incremental_ga_distance_over_time, label='Incremental GA', color='purple')
plt.title("Distance Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Distance")
plt.legend(loc="upper right")
plt.xscale('log')
plt.xticks([10**i for i in range(7)], [f'$10^{i}$' for i in range(7)])

plt.tight_layout()
plt.show()
