import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib.animation import FuncAnimation

# TSP problem : finding the shortest path to visit all the cities exactly once
# Random Search

# Input Data (1000 Coordinate Points and display it with matplotlib)

cities = []

with open('tsp.txt', 'r') as file:
    for line in file:
        x, y = map(float, line.strip().split(','))
        cities.append((x, y))


def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

def total_distance(order, cities):
    dist = 0
    for i in range(len(order) - 1):
        dist += distance(cities[order[i]], cities[order[i+1]])

    # connect the last city to the first city
    dist += distance(cities[order[-1]], cities[order[0]])

    return dist

def rs_tsp(cities, iterations = 1000000):
    best_order = list(range(len(cities))) # shortest order to visit the cities
    best_distance = total_distance(best_order, cities)
    distance_over_time = [best_distance]

    for _ in range(iterations):
        random_order = best_order.copy()
        random.shuffle(random_order)
        current_distance = total_distance(random_order, cities)

        if current_distance < best_distance:
            best_distance = current_distance
            best_order = random_order
        
        distance_over_time.append(best_distance)

    return best_order, best_distance, distance_over_time

def rmhc_tsp(cities, iterations = 1000000):
    best_order = list(range(len(cities)))
    best_distance = total_distance(best_order, cities)
    distance_over_time = [best_distance]

    for _ in range(iterations):
        mutated_order = swap_mutation(best_order)
        current_distance = total_distance(mutated_order, cities)

        if current_distance < best_distance:
            best_distance = current_distance
            best_order = mutated_order

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

        fitnesses = [1 / total_distance(path, cities) for path in population]
        total_fitness = sum(fitnesses)
        mating_pool = []

        # # Roulette Wheel Selection -> mating_pool reflects fitness
        # for _ in range(pop_size):
        #     pick = random.uniform(0, total_fitness)
        #     current = 0
        #     for idx, path in enumerate(population):
        #         current += fitnesses[idx]
        #         if current > pick:
        #             mating_pool.append(path)
        #             break

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


# def incremental_ga_tsp(cities, pop_size=50, generations=2000, crossover_prob=0.5, mutation_prob=0.1):
#     # Initially select a subset of 4 points randomly
#     subset_cities = random.sample(cities, 4)
#     remaining_cities = [city for city in cities if city not in subset_cities]

#     # Initial population for the subset of points
#     population = [list(range(len(subset_cities))) for _ in range(pop_size)]
#     for path in population:
#         random.shuffle(path)

#     best_order = None
#     best_distance = float('inf')
#     distance_over_time = []

#     while len(remaining_cities) >= 2:  # Ensure there are at least two cities to add
#         # Solve the TSP for the current subset of cities using GA
#         best_order, best_distance, _ = ga_tsp(subset_cities, initial_population=population, pop_size=pop_size, generations=generations, crossover_prob=crossover_prob, mutation_prob=mutation_prob)
#         distance_over_time.append(best_distance)

#         # Randomly add two new cities to the subset
#         new_cities = random.sample(remaining_cities, 2)
#         subset_cities.extend(new_cities)
#         for city in new_cities:
#             remaining_cities.remove(city)

#         # Update the population to include the new cities
#         for path in population:
#             for new_city_idx in range(len(subset_cities) - 2, len(subset_cities)):  # Last two added cities
#                 # Introduce each new city at a random position in the genome
#                 new_position = random.randint(0, len(path))
#                 path.insert(new_position, new_city_idx)

#         print(f"Now solving for {len(subset_cities)} cities...") 

#     return best_order, best_distance, distance_over_time


# incremental_ga_best_order, incremental_ga_best_distance, incremental_ga_distance_over_time = incremental_ga_tsp(cities)


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

# # Plotting for Incremental Genetic Algorithm
# plt.subplot(1, 4, 4)
# plt.scatter(x_set, y_set, color='orange')
# plt.plot([x_set[i] for i in incremental_ga_best_order] + [x_set[incremental_ga_best_order[0]]],
#          [y_set[i] for i in incremental_ga_best_order] + [y_set[incremental_ga_best_order[0]]], color='blue')
# plt.title(f"Incremental Genetic Algorithm TSP: Shortest Path Distance = {incremental_ga_best_distance:.2f}")
# plt.xlabel('x set')
# plt.ylabel('y set')

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
