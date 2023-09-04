import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib.animation import FuncAnimation

# TSP problem : finding the shortest path to visit all the cities exactly once
# Random Search

# Input Data (1000 Coordinate Points and display it with matplotlib)

x_set = []
y_set = []

with open('tsp.txt', 'r') as file:
    for line in file:
        x, y = map(float, line.strip().split(','))
        x_set.append(x)
        y_set.append(y)


def distance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

def total_distance(order, x_set, y_set):
    dist = 0
    for i in range(len(order) - 1):
        dist += distance(x_set[order[i]], y_set[order[i]], x_set[order[i+1]], y_set[order[i+1]])

    # connect the last city to the first city
    dist += distance(x_set[order[-1]], y_set[order[-1]], x_set[order[0]], y_set[order[0]])

    return dist



def rs_tsp(x_set, y_set, iterations = 100000):
    best_order = list(range(len(x_set))) # shortest order to visit the cities
    best_distance = total_distance(best_order, x_set, y_set)
    distance_over_time = [best_distance]

    for _ in range(iterations):
        random_order = best_order.copy()
        random.shuffle(random_order)
        current_distance = total_distance(random_order, x_set, y_set)

        if current_distance < best_distance:
            best_distance = current_distance
            best_order = random_order
        
        distance_over_time.append(best_distance)

    return best_order, best_distance, distance_over_time

best_order, best_distance, distance_over_time = rs_tsp(x_set, y_set)


# Plotting the cities and the path
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_set, y_set, color='blue')
plt.plot([x_set[i] for i in best_order] + [x_set[best_order[0]]], 
         [y_set[i] for i in best_order] + [y_set[best_order[0]]], color='red')
plt.title(f"Random Search TSP: Shortest Path Distance = {best_distance:.2f}")
plt.xlabel('x set')
plt.ylabel('y set')

# Plotting the distance over iterations
plt.subplot(1, 2, 2)
plt.plot(distance_over_time)
plt.title("Distance Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Distance")

plt.tight_layout()
plt.show()

# # Create the figure and axis
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.scatter(x_set, y_set, color='blue')
# line, = ax.plot([], [], color='red')  # This is a line object which will be updated
# ax.set_title("Random Search TSP")
# ax.set_xlabel('x set')
# ax.set_ylabel('y set')

# def init():
#     line.set_data([], [])
#     return line,

# def update(frame):
#     global best_order, best_distance  # make sure to update the global variables
#     for _ in range(10):  # 10 random searches per frame for smoother animation
#         random_order = best_order.copy()
#         random.shuffle(random_order)
#         current_distance = total_distance(random_order, x_set, y_set)
#         if current_distance < best_distance:
#             best_distance = current_distance
#             best_order = random_order
#             # Update line data
#             line.set_data([x_set[i] for i in best_order] + [x_set[best_order[0]]], 
#                           [y_set[i] for i in best_order] + [y_set[best_order[0]]])
#             ax.set_title(f"Random Search TSP: Shortest Path Distance = {best_distance:.2f}")
#     return line,

# # Create the animation
# ani = FuncAnimation(fig, update, frames=1000, init_func=init, blit=True)

# plt.show()