"""
Dijkstra - Explores everthing equally, regardless of how close it is to the goal. This can lead to inefficient exploration, especially in large grids with many obstacles.

A*- Explore towards thw goal intelligently

A* adds a heuristic:
   f(n) = g(n) + h(n)
   g(n) = cost so far (same as Dijkstra)
   h(n) = estimated cost to goal
   f(n) = total estimated cost
# for grid, h(n) can be Manhattan distance or Euclidean distance
   h=|x_goal-x| + |y_goal-y|
"""
import numpy as np
import heapq
import matplotlib.pyplot as plt

GRID_SIZE = 20

grid = np.zeros((GRID_SIZE, GRID_SIZE))
grid[5, 5:15] = 1
grid[10, 2:12] = 1

start = (0, 0)
goal = (19, 19)

# Heuristic function (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

#A* algorithm 
def astar(grid, start, goal):

    rows, cols = grid.shape

    g_cost = np.full((rows, cols), np.inf)
    g_cost[start] = 0

    parent = {}

    pq = []
    heapq.heappush(pq, (0, start))

    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    while pq:

        _, (x, y) = heapq.heappop(pq)

        if (x, y) == goal:
            break

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols:

                if grid[nx, ny] == 1:
                    continue

                new_g = g_cost[x, y] + 1

                if new_g < g_cost[nx, ny]:
                    g_cost[nx, ny] = new_g

                    h = heuristic((nx, ny), goal)
                    f = new_g + h

                    parent[(nx, ny)] = (x, y)

                    heapq.heappush(pq, (f, (nx, ny)))

    return parent

# Reconstruct path
def get_path(parent, start, goal):

    path = []
    node = goal

    while node != start:
        path.append(node)
        node = parent.get(node, start)

    path.append(start)
    path.reverse()

    return path

parent = astar(grid, start, goal)
path = get_path(parent, start, goal)

plt.imshow(grid.T, origin='lower', cmap='gray')

px = [p[0] for p in path]
py = [p[1] for p in path]

plt.plot(px, py, 'r-')
plt.scatter(start[0], start[1])
plt.scatter(goal[0], goal[1])

plt.title("A* Path Planning")
plt.show()

"""
A* is used in : ROS navigation stack , autonomous vehicles
"""