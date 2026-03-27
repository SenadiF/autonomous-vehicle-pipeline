"""
Why D*- 
A* & Dijkstras algorithm assumes the map is fixed 
If an obstacle appears- A* and Dijkstra will not be able to find a path
D* can replan in real time when the environment changes

D* - Reuse previous path + updaed onlt effected parts 
D* Goal> Start (Goal is fixed, local area changes)
"""
# Simplified D* algorithm - A* re run (before and new obstacles)
import numpy as np
import matplotlib.pyplot as plt
import heapq
import time


# GRID SETUP

GRID_SIZE = 30
grid = np.zeros((GRID_SIZE, GRID_SIZE))

# Start and goal
start = (2, 2)
goal = (25, 25)

# Initial obstacles
grid[10:20, 12] = 1
grid[15, 5:15] = 1



# HEURISTIC

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])



# NEIGHBORS

def neighbors(node):
    x, y = node
    moves = [(1,0), (-1,0), (0,1), (0,-1)]
    result = []
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            result.append((nx, ny))
    return result



# D* (simplified as replanning A*)

def dstar_like(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            break

        for n in neighbors(current):
            if grid[n] == 1:
                continue

            tentative_g = g[current] + 1

            if n not in g or tentative_g < g[n]:
                g[n] = tentative_g
                f = tentative_g + heuristic(n, goal)
                heapq.heappush(open_set, (f, n))
                came_from[n] = current

    # reconstruct path
    path = []
    node = goal
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()

    return path

def draw(grid, path, title):
    plt.imshow(grid.T, origin='lower', cmap='gray')

    if path:
        px, py = zip(*path)
        plt.plot(px, py, 'r-', linewidth=2)

    plt.scatter(start[0], start[1], c='green', s=100)
    plt.scatter(goal[0], goal[1], c='blue', s=100)

    plt.title(title)
    plt.legend()

    plt.draw()
    plt.pause(0.5)
    
    


# INITIAL PLAN
#
plt.figure(figsize=(6, 6))
plt.ion()  

path1 = dstar_like(grid, start, goal)
draw(grid, path1, "Initial Path (Before obstacle)")
plt.pause(5) 



# NEW OBSTACLE

grid[2:18, 18] = 1   # dynamic obstacle blocking path
plt.pause(1)   
path2 = dstar_like(grid, start, goal)

draw(grid, path2, "After Obstacle Change (D* Replanning)")
plt.show()
plt.pause(5)