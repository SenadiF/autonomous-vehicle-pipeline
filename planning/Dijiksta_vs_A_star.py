import numpy as np
import heapq
import matplotlib.pyplot as plt
import time

GRID_SIZE = 50

grid = np.zeros((GRID_SIZE, GRID_SIZE))

# Hard obstacles
grid[10:40, 20] = 1
grid[10, 20:45] = 1
grid[40, 5:30] = 1
grid[25:45, 35] = 1

start = (0, 0)
goal = (49, 49)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


#Dijkstra 
def dijkstra(grid, start, goal):

    rows, cols = grid.shape
    dist = np.full((rows, cols), np.inf)
    dist[start] = 0

    parent = {}
    visited = np.zeros_like(grid)

    pq = []
    heapq.heappush(pq, (0, start))

    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    while pq:
        current_dist, (x, y) = heapq.heappop(pq)

        visited[x, y] += 1

        if (x, y) == goal:
            break

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx, ny] == 1:
                    continue

                new_dist = current_dist + 1

                if new_dist < dist[nx, ny]:
                    dist[nx, ny] = new_dist
                    parent[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (new_dist, (nx, ny)))

    return parent, visited


#  A* 
def astar(grid, start, goal):

    rows, cols = grid.shape
    g_cost = np.full((rows, cols), np.inf)
    g_cost[start] = 0

    parent = {}
    visited = np.zeros_like(grid)

    pq = []
    heapq.heappush(pq, (0, start))

    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    while pq:
        _, (x, y) = heapq.heappop(pq)

        visited[x, y] += 1

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
                    f = new_g + heuristic((nx, ny), goal)

                    parent[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (f, (nx, ny)))

    return parent, visited


#  Path 
def get_path(parent, start, goal):
    path = []
    node = goal

    while node != start:
        path.append(node)
        node = parent.get(node, start)

    path.append(start)
    path.reverse()
    return path


# Execute Dijkstra 
start_time = time.time()
parent_dij, visited_dij = dijkstra(grid, start, goal)
time_dij = time.time() - start_time
path_dij = get_path(parent_dij, start, goal)

# Execute A*
start_time = time.time()
parent_astar, visited_astar = astar(grid, start, goal)
time_astar = time.time() - start_time
path_astar = get_path(parent_astar, start, goal)


print("Dijkstra:")
print("Time:", time_dij)
print("Visited nodes:", int(np.sum(visited_dij > 0)))

print("\nA*:")
print("Time:", time_astar)
print("Visited nodes:", int(np.sum(visited_astar > 0)))




fig, axs = plt.subplots(1, 3, figsize=(18,6))

# --- Dijkstra exploration ---
axs[0].imshow(grid.T, origin='lower', cmap='gray')
axs[0].imshow(visited_dij.T, origin='lower', cmap='Reds', alpha=0.6)

axs[0].plot([path[0] for path in path_dij], [path[1] for path in path_dij])
axs[0].set_title("Dijkstra Exploration")


# --- A* exploration ---
axs[1].imshow(grid.T, origin='lower', cmap='gray')
axs[1].imshow(visited_astar.T, origin='lower', cmap='Blues', alpha=0.6)

axs[1].plot([path[0] for path in path_astar], [path[1] for path in path_astar])
axs[1].set_title("A* Exploration")

#  Combined paths 
axs[2].imshow(grid.T, origin='lower', cmap='gray')

axs[2].plot([path[0] for path in path_dij], [path[1] for path in path_dij], label="Dijkstra")
axs[2].plot([path[0] for path in path_astar], [path[1] for path in path_astar], linestyle='--', label="A*")

axs[2].scatter(start[0], start[1])
axs[2].scatter(goal[0], goal[1])

axs[2].legend()
axs[2].set_title("Path Comparison")

plt.show()