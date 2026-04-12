"""
Dijkstra ALgorithm - It assumes that we know the full map 
                     Each move has a cost
                     It always expand the cheapest know path first 
                     Always optimal 
                     BUT - slow for large maps and No heuristic is used

                

"""
#Dijkstra on a grid 

import numpy as np
import heapq
import matplotlib.pyplot as plt

# Create map
GRID_SIZE = 20

grid = np.zeros((GRID_SIZE, GRID_SIZE))

# obstacles
grid[5, 5:15] = 1
grid[10, 2:12] = 1

#Start and goal 
start = (0, 0)
goal = (19, 19)

#Dijksta function 
def dijkstra(grid, start, goal):

    rows, cols = grid.shape

    dist = np.full((rows, cols), np.inf)
    dist[start] = 0

    parent = {}

    pq = []
    heapq.heappush(pq, (0, start))

    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    while pq:

        current_dist, (x, y) = heapq.heappop(pq)

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

    return parent


#Reconstruct path

def get_path(parent, start, goal):

    path = []
    node = goal

    while node != start:
        path.append(node)
        node = parent.get(node, start)

    path.append(start)
    path.reverse()

    return path

#Run Dijkstra
parent = dijkstra(grid, start, goal)
path = get_path(parent, start, goal)

#Visualize
plt.imshow(grid.T, origin='lower', cmap='gray')

px = [p[0] for p in path]
py = [p[1] for p in path]

plt.plot(px, py, 'r-')
plt.scatter(start[0], start[1], c='green')
plt.scatter(goal[0], goal[1], c='blue')

plt.title("Dijkstra Path Planning")
plt.show()


"""
Limitatiosn of Dijkstra's Algorithm:
  -Explores everything equally
  -Does not know the goal direction 

"""