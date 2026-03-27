import numpy as np
import matplotlib.pyplot as plt
import random


# MAP

GRID_SIZE = 50

obstacles = np.zeros((GRID_SIZE, GRID_SIZE))
obstacles[20:40, 25] = 1
obstacles[10, 10:40] = 1

start = (5, 5)
goal = (45, 45)

step_size = 2

# tree structure
nodes = [start]
parents = {start: None}



# collision check

def collision(x, y):
    if x < 0 or y < 0 or x >= GRID_SIZE or y >= GRID_SIZE:
        return True
    return obstacles[int(x), int(y)] == 1


# nearest node

def nearest(node_list, point):
    return min(node_list, key=lambda n: np.linalg.norm(np.array(n) - np.array(point)))



# steer

def steer(from_node, to_point):
    vec = np.array(to_point) - np.array(from_node)
    dist = np.linalg.norm(vec)

    if dist == 0:
        return from_node

    vec = vec / dist
    new_point = np.array(from_node) + step_size * vec

    return tuple(new_point)



# RRT MAIN LOOP

for i in range(1000):

    rand_point = (random.randint(0, GRID_SIZE-1),
                  random.randint(0, GRID_SIZE-1))

    nearest_node = nearest(nodes, rand_point)
    new_node = steer(nearest_node, rand_point)

    if not collision(*new_node):
        nodes.append(new_node)
        parents[new_node] = nearest_node

        # check goal reached
        if np.linalg.norm(np.array(new_node) - np.array(goal)) < 3:
            parents[goal] = new_node
            nodes.append(goal)
            break



# PATH RECONSTRUCTION

path = []
node = goal

while node is not None:
    path.append(node)
    node = parents.get(node)

path.reverse()



# VISUALIZATION

plt.imshow(obstacles.T, origin='lower', cmap='gray')

for n in nodes:
    if parents[n] is not None:
        p = parents[n]
        plt.plot([n[0], p[0]], [n[1], p[1]], 'b-', linewidth=0.5)

px, py = zip(*path)
plt.plot(px, py, 'r-', linewidth=3)

plt.scatter(start[0], start[1], c='green', s=100)
plt.scatter(goal[0], goal[1], c='red', s=100)

plt.title("RRT Path Planning")
plt.show()