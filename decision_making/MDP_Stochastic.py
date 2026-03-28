import numpy as np
import matplotlib.pyplot as plt

# GRID SETUP
grid_size = 5

start = (0, 0)
goal = (4, 4)
obstacle = (2, 2)

actions = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1)
}

gamma = 0.9
theta = 1e-4


# REWARD FUNCTION
def reward(state):
    if state == goal:
        return 100
    elif state == obstacle:
        return -100
    else:
        return -1


#Check if state is valid (within grid)
def valid(state):
    x, y = state
    return 0 <= x < grid_size and 0 <= y < grid_size


# Transition (deterministic)
def move(state, action):
    dx, dy = actions[action]
    ns = (state[0] + dx, state[1] + dy)
    return ns if valid(ns) else state

# VALUE FUNCTION
V = np.zeros((grid_size, grid_size))

#expected value for stochastic transitions
def expected_value(state, action):

    total = 0.0

    # 80% intended move
    ns = move(state, action)
    total += 0.8 * (reward(ns) + gamma * V[ns])

    # 10% slip up
    ns = move(state, "U")
    total += 0.1 * (reward(ns) + gamma * V[ns])

    # 10% slip down
    ns = move(state, "D")
    total += 0.1 * (reward(ns) + gamma * V[ns])

    return total

#Value iteration algorithm
def value_iteration():
    global V

    while True:
        delta = 0

        for i in range(grid_size):
            for j in range(grid_size):

                state = (i, j)

                if state in [goal, obstacle]:
                    continue

                v_old = V[i, j]

                V[i, j] = max(
                    expected_value(state, a)
                    for a in actions
                )

                delta = max(delta, abs(v_old - V[i, j]))

        if delta < theta:
            break

# Extract policy from value function
def extract_policy():
    policy = {}

    for i in range(grid_size):
        for j in range(grid_size):

            state = (i, j)

            if state == goal:
                policy[state] = "G"
                continue

            if state == obstacle:
                policy[state] = "X"
                continue

            best_a = None
            best_v = -1e9

            for a in actions:
                val = expected_value(state, a)

                if val > best_v:
                    best_v = val
                    best_a = a

            policy[state] = best_a

    return policy

value_iteration()
policy = extract_policy()

print("\nValue Function:")
print(np.round(V, 1))

print("\nPolicy:")
for i in range(grid_size):
    print([policy[(i, j)] for j in range(grid_size)])


state = start
trajectory = [state]

def stochastic_step(state, action):
    """Simulate stochastic behavior"""
    import random
    r = random.random()

    if r < 0.8:
        a = action
    elif r < 0.9:
        a = "U"
    else:
        a = "D"

    return move(state, a)

for _ in range(50):

    if state == goal:
        break

    action = policy[state]
    state = stochastic_step(state, action)
    trajectory.append(state)

#Visualization
traj_x = [s[1] for s in trajectory]
traj_y = [grid_size - 1 - s[0] for s in trajectory]

plt.figure()

for i in range(grid_size):
    for j in range(grid_size):
        plt.scatter(j, grid_size - 1 - i, c='lightgray')

plt.scatter(goal[1], grid_size - 1 - goal[0], c='green', s=200, label="Goal")
plt.scatter(obstacle[1], grid_size - 1 - obstacle[0], c='red', s=200, label="Obstacle")
plt.scatter(start[1], grid_size - 1 - start[0], c='blue', s=100, label="Start")

plt.plot(traj_x, traj_y, marker='o', label="Agent Path")

plt.title(" Stochastic MDP ")
plt.legend()
plt.grid()
plt.show()