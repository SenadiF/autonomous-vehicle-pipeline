"""
Policy -> Robot exexutes action -> Motion simulation
"""
import numpy as np
import matplotlib.pyplot as plt

# GRID SETUP

grid_size = 5

goal = (4, 4)
obstacle = (2, 2)

actions = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1)
}

gamma = 0.9


# VALUE FUNCTION

V = np.zeros((grid_size, grid_size))

def reward(state):
    if state == goal:
        return 100
    elif state == obstacle:
        return -100
    else:
        return -1

def valid(state):
    x, y = state
    return 0 <= x < grid_size and 0 <= y < grid_size

def step(state, action):
    dx, dy = actions[action]
    next_state = (state[0] + dx, state[1] + dy)

    if not valid(next_state):
        return state

    return next_state

# VALUE ITERATION

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

                values = []

                for a in actions:
                    ns = step(state, a)
                    r = reward(ns)
                    values.append(r + gamma * V[ns])

                V[i, j] = max(values)

                delta = max(delta, abs(v_old - V[i, j]))

        if delta < 1e-4:
            break


# POLICY EXTRACTION

policy = {}

def extract_policy():
    for i in range(grid_size):
        for j in range(grid_size):

            state = (i, j)

            if state in [goal, obstacle]:
                policy[state] = None
                continue

            best_a = None
            best_v = -1e9

            for a in actions:
                ns = step(state, a)
                r = reward(ns)
                val = r + gamma * V[ns]

                if val > best_v:
                    best_v = val
                    best_a = a

            policy[state] = best_a

# RUN MDP SOLVER

value_iteration()
extract_policy()


# AGENT SIMULATION

state = (0, 0)
trajectory = [state]

for _ in range(50):

    if state == goal:
        break

    action = policy[state]

    if action is None:
        break

    state = step(state, action)
    trajectory.append(state)


# VISUALIZATION

traj_x = [s[1] for s in trajectory]
traj_y = [grid_size - 1 - s[0] for s in trajectory]

plt.figure()

# grid
for i in range(grid_size):
    for j in range(grid_size):
        plt.scatter(j, grid_size-1-i, c='lightgray')

# obstacles + goal
plt.scatter(goal[1], grid_size-1-goal[0], c='green', s=200, label="Goal")
plt.scatter(obstacle[1], grid_size-1-obstacle[0], c='red', s=200, label="Obstacle")

# path
plt.plot(traj_x, traj_y, marker='o', label="MDP Agent Path")

plt.title("MDP Agent Execution of Learned Policy")
plt.legend()
plt.grid()
plt.show()