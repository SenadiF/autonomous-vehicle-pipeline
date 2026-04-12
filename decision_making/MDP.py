"""
Problem Setup:
  Start position :
  Goal : +100
  Obstacle : -100
  Empty cells :(-1 step cost)
  Robot can move in 4 directions (up, down, left, right) 

Value function - Best furture reward from state s
Policy - Best action from each state s

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

gamma = 0.9   # discount factor
theta = 1e-4  # convergence threshold

# VALUE FUNCTION

V = np.zeros((grid_size, grid_size))

# REWARD FUNCTION

def reward(state):
    if state == goal:
        return 100
    elif state == obstacle:
        return -100
    else:
        return -1


# CHECK VALID STATE

def valid(state):
    x, y = state
    return 0 <= x < grid_size and 0 <= y < grid_size


# GET NEXT STATE

def step(state, action):
    dx, dy = actions[action]
    next_state = (state[0] + dx, state[1] + dy)

    if not valid(next_state):
        return state  # stay if wall

    return next_state
# VALUE ITERATION

def value_iteration():
    global V

    while True:
        delta = 0

        for i in range(grid_size):
            for j in range(grid_size):

                state = (i, j)

                if state == goal:
                    continue

                if state == obstacle:
                    continue

                v_old = V[i, j]

                # compute best action value
                action_values = []

                for a in actions:
                    next_state = step(state, a)
                    r = reward(next_state)

                    action_value = r + gamma * V[next_state]
                    action_values.append(action_value)

                V[i, j] = max(action_values)

                delta = max(delta, abs(v_old - V[i, j]))

        if delta < theta:
            break

# POLICY EXTRACTION

policy = np.full((grid_size, grid_size), "", dtype=object)

def extract_policy():
    for i in range(grid_size):
        for j in range(grid_size):

            state = (i, j)

            if state == goal:
                policy[i, j] = "G"
                continue

            if state == obstacle:
                policy[i, j] = "X"
                continue

            best_action = None
            best_value = -1e9

            for a in actions:
                next_state = step(state, a)
                r = reward(next_state)

                value = r + gamma * V[next_state]

                if value > best_value:
                    best_value = value
                    best_action = a

            policy[i, j] = best_action


# RUN MDP 

value_iteration()
extract_policy()


print("Value Function:")
print(np.round(V, 1))

print("\nPolicy:")
for row in policy:
    print(row)


plt.figure()

for i in range(grid_size):
    for j in range(grid_size):

        plt.text(j, grid_size-1-i, policy[i, j],
                 ha='center', va='center')

plt.scatter(goal[1], grid_size-1-goal[0], c='green', s=200, label="Goal")
plt.scatter(obstacle[1], grid_size-1-obstacle[0], c='red', s=200, label="Obstacle")

plt.xlim(-0.5, grid_size-0.5)
plt.ylim(-0.5, grid_size-0.5)
plt.grid()
plt.title("MDP Policy (Value Iteration)")
plt.legend()
plt.show()