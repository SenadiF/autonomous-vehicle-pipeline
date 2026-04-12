"""
Q- learning - learn Q(s,a) by interacting with the environment and updating estimates based on observed rewards.

Q(s,a)= How good is taking action a in state s 

Q(s,a)←Q(s,a)+α[r+γmaxa′​Q(s′,a′)−Q(s,a)]
alpha: learning rate (0<α≤1)
gamma: discount factor (0≤γ<1)

"""
import numpy as np
import matplotlib.pyplot as plt
import random

grid_size = 5

start = (0, 0)
goal = (4, 4)
obstacle = (2, 2)

actions = ["U", "D", "L", "R"]

action_map = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1)
}
#Parameters
alpha = 0.1    # learning rate
gamma = 0.9      # discount
epsilon = 0.2    # exploration

episodes = 500


# Q TABLE

Q = {}

for i in range(grid_size):
    for j in range(grid_size):
        Q[(i, j)] = {a: 0.0 for a in actions}

#Check if state is valid (within grid)
def valid(state):
    x, y = state
    return 0 <= x < grid_size and 0 <= y < grid_size

def step(state, action):
    dx, dy = action_map[action]
    ns = (state[0] + dx, state[1] + dy)

    if not valid(ns):
        return state

    return ns
#Reward function
def reward(state):
    if state == goal:
        return 100
    elif state == obstacle:
        return -100
    else:
        return -1


# EPSILON-GREEDY

def choose_action(state):
    if random.random() < epsilon:
        return random.choice(actions)  # explore
    else:
        return max(Q[state], key=Q[state].get)  # exploit


# TRAINING
for ep in range(episodes):

    state = start

    for step_count in range(100):

        action = choose_action(state)
        next_state = step(state, action)
        r = reward(next_state)

        # Q-learning update
        best_next = max(Q[next_state].values())

        Q[state][action] += alpha * (
            r + gamma * best_next - Q[state][action]
        )

        state = next_state

        if state == goal:
            break

# Extract policy
policy = {}

for state in Q:
    if state == goal:
        policy[state] = "G"
    elif state == obstacle:
        policy[state] = "X"
    else:
        policy[state] = max(Q[state], key=Q[state].get)

print("\nLearned Policy:")
for i in range(grid_size):
    print([policy[(i, j)] for j in range(grid_size)])


state = start
trajectory = [state]

for _ in range(50):

    if state == goal:
        break

    action = policy[state]
    state = step(state, action)
    trajectory.append(state)


traj_x = [s[1] for s in trajectory]
traj_y = [grid_size - 1 - s[0] for s in trajectory]

plt.figure()

# grid
for i in range(grid_size):
    for j in range(grid_size):
        plt.scatter(j, grid_size - 1 - i, c='lightgray')


plt.scatter(goal[1], grid_size - 1 - goal[0], s=200, label="Goal")
plt.scatter(obstacle[1], grid_size - 1 - obstacle[0], s=200, label="Obstacle")
plt.scatter(start[1], grid_size - 1 - start[0], s=100, label="Start")

# path
plt.plot(traj_x, traj_y, marker='o', label="Learned Path")

plt.title("Q-Learning ")
plt.legend()
plt.grid()
plt.show()