"""
Q leaning - 
   Q table store Q(s,a) but this works only for small states not scalable 
   but 
Deep Q learning - ueses a neural network to approximate Q(s,a) and can handle large state spaces

"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt


# GRID

grid_size = 5
start = (0, 0)
goal = (4, 4)
obstacle = (2, 2)

actions = ["U", "D", "L", "R"]

action_map = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}


# PARAMETERS

gamma = 0.9
epsilon = 0.3
alpha = 0.001
episodes = 500


# NETWORK

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.net(x)

model = DQN()
optimizer = optim.Adam(model.parameters(), lr=alpha)
loss_fn = nn.MSELoss()


# ENVIRONMENT

def valid(state):
    x, y = state
    return 0 <= x < grid_size and 0 <= y < grid_size

def step(state, action):
    dx, dy = action_map[action]
    ns = (state[0] + dx, state[1] + dy)
    return ns if valid(ns) else state

def reward(state):
    if state == goal:
        return 100
    elif state == obstacle:
        return -100
    else:
        return -1


# TRAINING

for ep in range(episodes):

    state = start

    for _ in range(100):

        s = torch.tensor(state, dtype=torch.float32)

        # epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_vals = model(s)
                action = torch.argmax(q_vals).item()

        next_state = step(state, action)
        r = reward(next_state)

        s_next = torch.tensor(next_state, dtype=torch.float32)

        # target
        with torch.no_grad():
            target = r + gamma * torch.max(model(s_next))

        q_vals = model(s)
        loss = loss_fn(q_vals[action], target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

        if state == goal:
            break


# SIMULATION
state = start
trajectory = [state]

for _ in range(50):

    s = torch.tensor(state, dtype=torch.float32)
    action = torch.argmax(model(s)).item()

    state = step(state, action)
    trajectory.append(state)

    if state == goal:
        break


traj_x = [s[1] for s in trajectory]
traj_y = [grid_size - 1 - s[0] for s in trajectory]

plt.figure()

for i in range(grid_size):
    for j in range(grid_size):
        plt.scatter(j, grid_size - 1 - i)

plt.scatter(goal[1], grid_size - 1 - goal[0], s=200, label="Goal")
plt.scatter(obstacle[1], grid_size - 1 - obstacle[0], s=200, label="Obstacle")
plt.scatter(start[1], grid_size - 1 - start[0], s=100, label="Start")

plt.plot(traj_x, traj_y, marker='o', label="DQN Path")

plt.legend()
plt.title("Deep Q-Learning")
plt.show()


"""
Improvements :
 add a replay buffer to store past experiences and sample from it during training
 add a target network to stabilize training
"""