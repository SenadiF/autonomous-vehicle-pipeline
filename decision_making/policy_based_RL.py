"""
Policy -based RL - Directly learn what action to take (No Q table)
What the neural network learns is :
   Input -> state (x,y)
   Output -> probablities for each action (U,D,L,R)
   If it was a good action, increase the probability of taking that action in that state, if it was bad, decrease it.

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Environment setup

grid_size = 5
start = (0, 0)
goal = (4, 4)

actions = [( -1,0),(1,0),(0,-1),(0,1)]  # U D L R

def valid(s):
    return 0 <= s[0] < grid_size and 0 <= s[1] < grid_size

def step(state, action):
    dx, dy = action
    ns = (state[0]+dx, state[1]+dy)
    return ns if valid(ns) else state

def reward(state):
    return 10 if state == goal else -1


# POLICY NETWORK

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# VISUALIZATION

def draw_policy():
    plt.clf()
    for i in range(grid_size):
        for j in range(grid_size):
            s = torch.tensor([i,j], dtype=torch.float32)
            probs = policy(s).detach().numpy()
            best = np.argmax(probs)

            dx, dy = actions[best]

            plt.arrow(j, i, dy*0.3, -dx*0.3,
                      head_width=0.2)

    plt.scatter(goal[1], goal[0])
    plt.xlim(-1, grid_size)
    plt.ylim(-1, grid_size)
    plt.title("Policy")
    plt.pause(0.1)

# TRAINING
plt.figure()

for episode in range(200):

    state = start
    log_probs = []
    rewards = []

    for _ in range(30):

        s = torch.tensor(state, dtype=torch.float32)
        probs = policy(s)

        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()

        log_probs.append(dist.log_prob(action_idx))

        state = step(state, actions[action_idx])
        r = reward(state)
        rewards.append(r)

        if state == goal:
            break

    # total reward
    G = sum(rewards)

    # update policy
    loss = 0
    for log_p in log_probs:
        loss -= log_p * G

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # visualize every few episodes
    if episode % 10 == 0:
        draw_policy()

plt.show()