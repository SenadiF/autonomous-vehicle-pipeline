"""
Q-learnning /DQN - leans how good an action is but unstable and slow in big problems 

Policy gradient - learns probability of actions but since learning is noisy results wiht high variance 

Actor critic - Combines both, the actor learns the policy and the critic learns the value function to reduce variance and stabilize learning
A(s,a)=Q(s,a)−V(s)
if A>0 - action was better than expected - increase probability
if A<0 - worse than expected - decrease probability
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
#grid world setup
def render_grid(pos):
    grid = [" . "] * 5
    grid[pos] = " A "   # agent
    grid[4] = " G "     # goal

    print("".join(grid))
class GridWorld:
    def __init__(self):
        self.goal = 4
        self.reset()
    #Every episode starts at 0 and the agent can move left or right until it reaches the goal at 4
    def reset(self):
        self.pos = 0
        return self.pos

    def step(self, action):
        if action == 0:  # left
            self.pos = max(0, self.pos - 1)
        else:  # right
            self.pos = min(4, self.pos + 1)

        reward = 1 if self.pos == self.goal else 0
        done = self.pos == self.goal

        return self.pos, reward, done



# Actor-Critic Network

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(1, 32)

        self.actor = nn.Linear(32, 2)   # left/right
        self.critic = nn.Linear(32, 1)  # value
    
    def forward(self, state):
        x = torch.relu(self.shared(state))
        policy = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value



# Training setup

env = GridWorld()
model = ActorCritic()

optimizer = optim.Adam(model.parameters(), lr=0.01)

gamma = 0.99


# Training loop

for episode in range(200):  # fewer episodes for demo

    state = env.reset()

    log_probs = []
    values = []
    rewards = []

    done = False

    print(f"\nEpisode {episode}")

    while not done:

        state_t = torch.tensor([[state]], dtype=torch.float32)

        policy, value = model(state_t)

        action = torch.multinomial(policy, 1).item()

        log_prob = torch.log(policy[0, action])

        next_state, reward, done = env.step(action)

        #visualization
        render_grid(state)

        time.sleep(0.2)  

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)

        state = next_state

    

    print("Episode reward:", sum(rewards))

# Compute returns and advantages
    returns = []
    G = 0

    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns)
    values = torch.cat(values).squeeze()

    
    advantage = returns - values

    
    actor_loss = -(torch.stack(log_probs) * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    loss = actor_loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 50 == 0:
        print(f"Episode {episode}, total reward: {sum(rewards)}")