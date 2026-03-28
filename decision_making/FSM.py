"""
Robot behavior :
  - Move towards a goal
  -Gets near obstacle 
  -Switches to obstacle avoidance
  -Comes back to goal when safe
  - Stops when it reaches the goal

"""

import numpy as np
import matplotlib.pyplot as plt

# Environment setup

goal = np.array([10.0, 10.0])
obstacle = np.array([5.0, 5.0])

x, y = 0.0, 0.0
v = 0.2  # speed

state = "MOVE_TO_GOAL"

trajectory = []

def distance(a, b):
    return np.linalg.norm(a - b)

def is_obstacle_near(pos, obstacle, threshold=1.5):
    return distance(pos, obstacle) < threshold

def move_towards(current, target):
    direction = target - current
    norm = np.linalg.norm(direction)

    if norm == 0:
        return current

    return current + (direction / norm) * v


# Simulation

for _ in range(200):

    pos = np.array([x, y])

    #  FSM 
    if state == "MOVE_TO_GOAL":

        if is_obstacle_near(pos, obstacle):
            state = "AVOID_OBSTACLE"

        elif distance(pos, goal) < 0.5:
            state = "STOP"

        else:
            pos = move_towards(pos, goal)

    elif state == "AVOID_OBSTACLE":

        # avoidance: move sideways
        avoid_dir = np.array([-(y - obstacle[1]), (x - obstacle[0])])
        avoid_dir = avoid_dir / (np.linalg.norm(avoid_dir) + 1e-6)

        pos = pos + avoid_dir * v

        # if safe again → go back
        if not is_obstacle_near(pos, obstacle):
            state = "MOVE_TO_GOAL"

    elif state == "STOP":
        pass

    # update robot position
    x, y = pos
    trajectory.append([x, y])


#  results
trajectory = np.array(trajectory)

plt.figure()
plt.plot(trajectory[:,0], trajectory[:,1], label="Robot Path")
plt.scatter(goal[0], goal[1], label="Goal")
plt.scatter(obstacle[0], obstacle[1], label="Obstacle")

plt.title("FSM Robot Simulation")
plt.legend()
plt.axis("equal")
plt.show()