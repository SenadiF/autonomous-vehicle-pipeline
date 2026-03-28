import numpy as np
import matplotlib.pyplot as plt


# Path 

path_x = np.linspace(0, 20, 200)
path_y = 0.5 * path_x


# Robot state

x, y, theta = 0.0, 5.0, 0.0

v = 1.0
L = 2.0  # lookahead
dt = 0.1

# obstacle
obstacle = np.array([6.0, 3.0])
obstacle_radius = 1.5

# FSM state
state = "MOVE_TO_PATH"

trajectory = []
last_idx = 0

def dist(a, b):
    return np.linalg.norm(a - b)

def find_lookahead(x, y, path_x, path_y, L, start_idx):
    for i in range(start_idx, len(path_x)):
        d = np.sqrt((path_x[i] - x)**2 + (path_y[i] - y)**2)
        if d > L:
            return path_x[i], path_y[i], i
    return path_x[-1], path_y[-1], len(path_x)-1

def obstacle_near(x, y):
    return dist(np.array([x, y]), obstacle) < obstacle_radius


# simulation loop

for _ in range(300):

    pos = np.array([x, y])

    
    # FSM (DECISION MAKING)
    
    if state == "MOVE_TO_PATH":

        if obstacle_near(x, y):
            state = "AVOID_OBSTACLE"

    elif state == "AVOID_OBSTACLE":

        # simple rule: once safe, go back
        if not obstacle_near(x, y):
            state = "MOVE_TO_PATH"

    elif state == "STOP":
        pass

   
    # CONTROL (PURE PURSUIT)
    

    if state == "MOVE_TO_PATH":

        target_x, target_y, last_idx = find_lookahead(
            x, y, path_x, path_y, L, last_idx
        )

        angle_to_target = np.arctan2(target_y - y, target_x - x)
        alpha = angle_to_target - theta
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        K = 2.0
        steering = np.clip(K * alpha, -1.0, 1.0)

        theta += steering * dt
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt

    elif state == "AVOID_OBSTACLE":

        # avoidance motion -side push
        dx = x - obstacle[0]
        dy = y - obstacle[1]

        avoid_dir = np.array([-dy, dx])
        avoid_dir = avoid_dir / (np.linalg.norm(avoid_dir) + 1e-6)

        x += avoid_dir[0] * v * dt
        y += avoid_dir[1] * v * dt

    trajectory.append([x, y])

#Visualization 
trajectory = np.array(trajectory)

plt.figure()
plt.plot(path_x, path_y, label="Path")
plt.plot(trajectory[:,0], trajectory[:,1], label="Robot")
plt.scatter(obstacle[0], obstacle[1], label="Obstacle")

plt.legend()
plt.axis("equal")
plt.title("FSM + Pure Pursuit Integration")
plt.show()