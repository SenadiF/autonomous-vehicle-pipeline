"""
Instead of going to the nearest point on the path , here it goes to the lookahed point and move along that path till it reaches the goal
Now the robot has orientation ,
state =(x,y,theta) 
motion :
   x+= v * cos(theta) 
   y+= v * sin(theta)
   theta+= steering

Step 1 : find the nearest point on the path
Step 2 : Compute the angle to that point
Step 3: Compute steering.

"""
import numpy as np
import matplotlib.pyplot as plt

#  Path 
path_x = np.linspace(0, 20, 200)
path_y = 0.5 * path_x #y = 0.5x

# Robot state 
x, y, theta = 0.0, 5.0, 0.0

#  Parameters 
v = 2.0                 # forward speed
L = 3.0                 # lookahead distance
K = 2.0               # steering gain - How aggressively the robot turns toward the target.
dt = 0.1

trajectory = []
idx = 0
last_idx = 0
# Find lookahead point
def find_lookahead(x, y, path_x, path_y, L, last_idx):
    for i in range(last_idx, len(path_x)):
        px, py = path_x[i], path_y[i]
        dist = np.sqrt((px - x)**2 + (py - y)**2)
        if dist > L:
            return px, py, i
    return path_x[-1], path_y[-1], len(path_x)-1

#  Simulation 
for _ in range(300):

    target_x, target_y, last_idx = find_lookahead(x, y, path_x, path_y, L, last_idx)

    # angle to target
    angle_to_target = np.arctan2(target_y - y, target_x - x)

    # heading error (how much wrong am i facing the worng direction)
    alpha = angle_to_target - theta

    # normalize angle
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
    goal = np.array([path_x[-1], path_y[-1]])
    if np.linalg.norm([x - goal[0], y - goal[1]]) < 0.5:
      break
    # steering control 
    steering = np.clip(K * alpha, -1.0, 1.0)

    # update robot
    theta += steering * dt
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt

    trajectory.append([x, y])

trajectory = np.array(trajectory)

#  Plot 
plt.figure()
plt.plot(path_x, path_y, label="Path")
plt.plot(trajectory[:,0], trajectory[:,1], label="Robot")

plt.scatter(path_x[0], path_y[0])
plt.scatter(path_x[-1], path_y[-1])

plt.legend()
plt.title("Pure Pursuit Path Tracking")
plt.show()
