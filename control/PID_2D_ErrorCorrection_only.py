#PID contril - Implemeted error correction , so in the simulation it will correct and gte fixed to the target position it does ot move along as still oreinttaion and steering is not implemented .

import numpy as np
import matplotlib.pyplot as plt

# Path
path_x = np.linspace(0, 20, 100)
path_y = 0.5 * path_x

robot = np.array([0.0, 5.0])

Kp = 1.0
Ki = 0.1
Kd = 0.5

dt = 0.1

integral = np.array([0.0, 0.0])
prev_error = np.array([0.0, 0.0])

trajectory = []

def closest_point(robot, path_x, path_y):
    distances = (path_x - robot[0])**2 + (path_y - robot[1])**2
    idx = np.argmin(distances)
    return np.array([path_x[idx], path_y[idx]])

for _ in range(200):

    target = closest_point(robot, path_x, path_y)

    error = target - robot

    integral += error * dt
    derivative = (error - prev_error) / dt

    control = Kp * error + Ki * integral + Kd * derivative

    robot += control * dt

    trajectory.append(robot.copy())
    prev_error = error

trajectory = np.array(trajectory)

plt.figure()
plt.plot(path_x, path_y, label="Path")
plt.plot(trajectory[:,0], trajectory[:,1], label="Robot")
plt.legend()
plt.title(" PID Path Following")
plt.show()