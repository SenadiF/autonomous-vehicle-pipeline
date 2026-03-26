#Goal- Localization Simulator
# Robot moving in 2D
# noisy motion (realistic uncertainty) and Noisy sensor meauserments(landmarks )
#filter used - Particle filter

#Robot state=[x_position,y_position,orientation]

#2D environment setup with landmarks at known locationsimport numpy as np
import matplotlib.pyplot as plt
import numpy as np


# Environment setup


landmarks = np.array([
    [5, 10],
    [15, 5],
    [20, 15],
    [10, 20]
])

# initial robot state: x, y, theta
true_state = np.array([2.0, 2.0, 0.0])

dt = 1.0
num_steps = 20

# store trajectory
true_path = []


# Motion model

def move(state, control):
    x, y, theta = state
    v, w = control

    theta_new = theta + w * dt
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt

    return np.array([x_new, y_new, theta_new])


# Sensor model (distance to landmarks)


def sense(state, landmarks):
    x, y, _ = state
    measurements = []

    for lx, ly in landmarks:
        dist = np.sqrt((lx - x)**2 + (ly - y)**2)

        # add noise
        noise = np.random.normal(0, 0.5)
        measurements.append(dist + noise)

    return np.array(measurements)


# Simulation loop


for i in range(num_steps):

    # control input (forward motion + slight turn)
    control = [1.0, 0.1]

    # move true robot
    true_state = move(true_state, control)

    # store path
    true_path.append(true_state.copy())

    # sensor readings
    z = sense(true_state, landmarks)

# convert path
true_path = np.array(true_path)


# Plot environment


plt.figure()
plt.scatter(landmarks[:,0], landmarks[:,1], c='red', label="Landmarks")
plt.plot(true_path[:,0], true_path[:,1], label="True Path")
plt.legend()
plt.title("Robot Motion (True State)")
plt.axis("equal")
plt.show()