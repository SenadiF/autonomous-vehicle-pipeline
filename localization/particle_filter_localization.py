import numpy as np
import matplotlib.pyplot as plt


# Environment


landmarks = np.array([
    [5, 10],
    [15, 5],
    [20, 15],
    [10, 20]
])


# True robot state

true_state = np.array([2.0, 2.0, 0.0])


# Particles

N = 200
particles = np.empty((N, 3))  # x, y, theta
particles[:, 0] = np.random.uniform(0, 25, N)
particles[:, 1] = np.random.uniform(0, 25, N)
particles[:, 2] = np.random.uniform(-np.pi, np.pi, N)

weights = np.ones(N) / N

dt = 1.0
num_steps = 30

true_path = []
est_path = []


# Motion model

def move(state, control):
    x, y, theta = state
    v, w = control

    theta += w * dt
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt

    return np.array([x, y, theta])


# Sensor model

def sense(state):
    x, y, _ = state
    return np.sqrt((landmarks[:,0] - x)**2 + (landmarks[:,1] - y)**2)


# Weight update

def compute_weights(particles, measurement):
    w = np.zeros(len(particles))

    for i, p in enumerate(particles):
        pred = sense(p)
        error = np.linalg.norm(pred - measurement)
        w[i] = np.exp(-error**2 / 2.0)  # Gaussian likelihood

    w += 1e-300  # avoid zeros
    return w / np.sum(w)


# Resampling

def resample(particles, weights):
    idx = np.random.choice(len(particles), size=len(particles), p=weights)
    particles = particles[idx]
    weights = np.ones(len(particles)) / len(particles)
    return particles, weights


# Simulation loop

for t in range(num_steps):

    # control input
    control = np.array([1.0, 0.2])

    # move true robot
    true_state = move(true_state, control)

    # sensor measurement
    measurement = sense(true_state)

  
    #  Move particles
    
    for i in range(N):
        noise = np.random.normal(0, [0.2, 0.2, 0.05])
        particles[i] = move(particles[i], control) + noise

    
    #  Update weights
   
    weights = compute_weights(particles, measurement)

   
    #  Resample
   
    particles, weights = resample(particles, weights)

    # estimate = mean of particles
    est = np.mean(particles, axis=0)

    true_path.append(true_state.copy())
    est_path.append(est)


# Plot results

true_path = np.array(true_path)
est_path = np.array(est_path)

plt.figure()

# landmarks
plt.scatter(landmarks[:,0], landmarks[:,1], c='red', label='Landmarks')

# particles (final frame)
plt.scatter(particles[:,0], particles[:,1], s=5, alpha=0.3, label='Particles')

# paths
plt.plot(true_path[:,0], true_path[:,1], label='True Path')
plt.plot(est_path[:,0], est_path[:,1], label='Estimated Path')

plt.legend()
plt.title("Particle Filter Localization")
plt.axis("equal")
plt.show()