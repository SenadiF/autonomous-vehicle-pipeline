"""
FastSLAM Overview

Particle Filter=Robot Pose
EKF per landmark= MAP (each landmark is estimated using a gaussian)

Particle =robot pose [x, y, θ] + {landmark_id: EKF(mean, covariance)}



"""
import matplotlib.pyplot as plt

# store true trajectory for visualization
true_path = []
#Setup world

import numpy as np

true_landmarks = np.array([
    [5, 10],
    [15, 5],
    [20, 15],
    [10, 20]
])

true_state = np.array([2.0, 2.0, 0.0])

#Particle structure 

N = 100

particles = []
for _ in range(N):
    particles.append({
        "pose": np.array([
            np.random.uniform(0, 25),
            np.random.uniform(0, 25),
            np.random.uniform(-np.pi, np.pi)
        ]),
        "landmarks": {},  # EKF per landmark
        "weight": 1.0
    })

#Motion model 
def move(x, u):
    x, y, th = x
    v, w = u

    th += w
    x += v * np.cos(th)
    y += v * np.sin(th)

    return np.array([x, y, th])

#Sensor model
def sense(state):
    x, y, _ = state
    return np.linalg.norm(true_landmarks - np.array([x, y]), axis=1)

#EKF landmark update

#Intialization
def init_landmark(x, z):
    lx = x[0] + z * np.cos(0)
    ly = x[1] + z * np.sin(0)
    return {
        "mu": np.array([lx, ly]),
        "sigma": np.eye(2) * 5.0
    }

#Update landmarks 
def update_landmark(lm, x, z):
    mu = lm["mu"]
    sigma = lm["sigma"]

    # expected measurement
    dx = mu[0] - x[0]
    dy = mu[1] - x[1]
    expected = np.sqrt(dx**2 + dy**2)

    # Jacobian
    H = np.array([[dx/expected, dy/expected]])

    R = np.array([[0.5]])

    K = sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + R)

    z_pred = expected
    innovation = z - z_pred

    mu = mu + (K.flatten() * innovation)
    sigma = (np.eye(2) - K @ H) @ sigma

    lm["mu"] = mu
    lm["sigma"] = sigma

#FastSlam loop

def compute_weight(particle, measurement):
    x = particle["pose"]
    w = 1.0

    for i, z in enumerate(measurement):

        if i not in particle["landmarks"]:
            particle["landmarks"][i] = init_landmark(x,z)

        lm = particle["landmarks"][i]

        # EKF update
        update_landmark(lm, x, z)

        # likelihood (simple Gaussian)
        dx = lm["mu"][0] - x[0]
        dy = lm["mu"][1] - x[1]
        pred = np.sqrt(dx**2 + dy**2)

        w *= np.exp(- (z - pred)**2)

    return w


# main loop 
for t in range(30):

    u = np.array([1.0, 0.2])

    # true motion
    true_state = move(true_state, u)

    z = sense(true_state)

    # particle update
    for p in particles:

        noise = np.random.normal(0, [0.2, 0.2, 0.05])
        p["pose"] = move(p["pose"], u) + noise

        p["weight"] = compute_weight(p, z)

    # normalize weights
    w = np.array([p["weight"] for p in particles])
    w += 1e-9
    w /= np.sum(w)

    # resample
    idx = np.random.choice(N, N, p=w)
    particles = [particles[i] for i in idx]
    true_path.append(true_state.copy())

plt.figure(figsize=(8, 8))


# True landmarks

plt.scatter(
    true_landmarks[:, 0],
    true_landmarks[:, 1],
    c='red',
    marker='x',
    s=100,
    label='True Landmarks'
)


# Robot true path

true_path = np.array(true_path)
plt.plot(
    true_path[:, 0],
    true_path[:, 1],
    'b-',
    label='True Path'
)


# Estimated landmarks (from particles)

for p in particles[:10]:  
    for lm in p["landmarks"].values():
        mu = lm["mu"]
        plt.scatter(mu[0], mu[1], c='green', alpha=0.3)


# Particle positions

particle_positions = np.array([p["pose"] for p in particles])
plt.scatter(
    particle_positions[:, 0],
    particle_positions[:, 1],
    c='gray',
    s=10,
    alpha=0.3,
    label='Particles'
)

plt.title("FastSLAM Visualization (Particle + EKF Landmark Map)")
plt.legend()
plt.axis("equal")
plt.grid()
plt.show()