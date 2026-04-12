# Localization - Landmarks were known  , we only estimated the robot pose

# Now - landmarks are unknown , robot must estimate its pose , and gradually build pose 

#Unknown world
import numpy as np
import matplotlib.pyplot as plt

# TRUE WORLD (hidden map)

true_landmarks = np.array([
    [5, 10],
    [15, 5],
    [20, 15],
    [10, 20]
])

# robot true state
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
        "landmarks": {}  # unknown map per particle
    })

weights = np.ones(N) / N

#Motion model 
def move(state, control):
    x, y, theta = state
    v, w = control

    theta += w
    x += v * np.cos(theta)
    y += v * np.sin(theta)

    return np.array([x, y, theta])
#Sensor model 
def sense(state):
    x, y, _ = state
    return np.linalg.norm(true_landmarks - np.array([x, y]), axis=1)

#We estimate landmarks inside each particle 
#If robot sees a measurement - and the landmark already exists ,
# refine it else intilaize it 

def update_map(particle, measurement):
    x, y, _ = particle["pose"]

    for i, z in enumerate(measurement):
        lm = particle["landmarks"]

        # if landmark i not seen before
        if i not in lm:
            angle = np.random.uniform(-np.pi, np.pi)
            lm[i] = np.array([
                x + z * np.cos(angle),
                y + z * np.sin(angle)
            ])
        else:
            # refine estimate (simple averaging)
            lm[i] = 0.9 * lm[i] + 0.1 * np.array([
                x + z * np.cos(0),
                y + z * np.sin(0)
            ])
#Weight update 
def compute_weight(particle, measurement):
    x, y, _ = particle["pose"]

    predicted = []
    for lm in particle["landmarks"].values():
        dist = np.linalg.norm(lm - np.array([x, y]))
        predicted.append(dist)

    if len(predicted) == 0:
        return 1e-3

    error = np.linalg.norm(np.array(predicted[:len(measurement)]) - measurement)
    return np.exp(-error**2)

#Resampling 
def resample(particles, weights):
    idx = np.random.choice(len(particles), size=len(particles), p=weights)
    new_particles = [particles[i] for i in idx]
    new_weights = np.ones(len(particles)) / len(particles)
    return new_particles, new_weights

#Simulation 
true_path = []

for t in range(30):

    control = np.array([1.0, 0.2])

    # move true robot
    true_state = move(true_state, control)
    true_path.append(true_state.copy())

    # measurement
    measurement = sense(true_state)

    # particle update
    for p in particles:
        # motion noise
        noise = np.random.normal(0, [0.2, 0.2, 0.05])
        p["pose"] = move(p["pose"], control) + noise

        # map update
        update_map(p, measurement)

    # weights
    weights = np.array([compute_weight(p, measurement) for p in particles])
    weights += 1e-9
    weights /= np.sum(weights)

    # resample
    particles, weights = resample(particles, weights)

#Viualize 
plt.figure()

# true landmarks
plt.scatter(true_landmarks[:,0], true_landmarks[:,1], c='red', label='True Landmarks')

# particles + their map (only first few shown)
for p in particles[:10]:
    lm = np.array(list(p["landmarks"].values()))
    if len(lm) > 0:
        plt.scatter(lm[:,0], lm[:,1], alpha=0.3)

true_path = np.array(true_path)
plt.plot(true_path[:,0], true_path[:,1], label='True Path')

plt.legend()
plt.title("Particle Filter SLAM ")
plt.axis("equal")
plt.show()


#The accuracy of the genrated graphs r low -In side the logic its not mathematically perfect as I produced the graphs to get understanding about what happens inside the algorithms and filters 