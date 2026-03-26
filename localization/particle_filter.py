# Particle Filter Implementation for Localization
#Tracking a moving point in 2D space
import numpy as np
import matplotlib.pyplot as plt


# Simulation setup


np.random.seed(42)

num_particles = 500
steps = 50
dt = 1.0

# True state: [x, y]
true_pos = np.array([0.0, 0.0])
velocity = np.array([1.0, 0.5])

true_positions = []
measurements = []


# Generate true path + noisy measurements


for _ in range(steps):
    # Move true object
    true_pos = true_pos + velocity * dt
    true_positions.append(true_pos.copy())

    # Add noise (sensor)
    meas = true_pos + np.random.normal(0, 2.0, size=2)
    measurements.append(meas)

true_positions = np.array(true_positions)
measurements = np.array(measurements)


#  Initialize particles

#Create 500 random guesses of a where the object might be 
particles = np.random.uniform(low=-10, high=10, size=(num_particles, 2))
weights = np.ones(num_particles) / num_particles


#  estimate state


def estimate(particles, weights):
    return np.average(particles, weights=weights, axis=0)


#  Particle Filter Loop


estimates = []

for z in measurements:

    
    # Prediction
  
    # Move particles with motion model + noise
    #Each particle mives like the true object but with some noise added to it
    particles += velocity * dt
    particles += np.random.normal(0, 0.5, size=particles.shape)

    
    #  Update (weighting)
  #Update the weights of each particle based on how close it is to the measurement
    # Compare particles to measurement

    distances = np.linalg.norm(particles - z, axis=1)

    # Convert distance to probability 
    weights = np.exp(- (distances**2) / (2 * 2.0**2))

    # Avoid division by zero
    weights += 1e-300
    #Normalize weights
    weights /= np.sum(weights)

    
    #  Resampling
    
    # Pick particles based on weights
    #High -weight particles-duplicate, low-weight particles - discard
    indices = np.random.choice(
        range(num_particles),
        size=num_particles,
        p=weights
    )

    particles = particles[indices]
    weights = np.ones(num_particles) / num_particles

    
    #  Estimate
    # 
    est = estimate(particles, weights)
    estimates.append(est)


#  Plot results


estimates = np.array(estimates)

plt.figure()

plt.plot(true_positions[:,0], true_positions[:,1], label="True Path")
plt.scatter(measurements[:,0], measurements[:,1], s=10, label="Measurements")
plt.plot(estimates[:,0], estimates[:,1], label="Particle Filter")

plt.legend()
plt.title("Particle Filter Tracking")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()