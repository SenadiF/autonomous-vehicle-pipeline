#Problem Setup 
#Track an object using State : postion + velocity 
# x = [px, py, vx, vy]
# Motion : linear (constant velocity)
# Measuremnet : Non linear 
# r = sqrt(px² + py²)
#θ = atan2(py, px)

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

#  Simulation

dt = 0.1
steps = 100

true_state = np.array([0.1, 0.1, 1.0, 0.5])
true_states = []
measurements = []

def h(x):
    px, py = x[0], x[1]
    r = np.sqrt(px**2 + py**2)
    theta = np.arctan2(py, px)
    return np.array([r, theta])
def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
for _ in range(steps):
    # motion
    true_state[0] += true_state[2] * dt
    true_state[1] += true_state[3] * dt
    true_states.append(true_state.copy())

    # measurement (nonlinear + noise)
    z = h(true_state)
    z[0] += np.random.normal(0, 0.5)
    z[1] += np.random.normal(0, 0.05)
    measurements.append(z)

true_states = np.array(true_states)
measurements = np.array(measurements)


# Common matrices

F = np.array([
    [1,0,dt,0],
    [0,1,0,dt],
    [0,0,1,0],
    [0,0,0,1]
])

Q = np.eye(4) * 0.01
R = np.array([[0.5,0],[0,0.05]])


#  KF 

H = np.array([
    [1,0,0,0],
    [0,1,0,0]
])

x_kf = np.zeros(4)
P_kf = np.eye(4)
kf_est = []


#  EKF
# Jacobian of h(x) for EKF
def H_jac(x):
    px, py = x[0], x[1]
    r = np.sqrt(px**2 + py**2)
    if r < 1e-3: r = 1e-3
    return np.array([
        [px/r, py/r, 0, 0],
        [-py/(r**2), px/(r**2), 0, 0]
    ])

x_ekf = np.array([1.0, 1.0, 0.0, 0.0])
P_ekf = np.eye(4)
ekf_est = []


#  UKF 

n = 4
alpha, beta, kappa = 0.001, 2, 0
lam = alpha**2*(n+kappa)-n

def sigma_points(x,P):
    pts=[x] #start with mean
    S = np.linalg.cholesky((n+lam)*P) # cholesky to get sqrt of covariance- It gives u directions to spread points around the mean
    for i in range(n): #Add point from both sides of the mean in each direction
        pts.append(x+S[:,i])
        pts.append(x-S[:,i])
    return np.array(pts)

Wm = np.full(2*n+1,1/(2*(n+lam)))
Wc = Wm.copy()
Wm[0]=lam/(n+lam)
Wc[0]=lam/(n+lam)+(1-alpha**2+beta)

x_ukf = np.zeros(4)
P_ukf = np.eye(4)
ukf_est = []


#  Particle Filter

num_particles = 500
particles = np.random.randn(num_particles,4)
weights = np.ones(num_particles)/num_particles
pf_est = []


# MAIN LOOP

for z in measurements:

    # ---- KF ----
    x_kf = F @ x_kf
    P_kf = F @ P_kf @ F.T + Q

    y = z - H @ x_kf  # WRONG measurement assumption
    S = H @ P_kf @ H.T + R
    K = P_kf @ H.T @ np.linalg.inv(S)

    x_kf += K @ y
    P_kf = (np.eye(4)-K@H) @ P_kf
    kf_est.append(x_kf.copy())

    # ---- EKF ----
    x_ekf = F @ x_ekf
    P_ekf = F @ P_ekf @ F.T + Q

    Hk = H_jac(x_ekf)
    z_pred = h(x_ekf)
    y = z - z_pred
    y[1] = normalize_angle(y[1])

    S = Hk @ P_ekf @ Hk.T + R
    K = P_ekf @ Hk.T @ np.linalg.inv(S)

    x_ekf += K @ y
    P_ekf = (np.eye(4)-K@Hk) @ P_ekf
    ekf_est.append(x_ekf.copy())

    # ---- UKF ----
    sig = sigma_points(x_ukf, P_ukf)
    sig_pred = np.array([F @ s for s in sig])

    x_pred = np.sum(Wm[:,None]*sig_pred, axis=0)

    P_pred = np.zeros((4,4))
    for i in range(2*n+1):
        d = (sig_pred[i]-x_pred).reshape(-1,1)
        P_pred += Wc[i]*d@d.T
    P_pred += Q

    Zsig = np.array([h(s) for s in sig_pred])
    z_pred = np.sum(Wm[:,None]*Zsig, axis=0)

    Pzz = np.zeros((2,2))
    Pxz = np.zeros((4,2))
    for i in range(2*n+1):
        dz = (Zsig[i]-z_pred).reshape(-1,1)
        dx = (sig_pred[i]-x_pred).reshape(-1,1)
        Pzz += Wc[i]*dz@dz.T
        Pxz += Wc[i]*dx@dz.T
    Pzz += R

    K = Pxz @ np.linalg.inv(Pzz)
    y = z - z_pred
    y[1] = normalize_angle(y[1])
    
    
    x_ukf = x_pred + K @ (z - z_pred)
    P_ukf = P_pred - K@Pzz@K.T
    ukf_est.append(x_ukf.copy())

    #  Particle Filter 
    particles[:,0] += particles[:,2] * dt 
    particles[:,1] += particles[:,3] * dt
    particles += np.random.normal(0,0.1,(num_particles,4))

    # weight
    z_particles = np.array([h(p) for p in particles])
    diff = z_particles - z
    diff[:,1] = np.array([normalize_angle(a) for a in diff[:,1]])
    dist = np.linalg.norm(diff, axis=1)
    weights = np.exp(-dist**2)
    weights += 1e-300
    weights /= np.sum(weights)

    # resample
    idx = np.random.choice(range(num_particles), num_particles, p=weights)
    particles = particles[idx]
    weights = np.ones(num_particles)/num_particles

    pf_est.append(np.mean(particles, axis=0))

# Plot

kf_est = np.array(kf_est)
ekf_est = np.array(ekf_est)
ukf_est = np.array(ukf_est)
pf_est = np.array(pf_est)

plt.figure()

plt.plot(true_states[:,0], true_states[:,1], 'k-', linewidth=3, label="True")
plt.plot(kf_est[:,0], kf_est[:,1], 'r--', label="KF")
plt.plot(ekf_est[:,0], ekf_est[:,1], 'go-', label="EKF", markersize=3)
plt.plot(ukf_est[:,0], ukf_est[:,1], 'b-', label="UKF", markersize=3)
plt.plot(pf_est[:,0], pf_est[:,1], 'm:', label="Particle", markersize=3)

plt.legend()
plt.title("Nonlinear Tracking Comparison")
plt.show()
print(np.isnan(ekf_est).any())