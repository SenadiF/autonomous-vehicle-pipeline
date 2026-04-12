import numpy as np
import matplotlib.pyplot as plt


# angle between -pi and pi
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

#Simulation of a moving object 

dt = 0.1 #time step
steps = 200 #iterarions

# True state: [x, y, vx, vy]
x_true = np.array([0.0, 0.0, 1.0, 0.5])

true_states = []
measurements = []

for _ in range(steps):
    # motion model (constant velocity)
    x_true[0] += x_true[2] * dt
    x_true[1] += x_true[3] * dt

    true_states.append(x_true.copy())

    # nonlinear measurement: range + angle to the object from the origin
    px, py = x_true[0], x_true[1]
    r = np.sqrt(px**2 + py**2)
    theta = np.arctan2(py, px)

    # add noise (to mimic a sensor)
    r_meas = r + np.random.normal(0, 0.5)
    theta_meas = theta + np.random.normal(0, 0.05)

    measurements.append([r_meas, theta_meas])

true_states = np.array(true_states)
measurements = np.array(measurements)



# Kalman Filter (linear KF)

#State estimation
F = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
#measurement model
H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])
#noise covariances
Q = np.eye(4) * 0.01
R_kf = np.eye(2) * 0.5

x_kf = np.array([0, 0, 0, 0])
P_kf = np.eye(4)

kf_estimates = []



# EKF


def h(x):
    px, py = x[0], x[1]
    r = np.sqrt(px**2 + py**2)
    theta = np.arctan2(py, px)
    return np.array([r, theta])
#linearizes the measurement 
def H_jacobian(x):
    px, py = x[0], x[1]
    r = np.sqrt(px**2 + py**2)

    if r < 1e-5:
        r = 1e-5

    H = np.array([
        [px / r, py / r, 0, 0],
        [-py / (r**2), px / (r**2), 0, 0]
    ])
    return H

x_ekf = np.array([0, 0, 0, 0])
P_ekf = np.eye(4)

ekf_estimates = []



# UKF (Use sigma points to handle nonlinearity)


n = 4
alpha = 0.001
beta = 2
kappa = 0

lambda_ = alpha**2 * (n + kappa) - n

def sigma_points(x, P):
    sigma = [x]
    sqrtP = np.linalg.cholesky((n + lambda_) * P)

    for i in range(n):
        sigma.append(x + sqrtP[:, i])
        sigma.append(x - sqrtP[:, i])

    return np.array(sigma)

def ukf_predict(sigma):
    predicted = []
    for s in sigma:
        s_new = F @ s
        predicted.append(s_new)
    return np.array(predicted)

def ukf_measure(sigma):
    Z = []
    for s in sigma:
        Z.append(h(s))
    return np.array(Z)

Wm = np.full(2*n + 1, 1 / (2*(n + lambda_)))
Wc = np.full(2*n + 1, 1 / (2*(n + lambda_)))
Wm[0] = lambda_ / (n + lambda_)
Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

x_ukf = np.array([0, 0, 0, 0])
P_ukf = np.eye(4)

ukf_estimates = []



# Main loop


R_ekf = np.array([[0.5, 0], [0, 0.5]])
R_ukf = R_ekf

for z in measurements:

    # -------- KF --------
    x_kf = F @ x_kf
    P_kf = F @ P_kf @ F.T + Q

    y = z - H @ x_kf
    S = H @ P_kf @ H.T + R_kf
    K = P_kf @ H.T @ np.linalg.inv(S)

    x_kf = x_kf + K @ y
    P_kf = (np.eye(4) - K @ H) @ P_kf

    kf_estimates.append(x_kf.copy())


    # -------- EKF --------
    x_ekf = F @ x_ekf
    P_ekf = F @ P_ekf @ F.T + Q

    Hk = H_jacobian(x_ekf)
    z_pred = h(x_ekf)

    y = z - z_pred
    y[1] = wrap_angle(y[1])

    S = Hk @ P_ekf @ Hk.T + R_ekf
    K = P_ekf @ Hk.T @ np.linalg.inv(S)

    x_ekf = x_ekf + K @ y
    P_ekf = (np.eye(4) - K @ Hk) @ P_ekf

    ekf_estimates.append(x_ekf.copy())

    #  UKF
    sigma = sigma_points(x_ukf, P_ukf)

    sigma_pred = ukf_predict(sigma)

    x_pred = np.sum(Wm[:, None] * sigma_pred, axis=0)

    P_pred = np.zeros((4,4))
    for i in range(2*n + 1):
        diff = (sigma_pred[i] - x_pred).reshape(-1,1)
        P_pred += Wc[i] * diff @ diff.T
    P_pred += Q

    Z_sigma = ukf_measure(sigma_pred)
    z_pred = np.sum(Wm[:, None] * Z_sigma, axis=0)

    P_zz = np.zeros((2,2))
    P_xz = np.zeros((4,2))

    for i in range(2*n + 1):
        dz = (Z_sigma[i] - z_pred).reshape(-1,1)
        dx = (sigma_pred[i] - x_pred).reshape(-1,1)

        dz[1] = wrap_angle(dz[1])

        P_zz += Wc[i] * dz @ dz.T
        P_xz += Wc[i] * dx @ dz.T

    P_zz += R_ukf

    K = P_xz @ np.linalg.inv(P_zz)

    y = z - z_pred
    y[1] = wrap_angle(y[1])

    x_ukf = x_pred + K @ y
    P_ukf = P_pred - K @ P_zz @ K.T

    ukf_estimates.append(x_ukf.copy())



# Plot results


kf_estimates = np.array(kf_estimates)
ekf_estimates = np.array(ekf_estimates)
ukf_estimates = np.array(ukf_estimates)

plt.figure()

plt.plot(true_states[:,0], true_states[:,1], label="True trajectory")
plt.scatter(*zip(*[(m[0]*np.cos(m[1]), m[0]*np.sin(m[1])) for m in measurements]),
            s=5, label="Measurements")

plt.plot(kf_estimates[:,0], kf_estimates[:,1], label="KF")
plt.plot(ekf_estimates[:,0], ekf_estimates[:,1], label="EKF")
plt.plot(ukf_estimates[:,0], ukf_estimates[:,1], label="UKF")

plt.legend()
plt.title("Kalman Filter Comparison")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()