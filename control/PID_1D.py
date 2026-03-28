import matplotlib.pyplot as plt

# target position
target = 10

# initial state
position = 0

# PID gains
Kp = 0.6
Ki = 0.05
Kd = 0.3

dt = 0.1

integral = 0
previous_error = 0

positions = []

for i in range(100):

    error = target - position

    integral += error * dt
    derivative = (error - previous_error) / dt

    # PID output
    control = Kp * error + Ki * integral + Kd * derivative

    # update system (robot moves)
    position += control * dt

    positions.append(position)
    previous_error = error


plt.plot(positions, label="position")
plt.axhline(target, color='r', linestyle='--', label="target")
plt.legend()
plt.title("PID Control Simulation")
plt.show()