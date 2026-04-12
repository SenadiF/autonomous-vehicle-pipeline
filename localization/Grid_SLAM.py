#Set up world
import numpy as np
import matplotlib.pyplot as plt

# Grid size
GRID_SIZE = 50

# Occupancy grid (0 = unknown, 1 = occupied probability)
grid = np.zeros((GRID_SIZE, GRID_SIZE))

# True obstacles (hidden map)
true_map = np.zeros((GRID_SIZE, GRID_SIZE))
true_map[20, 20:40] = 1
true_map[10:30, 30] = 1

# Robot state: x, y, theta
robot = np.array([10.0, 10.0, 0.0])

#Motion model
def move(x, u):
    x, y, th = x
    v, w = u

    th += w
    x += v * np.cos(th)
    y += v * np.sin(th)

    return np.array([x, y, th])

#Simulated LIDAR sensor
def lidar_scan(robot):
    x, y, _ = robot
    angles = np.linspace(0, 2*np.pi, 20)

    measurements = []

    for a in angles:
        for r in np.linspace(0, 25, 50):  

            xi = int(x + r*np.cos(a))
            yi = int(y + r*np.sin(a))

            # boundary check
            if xi < 0 or yi < 0 or xi >= GRID_SIZE or yi >= GRID_SIZE:
                break

            # stop at first obstacle hit 
            if true_map[xi, yi] == 1:
                measurements.append((xi, yi))
                break

    return measurements

#Grid update 
def update_grid(robot, measurements):
    rx, ry, _ = robot

    for mx, my in measurements:

        # convert to grid indices
        gx = int(mx)
        gy = int(my)

        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
            grid[gx, gy] += 0.6   # FIX: probabilistic update instead of overwrite
            grid[gx, gy] = min(grid[gx, gy], 1.0)

        # free space along ray
        steps = 15  # FIX: better ray coverage
        for i in range(steps):

            x = int(rx + i*(mx - rx)/steps)
            y = int(ry + i*(my - ry)/steps)

            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                grid[x, y] -= 0.05   # FIX: gradual free-space update
                grid[x, y] = np.clip(grid[x, y], 0, 1)

#Simulation loop
path = []

for t in range(50):

    # move robot
    u = np.array([1.0, 0.2])
    robot = move(robot, u)
    path.append(robot.copy())

    # sensor
    measurements = lidar_scan(robot)

    # map update
    update_grid(robot, measurements)

#Visualization 

plt.figure(figsize=(6,6))

# 
plt.imshow(grid.T, origin='lower', cmap='gray')

path = np.array(path)
plt.plot(path[:,1], path[:,0], 'r-') 

plt.title("Grid SLAM ")
plt.show()
