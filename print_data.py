import numpy as np
import sys

def find_peeks(y_data):
    indices = []
    for i in range(1, len(y_data) - 1):
        if y_data[i - 1] < y_data[i] and y_data[i + 1] < y_data[i]:
            indices.append(i)
        elif y_data[i - 1] > y_data[i] and y_data[i + 1] > y_data[i]:
            indices.append(i)
    return np.array(indices, dtype="int")

def find_initial_angle(y_data):
    peeks = find_peeks(y_data)
    indices = np.arange(0, y_data.shape[0], 1, dtype="int")[peeks][y_data[peeks] < -0.1]
    return indices[0]

basename = sys.argv[1]

try:
    with open(f"data/{basename}.txt", "r") as file:
        content = file.readlines()
        pendulum1, pendulum2 = content[1], content[0]
except FileNotFoundError:
    print(f"Invalid filepath: data/{basename}.txt")
    sys.exit()

theta1 = np.array([float(n) for n in pendulum1.split(",")])
theta2 = np.array([float(n) for n in pendulum2.split(",")])

initial_index = find_initial_angle(theta1)
inital_angle = theta1[initial_index]
print("Initial angle:", round(inital_angle, 4))

theta1 = theta1[initial_index:]
theta2 = theta2[initial_index:]

end_time = theta1.shape[0] / 30
time = np.linspace(0, end_time, theta1.shape[0])

t1 = list(time[:7]) + [time[-1]]
print(",".join([str(round(n, 3)) for n in t1]))

l1 = list(theta1[:7]) + [theta1[-1]]
print(",".join([str(round(n, 3)) for n in l1]))

l2 = list(theta2[:7]) + [theta2[-1]]
print(",".join([str(round(n, 3)) for n in l2]))