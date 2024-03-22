import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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

# L0 = 0.235 # m
L0 = 0.165 # m
L1 = 0.66  # m
m = 0.352  # kg
g = 9.82   # m/s^2

def theta1_model(t, k):
    omega0 = np.sqrt(g / L1)
    omega_dash = np.sqrt(2 * ((k * L0**2) / (m * L1**2)) + g / L1)
    return inital_angle * (np.cos((omega0 + omega_dash) * (t / 2)) * np.cos((omega0 - omega_dash) * (t / 2)))

def theta2_model(t, k):
    omega0 = np.sqrt(g / L1)
    omega_dash = np.sqrt(2 * ((k * L0**2) / (m * L1**2)) + g / L1)
    return inital_angle * (np.sin((omega0 + omega_dash) * (t / 2)) * np.sin((omega_dash - omega0) * (t / 2)))

springconst = 1.966

def calculate_residuals(time, theta, model, n=100, k_min=1.8, k_max=2.4):

    best_k = (k_min + k_max) / 2
    best_avrg = 1000000
    
    for k_test in np.linspace(k_min, k_max, n):
        residuals = theta - model(time, k_test)
        avrg = np.mean(residuals)
        
        if np.abs(avrg) < best_avrg:
            best_avrg = np.abs(avrg)
            best_k = k_test
    
    print("Best average:", best_avrg)
    return best_k

# k_fit = calculate_residuals(time, theta1, theta1_model, n=1, k_min=springconst)
k_fit = calculate_residuals(time, theta1, theta1_model)
print("Best spring constant:", k_fit)

plt.scatter(time, theta1, s=2)
plt.plot(time, theta1_model(time, k_fit))
plt.show()