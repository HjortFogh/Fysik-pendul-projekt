import numpy as np
import matplotlib.pyplot as plt
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

# L0 = 0.165 # m
L0 = 0.235 # m
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


# cutoff_time = 35
cutoff_time = 35 / 2
cutoff_frame = int(cutoff_time * 30)
springconst = 1.966


# Create figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot for entire time
axs[0].plot(time, theta1_model(time, 1.925), c="darkorange", linestyle="-", label=f"Teoretiske vinkler med $k={springconst}$")
axs[0].scatter(time, theta1, s=4, marker="o", label="Målte vinkler", color="steelblue", zorder=2)
axs[0].set_xlabel("Tid i sekunder")
axs[0].set_ylabel("Vinkel $\\theta_1$ i radianer")
axs[0].set_title("Alle datapunker")
axs[0].grid(True)

# Plot for time interval [0; 1234]
axs[1].plot(time[:cutoff_frame], theta1_model(time, 1.925)[:cutoff_frame], c="darkorange", linestyle="-", label=f"Teoretiske vinkler med $k={springconst}$")
axs[1].scatter(time[:cutoff_frame], theta1[:cutoff_frame], s=4, marker="o", label="Målte vinkler", color="steelblue", zorder=2)
axs[1].set_xlabel("Tid i sekunder")
axs[1].set_title(f"Alle datapunker indtil {cutoff_time} sekunder")
axs[1].grid(True)
axs[1].legend()
axs[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Hide y-axis labels
axs[1].tick_params(axis='y', which='both', left=True, right=True)  # Show y-axis gridlines

fig.suptitle("Sammenligning af målte og teoretiske vinkler", fontsize=16)

plt.tight_layout()  # Adjust layout to prevent overlapping labels
plt.savefig(f"data/nice_{basename}_theta1.png")
# plt.show()


# Create figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot for entire time
axs[0].plot(time, theta2_model(time, 1.925), c="darkorange", linestyle="-", label=f"Teoretiske vinkler med $k={springconst}$")
axs[0].scatter(time, theta2, s=4, marker="o", label="Målte vinkler", color="steelblue", zorder=2)
axs[0].set_xlabel("Tid i sekunder")
axs[0].set_ylabel("Vinkel $\\theta_2$ i radianer")
axs[0].set_title("Alle datapunker")
axs[0].grid(True)

# Plot for time interval [0; 1234]
axs[1].plot(time[:cutoff_frame], theta2_model(time, 1.925)[:cutoff_frame], c="darkorange", linestyle="-", label=f"Teoretiske vinkler med $k={springconst}$")
axs[1].scatter(time[:cutoff_frame], theta2[:cutoff_frame], s=4, marker="o", label="Målte vinkler", color="steelblue", zorder=2)
axs[1].set_xlabel("Tid i sekunder")
axs[1].set_title(f"Alle datapunker indtil {cutoff_time} sekunder")
axs[1].grid(True)
axs[1].legend()
axs[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Hide y-axis labels
axs[1].tick_params(axis='y', which='both', left=True, right=True)  # Show y-axis gridlines

fig.suptitle("Sammenligning af målte og teoretiske vinkler", fontsize=16)

plt.tight_layout()  # Adjust layout to prevent overlapping labels
plt.savefig(f"data/nice_{basename}_theta2.png")
# plt.show()