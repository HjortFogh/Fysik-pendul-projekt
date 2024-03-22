import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys

def find_peeks(y_data, side=0):
    indices = []
    for i in range(1, len(y_data) - 1):
        if side > -1 and y_data[i - 1] < y_data[i] and y_data[i + 1] < y_data[i]:
            indices.append(i)
        elif side < 1 and y_data[i - 1] > y_data[i] and y_data[i + 1] > y_data[i]:
            indices.append(i)
    return np.array(indices, dtype="int")

def find_initial_angle(y_data):
    peeks = find_peeks(y_data)
    indices = np.arange(0, y_data.shape[0], 1, dtype="int")[peeks][y_data[peeks] < -0.08]
    return indices[0]
    # return indices[1]

def energy_loss(time, y_data):
    peeks_idx = find_peeks(y_data)
    peeks = y_data[peeks_idx]

    top_half = peeks[peeks >= 0.02]
    top_half_idx = find_peeks(top_half, side=1)
    bot_half = peeks[peeks < -0.02]
    bot_half_idx = find_peeks(bot_half, side=-1)

    return time[peeks_idx][peeks >= 0][top_half_idx], top_half[top_half_idx], time[peeks_idx][peeks < 0][bot_half_idx], bot_half[bot_half_idx]
    # return time[peeks_idx][top_half_idx], top_half[top_half_idx], time[peeks_idx][bot_half_idx], np.abs(bot_half[bot_half_idx])


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


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # Adjust bottom margin to accommodate the slider

# Plot initial data and model
plt.scatter(time, theta1, s=1, c="orange")
# plt.scatter(time, theta2, s=1)
line, = plt.plot(time, theta1_model(time, springconst), alpha=0.7)
# line, = plt.plot(time, theta2_model(time, springconst), alpha=0.7)

# Define a function to update the plot based on the slider value
def update(val):
    springconst = slider.val
    line.set_ydata(theta1_model(time, springconst))
    # line.set_ydata(theta2_model(time, springconst))
    fig.canvas.draw_idle()

# Create a slider widget
ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Spring Const', 1.6, 2.4, valinit=springconst)

# Connect the slider widget to the update function
slider.on_changed(update)

# Show the plot
plt.show()


# summed_theta = theta1 + theta2
# summed_theta = theta1
# peeks = find_peeks(summed_theta)

# lin_points = np.abs(summed_theta[peeks])
# a, b = np.polyfit(time[peeks], lin_points, 1)

# print(f"Linear fit:\n{a:.4E} t + {b:.4E}")

# plt.scatter(time[peeks], lin_points, marker="x", alpha=0.6)
# plt.plot(time[peeks], a * time[peeks] + b)
# plt.show()


# m  = 0.352 # kg
# R  = 0.018 # m
# mu = 17e-6 # Pa*s
# k = ((6 * np.pi * mu * R) / m)

# summed_theta = theta1 + theta2

# peeks_idx = find_peeks(summed_theta)
# theta_peeks = np.abs(summed_theta[peeks_idx])

# k_fit = -9.14
# model = 2 * (k_fit - k_fit * np.exp(k * time[peeks_idx]))

# corrected = []
# for i in range(theta_peeks.shape[0]):
#     corrected.append(theta_peeks[i] + 2 * (k_fit - k_fit * np.exp(k * time[peeks_idx][i])))

# plt.scatter(time[peeks_idx], theta_peeks)
# # plt.plot(time, summed_theta)
# plt.scatter(time[peeks_idx], corrected)
# plt.plot(time[peeks_idx], np.abs(inital_angle) - model)
# plt.show()


# m  = 0.352 # kg
# R  = 0.018 # m
# mu = 17e-6 # Pa*s
# k = ((6 * np.pi * mu * R) / m)

# k_fit = 19
# # Forsøg1a..c
# model = np.abs(inital_angle) - (1 - np.exp(-k * time)) / (k * 4216.22)
# # Forsøg2a..c
# # model = np.abs(inital_angle) - (1 - np.exp(-k * time)) / (k * 5717.28)

# plt.scatter(time, theta1, s=1)
# plt.scatter(time, theta2, s=1)
# plt.plot(time, model)
# plt.ylim(-0.15, 0.15)
# plt.show()


# t1time1, t1top, t1time2, t1bot = energy_loss(time, theta1)
# t2time1, t2top, t2time2, t2bot = energy_loss(time, theta2)

# plt.scatter(t1time1, t1top)
# plt.scatter(t1time2, t1bot)
# plt.scatter(t2time1, t2top)
# plt.scatter(t2time2, t2bot)

# plt.plot(time, -0.000158617398497197 * time + 0.126230691647819)
# plt.scatter(time, theta1, s=1, c="gray")
# plt.scatter(time, theta2, s=1, c="gray")
# plt.show()


# plt.plot(time, theta1 + theta2, alpha=0.5)
# plt.plot(time, theta1)
# plt.plot(time, theta2)
# plt.show()