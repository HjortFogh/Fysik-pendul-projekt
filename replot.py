import numpy as np
import matplotlib.pyplot as plt
import sys

basename = sys.argv[1]

try:
    with open(f"data/{basename}.txt", "r") as file:
        content = file.readlines()
        pendulum1, pendulum2 = content[0], content[1]
except FileNotFoundError:
    print(f"Invalid filepath: data/{basename}.txt")
    sys.exit()

theta1 = np.array([float(n) for n in pendulum1.split(",")])
theta2 = np.array([float(n) for n in pendulum2.split(",")])

end_time = theta1.shape[0] / 30
time = np.linspace(0, end_time, theta1.shape[0])

plt.plot(time, theta1, c="gold")
plt.plot(time, theta2, c="limegreen")
plt.savefig(f"data/{basename}.png")
