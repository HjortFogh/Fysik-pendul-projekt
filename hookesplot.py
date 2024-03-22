import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.0, 0.085, 0.125, 0.17, 0.21, 0.245, 0.29, 0.335])
y = np.array([0.0, 0.1964, 0.2946, 0.3928, 0.491, 0.5892, 0.6874, 0.7856])

# a, _ = np.polyfit(x, y, 1, full=False)
a = 2.357 
linear_func = a * x

plt.plot(x, linear_func, color="steelblue", label=f"$F(x) = {a:.3f}\\text{{ N/m}} \\cdot x$")
plt.scatter(x, y, c="darkorange", marker="o", zorder=2, s=75, alpha=0.8)

# plt.text(0.2, 0.6, f"$y = {a:.4f}x$", fontsize=12, color="steelblue")

plt.title("Fjederkraft som funktion af forlængelse")
plt.xlabel("$x$ (meter)")
plt.ylabel("$F$ (newton)")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("data/hookes_fig.png")
# plt.show()

