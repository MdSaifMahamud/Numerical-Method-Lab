
import numpy as np
import matplotlib.pyplot as plt

# Exact solution for Task 1
def y_exact(t):
    return 4*t - 3 + 4*np.exp(-t)

# Differential equation dy/dt = (1+4t) - y
def f(t, y):
    return (1 + 4*t) - y

def euler(f, t0, y0, h, t_end):
    t_vals = np.arange(t0, t_end + h, h)
    y_vals = np.zeros(len(t_vals))
    y_vals[0] = y0
    for i in range(1, len(t_vals)):
        y_vals[i] = y_vals[i-1] + h * f(t_vals[i-1], y_vals[i-1])
    return t_vals, y_vals

def heun(f, t0, y0, h, t_end):
    t_vals = np.arange(t0, t_end + h, h)
    y_vals = np.zeros(len(t_vals))
    y_vals[0] = y0
    for i in range(1, len(t_vals)):
        y_pred = y_vals[i-1] + h * f(t_vals[i-1], y_vals[i-1])
        y_vals[i] = y_vals[i-1] + (h/2) * (f(t_vals[i-1], y_vals[i-1]) + f(t_vals[i], y_pred))
    return t_vals, y_vals

steps = [0.5, 0.25, 0.1, 0.01, 0.001]

plt.figure(figsize=(10,6))
t_exact = np.linspace(0,5,5000)
plt.plot(t_exact, y_exact(t_exact), label="Exact", linewidth=2)

for h in steps:
    t_e, y_e = euler(f, 0, 1, h, 5)
    t_h, y_h = heun(f, 0, 1, h, 5)
    plt.plot(t_e, y_e, label=f"Euler h={h}")
    plt.plot(t_h, y_h, label=f"Heun h={h}")

plt.legend()
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Task 1: Exact vs Euler vs Heun")
plt.grid()
plt.show()
