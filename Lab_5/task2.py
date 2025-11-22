import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate



# dy/dt = y * sin^3(t),   y(0) = 1
# y(t) = exp( -cos(t) + (cos^3(t))/3 + 2/3 )


def f(t, y):
    return y * (np.sin(t))**3

def y_true(t):
    return np.exp(-np.cos(t) + (np.cos(t)**3)/3 + 2/3)


# Euler Method

def euler_method(f, t0, y0, h, tn):
    steps = int((tn - t0) / h)
    ts = [t0]
    ys = [y0]
    t, y = t0, y0
    for _ in range(steps):
        y = y + h * f(t, y)
        t = t + h
        ts.append(t)
        ys.append(y)
    return np.array(ts), np.array(ys)


# Heun Method 

def heun_method(f, t0, y0, h, tn):
    steps = int((tn - t0) / h)
    ts = [t0]
    ys = [y0]
    t, y = t0, y0
    for _ in range(steps):
        y_predict = y + h * f(t, y)
        y = y + (h/2) * (f(t, y) + f(t + h, y_predict))
        t += h
        ts.append(t)
        ys.append(y)
    return np.array(ts), np.array(ys)

t0 = 0
y0 = 1
tn = 5

h_values = [1,0.1, 0.005, 0.001, 0.0005, 0.0001]

skip_dict = {0.1: 1, 0.005: 20, 0.001: 100, 0.0005: 200, 0.0001: 1000}

all_euler = {}
all_heun = {}


for h in h_values:
    skip = skip_dict.get(h, 1)
    print(f"\n==================== Step size h = {h} ====================\n")

    te, ye = euler_method(f, t0, y0, h, tn)
    th, yh = heun_method(f, t0, y0, h, tn)

    yt = y_true(te)

    all_euler[h] = (te, ye)
    all_heun[h] = (th, yh)

    # Errors
    err_euler = np.abs(yt - ye)
    err_heun = np.abs(yt - yh)

    table = []
    for i in range(0, len(te), skip):
        table.append([
            i,
            te[i],
            ye[i],
            yh[i],
            yt[i],
            err_euler[i],
            err_heun[i]
        ])

    headers = ["Step", "t", "Euler y", "Heun y", "Exact y",
               "Euler Error", "Heun Error"]

    print(tabulate(table, headers=headers, floatfmt=".8f"))


plt.figure(figsize=(13, 7))

# Plot exact solution
t_plot = np.linspace(0, 5, 800)
plt.plot(t_plot, y_true(t_plot), linewidth=3, label="Exact Solution")

# Plot Euler & Heun curves
for h in h_values:
    te, ye = all_euler[h]
    th, yh = all_heun[h]
    skip = skip_dict.get(h, 1)

    plt.plot(te[::skip], ye[::skip], linestyle='--', marker='.', label=f"Euler (h={h})")
    plt.plot(th[::skip], yh[::skip], linestyle=':', marker='x', label=f"Heun (h={h})")

plt.title("Lab Task 6.2 — Exact vs Euler vs Heun for Multiple Step Sizes (Skipped Points)")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid(True)
plt.legend()
plt.show()