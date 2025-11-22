import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# dy/dt = (1 + 4t) - y
# y(0) = 1
# y(t) = 4t - 3 + 4e^{-t}


def f(t, y):
    return (1 + 4*t) - y

def y_true(t):
    return 4*t - 3 + 4*np.exp(-t)


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
        yp = y + h * f(t, y)                     
        y = y + (h/2) * (f(t, y) + f(t + h, yp)) 
        t = t + h
        ts.append(t)
        ys.append(y)

    return np.array(ts), np.array(ys)


t0 = 0
y0 = 1
tn = 2

h_values = [1,0.5, 0.25, 0.1, 0.01, 0.001]


skip_dict = {0.5: 1, 0.25: 1, 0.1: 1, 0.01: 10, 0.001: 100}

all_euler = {}
all_heun = {}


for h in h_values:
    skip = skip_dict.get(h, 1)
    print(f"\n==================== h = {h} ====================\n")

    
    te, ye = euler_method(f, t0, y0, h, tn)
    th, yh = heun_method(f, t0, y0, h, tn)
    yt = y_true(te)

    
    all_euler[h] = (te, ye)
    all_heun[h] = (th, yh)

    
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

    print(tabulate(table, headers=headers, floatfmt=".6f"))




plt.figure(figsize=(12, 7))

# Plot exact curve 
t_plot = np.linspace(0, 2, 500)
plt.plot(t_plot, y_true(t_plot), label="Exact Solution", linewidth=3)

# Plot Euler & Heun 
for h in h_values:
    te, ye = all_euler[h]
    th, yh = all_heun[h]
    skip = skip_dict.get(h, 1)

    plt.plot(te[::skip], ye[::skip], marker='o', linestyle='--', label=f"Euler (h={h})")
    plt.plot(th[::skip], yh[::skip], marker='s', linestyle=':', label=f"Heun (h={h})")

plt.xlabel("t")
plt.ylabel("y")
plt.title("Exact vs Euler vs Heun for Multiple Step Sizes (Skipped Points)")
plt.grid(True)
plt.legend()
plt.show()