import numpy as np
import matplotlib.pyplot as plt

# Temperature sensor data from highway
data = np.array([
    [0, 25.0],
    [10, 26.7],
    [20, 29.4],
    [35, 33.2],
    [50, 35.5],
    [65, 36.1],
    [80, 37.8],
    [90, 38.9],
    [100, 40.0]
])

x_data = data[:, 0]  # Distance in km
y_data = data[:, 1]  # Temperature in °C

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def lagrange_interpolation(x_points, y_points, x_eval):
    """Lagrange interpolation at x_eval"""
    n = len(x_points)
    result = 0.0
    
    for i in range(n):
        Li = 1.0
        for j in range(n):
            if j != i:
                Li *= (x_eval - x_points[j]) / (x_points[i] - x_points[j])
        result += y_points[i] * Li
    
    return result

def divided_difference_table(x_points, y_points):
    """Construct divided difference table"""
    n = len(x_points)
    table = np.zeros((n, n))
    table[:, 0] = y_points
    
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x_points[i + j] - x_points[i])
    
    return table

def newton_interpolation(x_points, y_points, x_eval):
    """Newton's divided difference interpolation at x_eval"""
    n = len(x_points)
    table = divided_difference_table(x_points, y_points)
    
    result = table[0, 0]
    
    for k in range(1, n):
        term = table[0, k]
        for j in range(k):
            term *= (x_eval - x_points[j])
        result += term
    
    return result

def select_nearest_nodes(x_data, y_data, x_target, num_nodes):
    """Select nodes closest to x_target"""
    distances = np.abs(x_data - x_target)
    indices = np.argsort(distances)[:num_nodes]
    indices = np.sort(indices)
    return x_data[indices], y_data[indices], indices

# ============================================================================
# MAIN LAB REPORT
# ============================================================================

print("="*100)
print("LAB REPORT: TEMPERATURE VARIATION ALONG 100 KM HIGHWAY")
print("="*100)
print("\nTemperature Sensor Data:")
print(f"{'Sensor i':<10} {'Distance (km)':<20} {'Temperature (°C)':<20}")
print("-"*50)
for i, (x, y) in enumerate(data):
    print(f"{i:<10} {x:<20.1f} {y:<20.1f}")

x_target = 45.0  # Midpoint

print("\n" + "="*100)
print("TASK 1: LAGRANGE INTERPOLATION")
print("="*100)

# Store Lagrange results
lagrange_results = []
lagrange_degrees = []

# Task 1: Lagrange interpolation for different degrees
degrees_to_test = [2, 3, 4, len(x_data)-1]  # 2nd, 3rd, 4th, and full degree

for degree in degrees_to_test:
    num_nodes = degree + 1
    
    print(f"\n{'-'*100}")
    print(f"Lagrange Interpolation - Degree {degree} (using {num_nodes} nodes)")
    print(f"{'-'*100}")
    
    if num_nodes <= len(x_data):
        x_sel, y_sel, indices = select_nearest_nodes(x_data, y_data, x_target, num_nodes)
        
        print(f"\nSelected nodes closest to x = {x_target} km:")
        print(f"{'Index':<10} {'Distance (km)':<20} {'Temperature (°C)':<20}")
        print("-"*50)
        for idx, (x, y) in zip(indices, zip(x_sel, y_sel)):
            print(f"{idx:<10} {x:<20.1f} {y:<20.1f}")
        
        result = lagrange_interpolation(x_sel, y_sel, x_target)
        lagrange_results.append(result)
        lagrange_degrees.append(degree)
        
        print(f"\nInterpolated temperature at x = {x_target} km:")
        print(f"P{degree}({x_target}) = {result:.6f} °C")
        
        if len(lagrange_results) > 1:
            delta = abs(lagrange_results[-1] - lagrange_results[-2])
            print(f"Difference from previous: Δ{degree} = {delta:.8e} °C")

# ============================================================================
print("\n" + "="*100)
print("TASK 2: NEWTON'S DIVIDED DIFFERENCE INTERPOLATION")
print("="*100)

# Order nodes by proximity to target
distances = np.abs(x_data - x_target)
sorted_indices = np.argsort(distances)
ordered_x = x_data[sorted_indices]
ordered_y = y_data[sorted_indices]

print(f"\nNodes ordered by distance from x = {x_target} km:")
print(f"{'Position':<10} {'Index':<10} {'Distance (km)':<20} {'Temperature (°C)':<20}")
print("-"*60)
for i, idx in enumerate(sorted_indices):
    print(f"{i+1:<10} {idx:<10} {x_data[idx]:<20.1f} {y_data[idx]:<20.1f}")

# Compute divided difference table
dd_table = divided_difference_table(ordered_x, ordered_y)

print("\nDivided Difference Table:")
print(f"{'i':<5} {'xi':<12} {'f[xi]':<15} {'f[xi,xi+1]':<15} {'f[xi,xi+1,xi+2]':<18}")
print("-"*70)
for i in range(min(6, len(ordered_x))):
    print(f"{i:<5} {ordered_x[i]:<12.1f} ", end="")
    for j in range(min(4, len(ordered_x))):
        if j <= len(ordered_x) - 1 - i:
            print(f"{dd_table[i, j]:<15.8f} ", end="")
        else:
            print(f"{'---':<15} ", end="")
    print()

# Store Newton results
newton_results = []
newton_degrees = []

# Newton interpolation for different degrees
for degree in degrees_to_test:
    num_nodes = degree + 1
    
    print(f"\n{'-'*100}")
    print(f"Newton Interpolation - Degree {degree} (using {num_nodes} nodes)")
    print(f"{'-'*100}")
    
    if num_nodes <= len(ordered_x):
        x_sel = ordered_x[:num_nodes]
        y_sel = ordered_y[:num_nodes]
        
        result = newton_interpolation(x_sel, y_sel, x_target)
        newton_results.append(result)
        newton_degrees.append(degree)
        
        print(f"\nInterpolated temperature at x = {x_target} km:")
        print(f"N{degree}({x_target}) = {result:.6f} °C")
        
        if len(newton_results) > 1:
            delta = abs(newton_results[-1] - newton_results[-2])
            print(f"Difference from previous: Δ{degree} = {delta:.8e} °C")

# ============================================================================
print("\n" + "="*100)
print("TASK 3: COMPARISON OF METHODS")
print("="*100)

print("\nSummary Table:")
print(f"{'Degree':<10} {'Lagrange P(45)':<25} {'Newton N(45)':<25} {'Difference':<20}")
print("-"*80)
for i, deg in enumerate(lagrange_degrees):
    diff = abs(lagrange_results[i] - newton_results[i])
    print(f"{deg:<10} {lagrange_results[i]:<25.8f} {newton_results[i]:<25.8f} {diff:<20.2e}")

print("\nStability Analysis:")
print("\nLagrange Method - Successive Differences:")
for i in range(1, len(lagrange_results)):
    delta = abs(lagrange_results[i] - lagrange_results[i-1])
    print(f"  Δ{lagrange_degrees[i]} = {delta:.8e} °C")

print("\nNewton Method - Successive Differences:")
for i in range(1, len(newton_results)):
    delta = abs(newton_results[i] - newton_results[i-1])
    print(f"  Δ{newton_degrees[i]} = {delta:.8e} °C")

print("\nConclusion:")
print("Both methods yield numerically identical results (as expected mathematically).")
print("The differences are only due to numerical round-off errors in computation.")

# ============================================================================
print("\n" + "="*100)
print("TASK 4: GENERATING PLOTS")
print("="*100)

# Generate interpolation curves with 200+ points
x_plot = np.linspace(0, 100, 250)

# Figure 1: Lagrange Interpolation Curves
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.plot(x_data, y_data, 'ko', markersize=8, label='Sensor Data', zorder=5)
colors = ['blue', 'green', 'red', 'purple']
for idx, degree in enumerate(degrees_to_test):
    num_nodes = degree + 1
    x_sel, y_sel, _ = select_nearest_nodes(x_data, y_data, x_target, num_nodes)
    y_lagrange = [lagrange_interpolation(x_sel, y_sel, x) for x in x_plot]
    plt.plot(x_plot, y_lagrange, label=f'Degree {degree}', linewidth=2, color=colors[idx], alpha=0.7)
plt.axvline(x_target, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label=f'x = {x_target} km')
plt.xlabel('Distance (km)', fontsize=11)
plt.ylabel('Temperature (°C)', fontsize=11)
plt.title('Lagrange Interpolation - Different Degrees', fontsize=12, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

# Figure 2: Newton Interpolation Curves
plt.subplot(2, 2, 2)
plt.plot(x_data, y_data, 'ko', markersize=8, label='Sensor Data', zorder=5)
for idx, degree in enumerate(degrees_to_test):
    num_nodes = degree + 1
    x_sel = ordered_x[:num_nodes]
    y_sel = ordered_y[:num_nodes]
    y_newton = [newton_interpolation(x_sel, y_sel, x) for x in x_plot]
    plt.plot(x_plot, y_newton, label=f'Degree {degree}', linewidth=2, color=colors[idx], alpha=0.7)
plt.axvline(x_target, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label=f'x = {x_target} km')
plt.xlabel('Distance (km)', fontsize=11)
plt.ylabel('Temperature (°C)', fontsize=11)
plt.title('Newton Interpolation - Different Degrees', fontsize=12, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

# Figure 3: Convergence - Lagrange
plt.subplot(2, 2, 3)
lagrange_deltas = [abs(lagrange_results[i] - lagrange_results[i-1]) for i in range(1, len(lagrange_results))]
plt.plot(lagrange_degrees[1:], lagrange_deltas, 'o-', linewidth=2.5, markersize=10, 
         color='crimson', markerfacecolor='yellow', markeredgewidth=2)
plt.xlabel('Polynomial Degree', fontsize=11)
plt.ylabel('|P_k(45) - P_{k-1}(45)| (°C)', fontsize=11)
plt.title('Lagrange: Convergence with Increasing Degree', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.xticks(lagrange_degrees[1:])

# Figure 4: Convergence - Newton
plt.subplot(2, 2, 4)
newton_deltas = [abs(newton_results[i] - newton_results[i-1]) for i in range(1, len(newton_results))]
plt.plot(newton_degrees[1:], newton_deltas, 's-', linewidth=2.5, markersize=10, 
         color='darkblue', markerfacecolor='lightblue', markeredgewidth=2)
plt.xlabel('Polynomial Degree', fontsize=11)
plt.ylabel('|N_k(45) - N_{k-1}(45)| (°C)', fontsize=11)
plt.title('Newton: Convergence with Increasing Degree', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.xticks(newton_degrees[1:])

plt.tight_layout()
plt.savefig('highway_temperature_analysis.png', dpi=200, bbox_inches='tight')
print("\n✓ Complete analysis plot saved as 'highway_temperature_analysis.png'")

# Additional plot: Both methods on same graph
plt.figure(figsize=(12, 7))
plt.plot(x_data, y_data, 'ko', markersize=10, label='Sensor Data', zorder=5)

# Plot full-degree interpolation for both methods
num_nodes = len(x_data)
x_sel_lag, y_sel_lag, _ = select_nearest_nodes(x_data, y_data, x_target, num_nodes)
y_lagrange_full = [lagrange_interpolation(x_sel_lag, y_sel_lag, x) for x in x_plot]
plt.plot(x_plot, y_lagrange_full, label=f'Lagrange (Degree {num_nodes-1})', 
         linewidth=3, color='red', alpha=0.7)

y_newton_full = [newton_interpolation(ordered_x, ordered_y, x) for x in x_plot]
plt.plot(x_plot, y_newton_full, '--', label=f'Newton (Degree {num_nodes-1})', 
         linewidth=2, color='blue', alpha=0.7)

plt.axvline(x_target, color='green', linestyle=':', linewidth=2, 
            label=f'Target: x = {x_target} km', alpha=0.7)
plt.scatter([x_target], [lagrange_results[-1]], color='red', s=150, 
            marker='*', edgecolors='black', linewidth=1.5, zorder=6,
            label=f'T({x_target}) = {lagrange_results[-1]:.2f}°C')

plt.xlabel('Distance along Highway (km)', fontsize=13)
plt.ylabel('Temperature (°C)', fontsize=13)
plt.title('Temperature Interpolation: Lagrange vs Newton (Full Degree)', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('highway_temperature_comparison.png', dpi=200, bbox_inches='tight')
print("✓ Method comparison plot saved as 'highway_temperature_comparison.png'")

plt.show()

# ============================================================================
print("\n" + "="*100)
print("LAB REPORT COMPLETED SUCCESSFULLY")
print("="*100)
print(f"\nFINAL ANSWER: Temperature at x = {x_target} km:")
print(f"  Lagrange Method: {lagrange_results[-1]:.6f} °C")
print(f"  Newton Method:   {newton_results[-1]:.6f} °C")
print(f"\nBoth methods converge to the same value, confirming the accuracy of interpolation.")
print("="*100)