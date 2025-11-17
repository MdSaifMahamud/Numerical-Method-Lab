# import numpy as np
# import matplotlib.pyplot as plt

# data = np.array([
#     [0, 2.00000],
#     [1, 5.43750],
#     [2.5, 7.35160],
#     [3, 7.56250],
#     [4.5, 8.44530],
#     [5, 9.18750],
#     [6, 12.00000]
# ])

# x_data = data[:, 0]
# y_data = data[:, 1]

# def lagrange_interpolation(x_points, y_points, x_eval):
#     """Lagrange interpolation at x_eval"""
#     n = len(x_points)
#     result = 0.0
    
#     for i in range(n):
#         Li = 1.0
#         for j in range(n):
#             if j != i:
#                 Li *= (x_eval - x_points[j]) / (x_points[i] - x_points[j])
#         result += y_points[i] * Li
    
#     return result

# def select_nearest_nodes(x_data, y_data, x_target, num_nodes):
#     """Select nodes closest to x_target"""
#     distances = np.abs(x_data - x_target)
#     indices = np.argsort(distances)[:num_nodes]
#     indices = np.sort(indices)
#     return x_data[indices], y_data[indices], indices

# print("="*80)
# print("LAB TASK 1 - LAGRANGE INTERPOLATION")
# print("="*80)
# print("\nGiven Data:")
# for i, (x, y) in enumerate(data):
#     print(f"i={i}: x={x}, y={y}")

# x_target = 3.5

# # Task 1.1(a) - Second-order (quadratic) using 3 nodes
# print("\n" + "="*80)
# print("Task 1.1(a) - Second-order (Quadratic) Lagrange interpolant P2(x)")
# print("="*80)

# x_sel, y_sel, indices = select_nearest_nodes(x_data, y_data, x_target, 3)
# print(f"\nChosen 3 nodes closest to x=3.5:")
# print(f"Indices: {indices}")
# print(f"x values: {x_sel}")
# print(f"y values: {y_sel}")

# P2_value = lagrange_interpolation(x_sel, y_sel, x_target)
# print(f"\nP2(3.5) = {P2_value:.8f}")

# # Task 1.2(a) - Cubic interpolant using 4 nodes
# print("\n" + "="*80)
# print("Task 1.2(a) - Cubic Lagrange interpolant P3(x)")
# print("="*80)

# x_sel, y_sel, indices = select_nearest_nodes(x_data, y_data, x_target, 4)
# print(f"\nChosen 4 nodes closest to x=3.5:")
# print(f"Indices: {indices}")
# print(f"x values: {x_sel}")
# print(f"y values: {y_sel}")

# P3_value = lagrange_interpolation(x_sel, y_sel, x_target)
# print(f"\nP3(3.5) = {P3_value:.8f}")
# print(f"Δ3 = |P3(3.5) - P2(3.5)| = {abs(P3_value - P2_value):.8e}")

# # Task 1.2(b) - Higher degree interpolants
# print("\n" + "="*80)
# print("Task 1.2(b) - Higher degree interpolants")
# print("="*80)

# results = [P2_value, P3_value]
# degrees = [2, 3]

# for num_nodes in [5, 6, 7]:
#     degree = num_nodes - 1
#     x_sel, y_sel, indices = select_nearest_nodes(x_data, y_data, x_target, num_nodes)
    
#     print(f"\nDegree {degree} - using {num_nodes} nodes:")
#     print(f"Indices: {indices}")
    
#     P_value = lagrange_interpolation(x_sel, y_sel, x_target)
#     results.append(P_value)
#     degrees.append(degree)
    
#     delta = abs(P_value - results[-2])
#     print(f"P{degree}(3.5) = {P_value:.8f}")
#     print(f"Δ{degree} = |P{degree}(3.5) - P{degree-1}(3.5)| = {delta:.8e}")

# # Task 1.3 - Deliverables: Summary table
# print("\n" + "="*80)
# print("Task 1.3 - SUMMARY TABLE")
# print("="*80)
# print(f"{'Degree k':<12} {'Pk(3.5)':<18} {'Δk':<18}")
# print("-"*50)
# for i, (deg, val) in enumerate(zip(degrees, results)):
#     if i == 0:
#         print(f"{deg:<12} {val:<18.8f} {'---':<18}")
#     else:
#         delta = abs(results[i] - results[i-1])
#         print(f"{deg:<12} {val:<18.8f} {delta:<18.8e}")

# # Plotting
# print("\n" + "="*80)
# print("Generating plots...")
# print("="*80)

# # Create only one plot
# plt.figure(figsize=(8, 6))

# # Calculate delta values (skip first one as it has no previous value)
# degrees_delta = degrees[1:]  # Start from degree 3
# delta_values = []
# for i in range(1, len(results)):
#     delta = abs(results[i] - results[i-1])
#     delta_values.append(delta)

# # Convergence difference plot (Δk vs degree)
# plt.plot(degrees_delta, delta_values, 'o-', linewidth=2, markersize=8, color='red')
# plt.xlabel('Polynomial Degree k')
# plt.ylabel('Δk = |Pk(3.5) - Pk-1(3.5)|')
# plt.title('Convergence Difference Δk vs Polynomial Degree')
# plt.grid(True, alpha=0.3)
# plt.yscale('log')  # Use log scale for better visualization of small differences

# plt.tight_layout()
# plt.savefig('lab_task1_delta_convergence.png', dpi=150)
# print("Plot saved as 'lab_task1_delta_convergence.png'")
# plt.show()
# x_vals = np.linspace(0, 6, 200)
# plt.figure(figsize=(8,6))
# plt.plot(x_data, y_data, 'ko', label='Data points')
# for num_nodes in [3, 4, 5, 6, 7]:
#     x_sel, y_sel, _ = select_nearest_nodes(x_data, y_data, x_target, num_nodes)
#     y_plot = [lagrange_interpolation(x_sel, y_sel, x) for x in x_vals]
#     plt.plot(x_vals, y_plot, label=f'P{num_nodes-1}(x)')
# plt.axvline(x_target, color='gray', linestyle='--')
# plt.xlabel('x'); plt.ylabel('y')
# plt.legend(); plt.grid(True)
# plt.title('Lagrange Interpolants (P2–P6)')
# plt.tight_layout()
# plt.show()


# print("\n" + "="*80)
# print("LAB TASK 1 COMPLETED")
# print("="*80)







import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [0, 2.00000],
    [1, 5.43750],
    [2.5, 7.35160],
    [3, 7.56250],
    [4.5, 8.44530],
    [5, 9.18750],
    [6, 12.00000]
])

x_data = data[:, 0]
y_data = data[:, 1]

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

def lagrange_basis_numeric(x_points, i, x_eval):
    """Compute the i-th Lagrange basis polynomial Li(x) at x_eval"""
    n = len(x_points)
    Li = 1.0
    for j in range(n):
        if j != i:
            Li *= (x_eval - x_points[j]) / (x_points[i] - x_points[j])
    return Li

def display_lagrange_basis(x_points, y_points):
    """Display Lagrange basis functions symbolically"""
    n = len(x_points)
    print("\nLagrange Basis Functions:")
    print("-" * 60)
    for i in range(n):
        numerator = []
        denominator = 1.0
        for j in range(n):
            if j != i:
                numerator.append(f"(x - {x_points[j]})")
                denominator *= (x_points[i] - x_points[j])
        
        num_str = " * ".join(numerator)
        print(f"L{i}(x) = {num_str} / {denominator:.6f}")
        print(f"       = {num_str} / {denominator:.6f}")

def compute_expanded_polynomial(x_points, y_points):
    """Compute expanded form coefficients of Lagrange polynomial"""
    n = len(x_points)
    result_poly = np.poly1d([0.0])
    
    for i in range(n):
        Li_poly = np.poly1d([1.0])
        denom = 1.0
        
        for j in range(n):
            if j != i:
                Li_poly = Li_poly * np.poly1d([1.0, -x_points[j]])
                denom *= (x_points[i] - x_points[j])
        Li_poly = Li_poly / denom
        result_poly = result_poly + y_points[i] * Li_poly
    
    return result_poly

def select_nearest_nodes(x_data, y_data, x_target, num_nodes):
    """Select nodes closest to x_target"""
    distances = np.abs(x_data - x_target)
    indices = np.argsort(distances)[:num_nodes]
    indices = np.sort(indices)  # Keep in original order
    return x_data[indices], y_data[indices], indices

# ============================================================================
print("="*80)
print("LAGRANGE INTERPOLATION")
print("="*80)
print("\nGiven Data:")
print(f"{'i':<5} {'x_i':<10} {'y_i':<10}")
print("-" * 25)
for i, (x, y) in enumerate(data):
    print(f"{i:<5} {x:<10} {y:<10.5f}")

x_target = 3.5

print("\n" + "="*80)
print("Task 1.1(a) - Second-order (Quadratic) Lagrange interpolant P2(x)")
print("="*80)

x_sel, y_sel, indices = select_nearest_nodes(x_data, y_data, x_target, 3)
print(f"\nNode Selection Strategy: Choose 3 nodes closest to x = {x_target}")
print(f"Selected indices: {indices}")
print(f"\nSelected nodes:")
print(f"{'i':<5} {'x_i':<10} {'y_i':<10}")
print("-" * 25)
for idx, (x, y) in zip(indices, zip(x_sel, y_sel)):
    print(f"{idx:<5} {x:<10} {y:<10.5f}")


display_lagrange_basis(x_sel, y_sel)

# Compute expanded polynomial
poly_P2 = compute_expanded_polynomial(x_sel, y_sel)
print("\nExpanded Polynomial P2(x):")
coeffs = poly_P2.coefficients
print(f"P2(x) = ", end="")
for i, c in enumerate(coeffs):
    power = len(coeffs) - 1 - i
    if i == 0:
        print(f"{c:.8f}*x^{power}", end="")
    else:
        sign = "+" if c >= 0 else ""
        if power == 0:
            print(f" {sign}{c:.8f}", end="")
        elif power == 1:
            print(f" {sign}{c:.8f}*x", end="")
        else:
            print(f" {sign}{c:.8f}*x^{power}", end="")
print()

# Evaluate P2(3.5)
P2_value = lagrange_interpolation(x_sel, y_sel, x_target)
print(f"\n**Evaluation at x = {x_target}:**")
print(f"P2({x_target}) = {P2_value:.8f}")

# ============================================================================
# Task 1.2(a) - Cubic interpolant using 4 nodes
# ============================================================================
print("\n" + "="*80)
print("Task 1.2(a) - Cubic Lagrange interpolant P3(x)")
print("="*80)

x_sel, y_sel, indices = select_nearest_nodes(x_data, y_data, x_target, 4)
print(f"\nNode Selection Strategy: Choose 4 nodes closest to x = {x_target}")
print(f"Selected indices: {indices}")
print(f"\nSelected nodes:")
print(f"{'i':<5} {'x_i':<10} {'y_i':<10}")
print("-" * 25)
for idx, (x, y) in zip(indices, zip(x_sel, y_sel)):
    print(f"{idx:<5} {x:<10} {y:<10.5f}")

# Display Lagrange basis functions
display_lagrange_basis(x_sel, y_sel)

# Compute expanded polynomial
poly_P3 = compute_expanded_polynomial(x_sel, y_sel)
print("\nExpanded Polynomial P3(x):")
coeffs = poly_P3.coefficients
print(f"P3(x) = ", end="")
for i, c in enumerate(coeffs):
    power = len(coeffs) - 1 - i
    if i == 0:
        print(f"{c:.8f}*x^{power}", end="")
    else:
        sign = "+" if c >= 0 else ""
        if power == 0:
            print(f" {sign}{c:.8f}", end="")
        elif power == 1:
            print(f" {sign}{c:.8f}*x", end="")
        else:
            print(f" {sign}{c:.8f}*x^{power}", end="")
print()

# Evaluate P3(3.5)
P3_value = lagrange_interpolation(x_sel, y_sel, x_target)
print(f"\n**Evaluation at x = {x_target}:**")
print(f"P3({x_target}) = {P3_value:.8f}")
print(f"Δ3 = |P3({x_target}) - P2({x_target})| = {abs(P3_value - P2_value):.8e}")

# ============================================================================
# Task 1.2(b) - Higher degree interpolants
# ============================================================================
print("\n" + "="*80)
print("Task 1.2(b) - Higher degree interpolants (P4, P5, P6)")
print("="*80)

results = [P2_value, P3_value]
degrees = [2, 3]
polynomials = [poly_P2, poly_P3]

for num_nodes in [5, 6, 7]:
    degree = num_nodes - 1
    x_sel, y_sel, indices = select_nearest_nodes(x_data, y_data, x_target, num_nodes)
    
    print(f"\n{'='*60}")
    print(f"Degree {degree} Interpolant P{degree}(x) - using {num_nodes} nodes")
    print(f"{'='*60}")
    print(f"Selected indices: {indices}")
    print(f"\nSelected nodes:")
    print(f"{'i':<5} {'x_i':<10} {'y_i':<10}")
    print("-" * 25)
    for idx, (x, y) in zip(indices, zip(x_sel, y_sel)):
        print(f"{idx:<5} {x:<10} {y:<10.5f}")
    
    # Compute expanded polynomial
    poly_Pk = compute_expanded_polynomial(x_sel, y_sel)
    polynomials.append(poly_Pk)
    
    print("\nExpanded Polynomial P{}(x):".format(degree))
    coeffs = poly_Pk.coefficients
    print(f"P{degree}(x) = ", end="")
    for i, c in enumerate(coeffs):
        power = len(coeffs) - 1 - i
        if i == 0:
            print(f"{c:.8f}*x^{power}", end="")
        else:
            sign = "+" if c >= 0 else ""
            if power == 0:
                print(f" {sign}{c:.8f}", end="")
            elif power == 1:
                print(f" {sign}{c:.8f}*x", end="")
            else:
                print(f" {sign}{c:.8f}*x^{power}", end="")
    print()
    
    # Evaluate
    P_value = lagrange_interpolation(x_sel, y_sel, x_target)
    results.append(P_value)
    degrees.append(degree)
    
    delta = abs(P_value - results[-2])
    print(f"\n**Evaluation at x = {x_target}:**")
    print(f"P{degree}({x_target}) = {P_value:.8f}")
    print(f"Δ{degree} = |P{degree}({x_target}) - P{degree-1}({x_target})| = {delta:.8e}")

print("\n" + "="*80)
print("Task 1.3 - SUMMARY TABLES AND DELIVERABLES")
print("="*80)

print("\n1. VALUES Pk(3.5) FOR EACH DEGREE:")
print("-" * 50)
print(f"{'Degree k':<12} {'Pk(3.5)':<20}")
print("-" * 50)
for deg, val in zip(degrees, results):
    print(f"{deg:<12} {val:<20.8f}")

print("\n2. CONVERGENCE DIFFERENCES Δk:")
print("-" * 50)
print(f"{'Degree k':<12} {'Δk = |Pk(3.5) - Pk-1(3.5)|':<30}")
print("-" * 50)
for i in range(1, len(results)):
    delta = abs(results[i] - results[i-1])
    print(f"{degrees[i]:<12} {delta:<30.8e}")

print("\n3. COMPLETE SUMMARY TABLE:")
print("-" * 60)
print(f"{'Degree k':<12} {'Pk(3.5)':<20} {'Δk':<20}")
print("-" * 60)
for i, (deg, val) in enumerate(zip(degrees, results)):
    if i == 0:
        print(f"{deg:<12} {val:<20.8f} {'---':<20}")
    else:
        delta = abs(results[i] - results[i-1])
        print(f"{deg:<12} {val:<20.8f} {delta:<20.8e}")

# ============================================================================
# PLOTTING
# ============================================================================
print("\n" + "="*80)
print("Generating plots...")
print("="*80)

# Plot 1: All Lagrange interpolants Pk(x)
plt.figure(figsize=(12, 6))

x_vals = np.linspace(0, 6, 300)

# Plot original data points
plt.plot(x_data, y_data, 'ko', markersize=8, label='Data points', zorder=5)

# Plot each interpolant
colors = ['blue', 'green', 'red', 'purple', 'orange']
for idx, num_nodes in enumerate([3, 4, 5, 6, 7]):
    degree = num_nodes - 1
    x_sel, y_sel, _ = select_nearest_nodes(x_data, y_data, x_target, num_nodes)
    y_plot = [lagrange_interpolation(x_sel, y_sel, x) for x in x_vals]
    plt.plot(x_vals, y_plot, label=f'P{degree}(x) (degree {degree})', 
             linewidth=2, color=colors[idx], alpha=0.7)

# Mark x = 3.5
plt.axvline(x_target, color='gray', linestyle='--', linewidth=1.5, 
            label=f'x = {x_target}', alpha=0.7)

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Lagrange Interpolating Polynomials P2(x) through P6(x)', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task1_lagrange_interpolants.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved as 'task1_lagrange_interpolants.png'")

# Plot 2: Convergence plot (Δk vs degree)
plt.figure(figsize=(10, 6))

degrees_delta = degrees[1:]
delta_values = [abs(results[i] - results[i-1]) for i in range(1, len(results))]

plt.plot(degrees_delta, delta_values, 'o-', linewidth=2.5, markersize=10, 
         color='crimson', markerfacecolor='yellow', markeredgewidth=2, markeredgecolor='crimson')

plt.xlabel('Polynomial Degree k', fontsize=12)
plt.ylabel('Δk = |Pk(3.5) - Pk-1(3.5)|', fontsize=12)
plt.title('Convergence: Difference Between Successive Interpolants', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.xticks(degrees_delta)

# Annotate points
for deg, delta in zip(degrees_delta, delta_values):
    plt.annotate(f'{delta:.2e}', xy=(deg, delta), xytext=(5, 5), 
                textcoords='offset points', fontsize=9)

plt.tight_layout()
plt.savefig('task1_convergence_differences.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved as 'task1_convergence_differences.png'")

plt.show()

# ============================================================================
# print("\n" + "="*80)
# print("LAB TASK 1 COMPLETED SUCCESSFULLY")
# print("="*80)
# print("\nDeliverables Generated:")
# print("  ✓ Lagrange basis functions (symbolic)")
# print("  ✓ Expanded polynomials with coefficients (≥6 significant digits)")
# print("  ✓ Values Pk(3.5) for all degrees k = 2, 3, 4, 5, 6")
# print("  ✓ Convergence differences Δk")
# print("  ✓ Summary tables")
# print("  ✓ Plot of Pk(x) for all degrees used")
# print("  ✓ Convergence plot")
# print("="*80)