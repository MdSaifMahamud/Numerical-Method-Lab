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

def get_newton_polynomial_string(x_points, dd_table, degree):
    """Generate Newton polynomial in Newton form"""
    n = degree + 1
    terms = []
    
    terms.append(f"{dd_table[0, 0]:.8f}")
    
    for k in range(1, n):
        coef = dd_table[0, k]
        
        # Build the product term
        product = []
        for j in range(k):
            if x_points[j] >= 0:
                product.append(f"(x - {x_points[j]:.1f})")
            else:
                product.append(f"(x + {abs(x_points[j]):.1f})")
        
        product_str = "".join(product)
        
        if coef >= 0:
            terms.append(f"+ {coef:.8f}{product_str}")
        else:
            terms.append(f"- {abs(coef):.8f}{product_str}")
    
    return " ".join(terms)

print("="*100)
print("TASK 2 — NEWTON'S DIVIDED-DIFFERENCE POLYNOMIAL")
print("="*100)
print("\nGiven Data:")
print(f"{'i':<5} {'xi':<10} {'yi':<15}")
print("-"*30)
for i, (x, y) in enumerate(data):
    print(f"{i:<5} {x:<10.1f} {y:<15.5f}")

x_target = 3.5

# ============================================
print("\n" + "="*100)
print("STEP 1: NODE SELECTION AND ORDERING")
print("="*100)
print(f"\nTarget point: x = {x_target}")
print("\nStrategy: Choose nodes by increasing distance from x = 3.5")
print("This centers the Newton form near x = 3.5 for best local accuracy.\n")

# Calculate distances from target
distances = np.abs(x_data - x_target)
sorted_indices = np.argsort(distances)

print(f"{'Node i':<10} {'xi':<10} {'yi':<15} {'Distance from 3.5':<20}")
print("-"*55)
for idx in sorted_indices:
    print(f"{idx:<10} {x_data[idx]:<10.1f} {y_data[idx]:<15.5f} {distances[idx]:<20.2f}")

print("\nOrdered node selection (by proximity to 3.5):")
ordered_x = x_data[sorted_indices]
ordered_y = y_data[sorted_indices]
for i, idx in enumerate(sorted_indices):
    print(f"Position {i+1}: Node {idx} (x={x_data[idx]:.1f}, y={y_data[idx]:.5f})")



print("\n" + "="*100)
print("STEP 2: DIVIDED DIFFERENCE TABLE")
print("="*100)
print("\nComputing full divided-difference table with 6-8 significant digits:\n")

# Compute full divided difference table with all nodes
dd_table_full = divided_difference_table(ordered_x, ordered_y)

# Print the table
n = len(ordered_x)
print(f"{'i':<5} {'xi':<10} {'f[xi]':<18}", end="")
for j in range(1, min(6, n)):
    if j == 1:
        print(f"{'f[xi,xi+1]':<18}", end="")
    elif j == 2:
        print(f"{'f[xi,xi+1,xi+2]':<18}", end="")
    else:
        print(f"{'f[...]':<18}", end="")
print()
print("-"*(28 + 18*min(6, n)))

for i in range(n):
    print(f"{i:<5} {ordered_x[i]:<10.2f} ", end="")
    for j in range(min(6, n)):
        if j <= n - 1 - i:
            print(f"{dd_table_full[i, j]:<18.8f}", end="")
        else:
            print(f"{'---':<18}", end="")
    print()

print("\nDivided difference coefficients (for Newton polynomial):")
print("These are the values from the first row of the table:")
for k in range(n):
    print(f"f[x0,...,x{k}] = {dd_table_full[0, k]:.8f}")

print("\n" + "="*100)
print("STEP 3: CONSTRUCT NEWTON POLYNOMIAL AND EVALUATE")
print("="*100)

results = []
degrees_list = []


for k in range(2, n):
    degree = k
    num_nodes = k + 1
    
    print(f"\n{'='*100}")
    print(f"DEGREE k = {degree} (using {num_nodes} nodes)")
    print(f"{'='*100}")
    
    # Select first num_nodes from ordered data
    x_selected = ordered_x[:num_nodes]
    y_selected = ordered_y[:num_nodes]
    
    print(f"\nNodes used:")
    for i in range(num_nodes):
        print(f"  x{i} = {x_selected[i]:.1f}, f(x{i}) = {y_selected[i]:.5f}")
    
    # Compute divided difference table for these nodes
    dd_table = divided_difference_table(x_selected, y_selected)
    
   
    print(f"\nNewton Polynomial N{degree}(x) in Newton form:")
    poly_str = get_newton_polynomial_string(x_selected, dd_table, degree)
    print(f"N{degree}(x) = {poly_str}")
    
  
    result = newton_interpolation(x_selected, y_selected, x_target)
    results.append(result)
    degrees_list.append(degree)
    
    print(f"\nEvaluation at x = {x_target}:")
    print(f"N{degree}({x_target}) = {result:.8f}")
    
    if len(results) > 1:
        delta = abs(results[-1] - results[-2])
        print(f"Δ{degree} = |N{degree}({x_target}) - N{degree-1}({x_target})| = {delta:.8e}")

# ============================================
# STEP 4: DELIVERABLES SUMMARY
# ============================================
print("\n" + "="*100)
print("STEP 4: DELIVERABLES FOR TASK 2")
print("="*100)

print("\n1. FULL DIVIDED-DIFFERENCE TABLE:")
print("   (See table above with all columns)")

print("\n2. NEWTON POLYNOMIALS IN NEWTON FORM:")
for k in range(2, n):
    num_nodes = k + 1
    x_selected = ordered_x[:num_nodes]
    dd_table = divided_difference_table(x_selected, ordered_y[:num_nodes])
    poly_str = get_newton_polynomial_string(x_selected, dd_table, k)
    print(f"\n   N{k}(x) = {poly_str}")

print("\n3. VALUES Nk(3.5) AND CONVERGENCE DIFFERENCES:")
print(f"\n{'Degree k':<12} {'Nk(3.5)':<20} {'Δk = |Nk(3.5) - Nk-1(3.5)|':<30}")
print("-"*62)
for i, (deg, val) in enumerate(zip(degrees_list, results)):
    if i == 0:
        print(f"{deg:<12} {val:<20.10f} {'---':<30}")
    else:
        delta = abs(results[i] - results[i-1])
        print(f"{deg:<12} {val:<20.10f} {delta:<30.10e}")



print("\n" + "="*100)
print("GENERATING PLOTS...")
print("="*100)

# Create only one plot
plt.figure(figsize=(8, 6))

# Calculate delta values (skip first one as it has no previous value)
degrees_delta = degrees_list[1:]  # Start from degree 3
delta_values = []
for i in range(1, len(results)):
    delta = abs(results[i] - results[i-1])
    delta_values.append(delta)

# Convergence difference plot (Δk vs degree)v
plt.plot(degrees_delta, delta_values, 's-', linewidth=2, markersize=8, color='red')
plt.xlabel('Polynomial Degree k')
plt.ylabel('Δk = |Nk(3.5) - Nk-1(3.5)|')
plt.title('Convergence Difference Δk vs Polynomial Degree')
plt.grid(True, alpha=0.3)
plt.yscale('log')  

plt.tight_layout()

plt.show()

print("\n" + "="*100)
print("TASK 2 COMPLETED SUCCESSFULLY")
print("="*100)