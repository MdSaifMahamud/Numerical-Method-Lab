

import numpy as np
import matplotlib.pyplot as plt

class IterativeSolver:
    def __init__(self, A, b, x0, max_iter, tolerance):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.x0 = np.array(x0, dtype=float)
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.n = len(b)
        self._validate_inputs_and_fix_diagonal()
    
    def _fix_diagonal_by_swapping(self):
        n = self.A.shape[0]
        swapped_any = False

        for i in range(n):
            if abs(self.A[i, i]) < 1e-12:
                k_found = None
                for k in range(i + 1, n):
                    if abs(self.A[k, i]) >= 1e-12:
                        k_found = k
                        break
                if k_found is None:
                    for k in range(0, i):
                        if abs(self.A[k, i]) >= 1e-12:
                            k_found = k
                            break

                if k_found is not None:
                    print(f"Swapping rows {i+1} and {k_found+1} to fix zero diagonal at A[{i+1},{i+1}]")
                    self.A[[i, k_found], :] = self.A[[k_found, i], :]
                    self.b[[i, k_found]] = self.b[[k_found, i]]
                    swapped_any = True
                else:
                    return False, swapped_any

        for i in range(n):
            if abs(self.A[i, i]) < 1e-12:
                return False, swapped_any
        return True, swapped_any
   
    def _validate_inputs_and_fix_diagonal(self):
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Coefficient matrix A must be square")
        if self.A.shape[0] != len(self.b):
            raise ValueError("Dimensions of A and b do not match")
        if len(self.x0) != self.n:
            raise ValueError("Initial guess dimension does not match system size")
        if self.max_iter <= 0:
            raise ValueError("Maximum iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("Tolerance must be positive")

        ok, swapped_any = self._fix_diagonal_by_swapping()
        if not ok:
            raise ValueError("Diagonal contains zero that cannot be fixed by row swapping.")
        if swapped_any:
            print("Diagonal zero resolved by row swapping. Proceeding with the updated system.")

    def _calculate_residual(self, x):
        return np.linalg.norm(np.dot(self.A, x) - self.b, ord=2)
    
    def _calculate_absolute_errors(self, x_new, x_old):
        return np.abs(x_new - x_old)
    
    def jacobi_method(self, verbose=True):
        x = self.x0.copy()
        x_new = np.zeros_like(x)
        history = []
        
        if verbose:
            print("\n" + "=" * 100)
            print("JACOBI METHOD")
            print("=" * 100)
            print(f"{'Iter':<6} ", end='')
            for i in range(self.n):
                print(f"{'x'+str(i+1):<14} ", end='')
            print(f"{'Max Error':<14} {'Residual':<14}")
            print("-" * 100)
        
        for k in range(self.max_iter):
            x_old = x.copy()
            for i in range(self.n):
                sum_val = sum(self.A[i][j] * x[j] for j in range(self.n) if j != i)
                x_new[i] = (self.b[i] - sum_val) / self.A[i][i]
            
            abs_errors = self._calculate_absolute_errors(x_new, x_old)
            max_error = np.max(abs_errors)
            residual = self._calculate_residual(x_new)
            
            history.append({
                'iteration': k,
                'x': x_new.copy(),
                'abs_errors': abs_errors.copy(),
                'max_error': max_error,
                'residual': residual
            })
            
            x = x_new.copy()
            
            if verbose:
                print(f"{k:<6} ", end='')
                for i in range(self.n):
                    print(f"{x[i]:<14.8f} ", end='')
                print(f"{max_error:<14.8e} {residual:<14.8e}")
            
            if max_error < self.tolerance:
                if verbose:
                    print(f"\n Converged in {k+1} iterations!")
                return x, k+1, True, history
        
        if verbose:
            print(f"\n✗ Did not converge within {self.max_iter} iterations")
            print(f"  Final max error: {max_error:.8e}")
        
        return x, self.max_iter, False, history
    
    def gauss_seidel_method(self, verbose=True):
        x = self.x0.copy()
        history = []
        
        if verbose:
            print("\n" + "=" * 100)
            print("GAUSS-SEIDEL METHOD")
            print("=" * 100)
            print(f"{'Iter':<6} ", end='')
            for i in range(self.n):
                print(f"{'x'+str(i+1):<14} ", end='')
            print(f"{'Max Error':<14} {'Residual':<14}")
            print("-" * 100)
        
        for k in range(self.max_iter):
            x_old = x.copy()
            for i in range(self.n):
                sum_val = sum(self.A[i][j] * x[j] for j in range(i))
                sum_val += sum(self.A[i][j] * x[j] for j in range(i+1, self.n))
                x[i] = (self.b[i] - sum_val) / self.A[i][i]
            
            abs_errors = self._calculate_absolute_errors(x, x_old)
            max_error = np.max(abs_errors)
            residual = self._calculate_residual(x)
            
            history.append({
                'iteration': k,
                'x': x.copy(),
                'abs_errors': abs_errors.copy(),
                'max_error': max_error,
                'residual': residual
            })
            
            if verbose:
                print(f"{k:<6} ", end='')
                for i in range(self.n):
                    print(f"{x[i]:<14.8f} ", end='')
                print(f"{max_error:<14.8e} {residual:<14.8e}")
            
            if max_error < self.tolerance:
                if verbose:
                    print(f"\n Converged in {k+1} iterations!")
                return x, k+1, True, history
        
        if verbose:
            print(f"\n✗ Did not converge within {self.max_iter} iterations")
            print(f"  Final max error: {max_error:.8e}")
        
        return x, self.max_iter, False, history
    
    def verify_solution(self, x, method_name):
        print("\n" + "=" * 100)
        print(f"SOLUTION VERIFICATION - {method_name}")
        print("=" * 100)
        
        result = np.dot(self.A, x)
        print(f"\nFinal solution vector x*:")
        for i in range(self.n):
            print(f"  x{i+1} = {x[i]:.6f}")
        
        print(f"\nVerification (Ax should equal b):")
        print(f"{'Equation':<12} {'Ax':<16} {'b':<16} {'Difference':<16} {'Status':<8}")
        print("-" * 100)
        
        all_correct = True
        for i in range(self.n):
            diff = abs(result[i] - self.b[i])
            status = '✓' if diff < 1e-4 else '✗'
            if diff >= 1e-4:
                all_correct = False
            print(f"Eq {i+1:<9} {result[i]:<16.10f} {self.b[i]:<16.10f} {diff:<16.10e} {status:<8}")
        
        residual = np.linalg.norm(result - self.b, ord=2)
        print(f"\nResidual norm ||Ax - b||₂ = {residual:.10e}")
        
        if all_correct:
            print("✓ Solution verified successfully!")
        else:
            print("⚠ Solution verification shows significant errors")
    
    def compare_methods(self, jacobi_result, gauss_seidel_result):
        j_sol, j_iter, j_conv, j_hist = jacobi_result
        gs_sol, gs_iter, gs_conv, gs_hist = gauss_seidel_result
        
        print("\n" + "=" * 100)
        print("COMPARATIVE ANALYSIS")
        print("=" * 100)
        
        print(f"\n{'Metric':<30} {'Jacobi':<25} {'Gauss-Seidel':<25}")
        print("-" * 100)
        print(f"{'Convergence Status':<30} {'Converged' if j_conv else 'Not Converged':<25} {'Converged' if gs_conv else 'Not Converged':<25}")
        print(f"{'Number of Iterations':<30} {j_iter:<25} {gs_iter:<25}")
        
        if j_conv and gs_conv:
            print(f"{'Final Max Error':<30} {j_hist[-1]['max_error']:<25.10e} {gs_hist[-1]['max_error']:<25.10e}")
            print(f"{'Final Residual':<30} {j_hist[-1]['residual']:<25.10e} {gs_hist[-1]['residual']:<25.10e}")
            
            speedup = j_iter / gs_iter if gs_iter > 0 else float('inf')
            print(f"\n{'Iteration Speedup':<30} {speedup:<25.4f} (Gauss-Seidel is {speedup:.2f}x faster)")
        
        print("\nConvergence Behavior:")
        if gs_conv and j_conv and gs_iter < j_iter:
            print("  • Gauss-Seidel converged faster than Jacobi")
            print("  • This is typical for diagonally dominant systems")
        elif gs_conv and j_conv and gs_iter == j_iter:
            print("  • Both methods converged in the same number of iterations")
        elif j_conv and not gs_conv:
            print("  • Jacobi converged but Gauss-Seidel did not")
        elif gs_conv and not j_conv:
            print("  • Gauss-Seidel converged but Jacobi did not")
        
        print("\nComputational Efficiency:")
        print(f"  • Gauss-Seidel typically requires less memory as it updates in-place")
        print(f"  • For this system, Gauss-Seidel performed {'better' if gs_iter < j_iter else 'similarly'}")


def get_user_input():
    try:
        n = int(input("\nEnter number of equations: "))
        if n < 2:
            raise ValueError("Number of equations must be at least 2")
        
        print(f"\nEnter coefficient matrix A ({n}×{n}) row-wise:")
        A = []
        for i in range(n):
            row = list(map(float, input(f"Row {i+1}: ").split()))
            if len(row) != n:
                raise ValueError(f"Row {i+1} must have {n} elements")
            A.append(row)
        
        print(f"\nEnter constants vector b ({n} elements):")
        b = [float(input(f"b[{i+1}]: ")) for i in range(n)]
        
        print(f"\nEnter initial guess vector ({n} elements):")
        x0 = [float(input(f"x0[{i+1}]: ")) for i in range(n)]
        
        max_iter = int(input("Enter maximum iterations: "))
        if max_iter <= 0:
            raise ValueError("Maximum iterations must be positive")
        
        tolerance = float(input("Enter tolerance: "))
        if tolerance <= 0:
            raise ValueError("Tolerance must be positive")
        
        return A, b, x0, max_iter, tolerance
    except ValueError as e:
        print(f"\n✗ Input Error: {e}")
        return None


def main():
    print("-" * 100)
    print("LINEAR SYSTEM SOLVER - JACOBI AND GAUSS-SEIDEL METHODS")
    print("-" * 100)

    while True:
        result = get_user_input()
        if result:
            A, b, x0, max_iter, tolerance = result
            try:
                solver = IterativeSolver(A, b, x0, max_iter, tolerance)
                jacobi_result = solver.jacobi_method(verbose=True)
                gauss_seidel_result = solver.gauss_seidel_method(verbose=True)
                solver.verify_solution(jacobi_result[0], "JACOBI")
                solver.verify_solution(gauss_seidel_result[0], "GAUSS-SEIDEL")
                solver.compare_methods(jacobi_result, gauss_seidel_result)

    
                j_hist = jacobi_result[3]
                gs_hist = gauss_seidel_result[3]
                plt.figure(figsize=(7,5))
                if j_hist:
                    plt.plot([h['iteration']+1 for h in j_hist],
                             [h['residual'] for h in j_hist],
                             marker='o', label="Jacobi")
                if gs_hist:
                    plt.plot([h['iteration']+1 for h in gs_hist],
                             [h['residual'] for h in gs_hist],
                             marker='s', label="Gauss-Seidel")
                plt.xlabel("Iteration")
                plt.ylabel("Residual (||Ax - b||₂)")
                plt.title("Convergence Comparison")
                plt.grid(True, linestyle=":")
                plt.legend()
                plt.show()

            except Exception as e:
                print(f"\n✗ Solver Error: {e}")
        
        choice = input("\nDo you want to solve another system? (y/n): ").strip().lower()
        if choice != 'y':
            print("Exiting program. Goodbye!")
            break


if __name__ == "__main__":
    main()


