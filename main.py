"""
Root-Finding Methods Comparison

This program implements and compares three numerical methods for finding roots of nonlinear equations:
1. Bisection Method (linear convergence)
2. Newton's Method (quadratic convergence)
3. Secant Method (superlinear convergence)

The methods are tested on three different functions:
- f1(x) = x^2 - 4*sin(x)
- f2(x) = x^2 - 1
- f3(x) = x^3 - 3*x^2 + 3*x - 1

For each method and function, the program:
1. Finds at least one root
2. Analyzes convergence behavior
3. Compares performance metrics between methods
"""

from Bisect_Method import bisect_method_with_plot as bisect_plot, bisect_animation as bisect_anim
from Newton_Method import newton_method_with_plot as newton_plot, newton_animation as newton_anim
from Secant_Method import secant_method_with_plot as secant_plot, secant_animation as secant_anim
from Animation_Manager import manage_animations
import numpy as np
import matplotlib.pyplot as plt
import math

# Define test functions and their derivatives
def f1(x):
    """f1(x) = x^2 - 4*sin(x)"""
    return x**2 - 4 * np.sin(x)

def f2(x):
    """f2(x) = x^2 - 1"""
    return x**2 - 1

def f3(x):
    """f3(x) = x^3 - 3*x^2 + 3*x - 1"""
    return x**3 - 3*x**2 + 3*x - 1

def f1_prime(x):
    """Derivative of f1(x)"""
    return 2*x - 4 * np.cos(x)

def f2_prime(x):
    """Derivative of f2(x)"""
    return 2*x

def f3_prime(x):
    """Derivative of f3(x)"""
    return 3*x**2 - 6*x + 3

def plot_convergence_graphs(x_vals, fx_vals, method_name, fname, subplot_position):
    """
    Create convergence plots for a method showing:
    1. Function value convergence to 0
    2. Consecutive iteration difference convergence
    
    Parameters:
    - x_vals: List of x values from iterations
    - fx_vals: List of f(x) values from iterations
    - method_name: Name of the method being plotted
    - fname: Name of the function being analyzed
    - subplot_position: Position in the subplot grid
    """
    iterations = list(range(len(x_vals)))
    
    # Plot f(x) vs iterations
    plt.subplot(3, 2, subplot_position)
    plt.plot(iterations, fx_vals, marker='o', label=f"f(x) - {method_name}")
    plt.axhline(0, color='red', linestyle='--', label="y=0 (root line)")
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.title(f"Convergence of f(x) - {method_name} for {fname}")
    plt.legend()
    plt.yscale('symlog')  # Use symlog for better visualization of near-zero values
    plt.grid(True, alpha=0.3)
    
    # Plot absolute differences vs iterations
    plt.subplot(3, 2, subplot_position + 1)
    differences = [abs(x_vals[i + 1] - x_vals[i]) for i in range(len(x_vals) - 1)]
    plt.plot(iterations[:-1], differences, marker='x', label=f"|x_(i+1) - x_i| - {method_name}")
    plt.axhline(1e-12, color='green', linestyle='--', label="Convergence threshold")  # Updated threshold
    plt.xlabel("Iteration")
    plt.ylabel("|x_(i+1) - x_i|")
    plt.title(f"Convergence of x - {method_name} for {fname}")
    plt.legend()
    plt.yscale('log')  # Use log scale to better visualize convergence
    plt.grid(True, alpha=0.3)

def calculate_convergence_rate(x_vals):
    """
    Calculate the empirical convergence rate using consecutive errors
    
    For a sequence converging to x*, the convergence rate p can be estimated by:
    |e_{n+1}| ≈ C|e_n|^p, where e_n = x_n - x*
    
    Since we don't know x* exactly, we use the final value as an approximation
    """
    if len(x_vals) < 4:  # Need at least 4 points to estimate rate
        return "Insufficient data"
    
    # Use the last value as approximation of x*
    x_star = x_vals[-1]
    errors = [abs(x - x_star) for x in x_vals[:-1]]
    
    # Avoid log(0) errors by filtering out very small or zero errors
    filtered_errors = [(errors[i], errors[i+1]) 
                      for i in range(len(errors)-1) 
                      if errors[i] > 1e-10 and errors[i+1] > 1e-10]
    
    if len(filtered_errors) < 3:
        return "Insufficient data"
    
    # Calculate convergence rates for consecutive iterations
    rates = [math.log(e2) / math.log(e1) if e1 != 1 else 0 
             for e1, e2 in filtered_errors]
    
    # Return the average of the last few rates (more stable)
    return sum(rates[-3:]) / len(rates[-3:]) if rates else "Insufficient data"

def main() -> None:
    """
    Main function to test and compare root-finding methods
    """
    print("=" * 80)
    print("COMPARISON OF ROOT-FINDING METHODS")
    print("=" * 80)
    print("\nThis program compares the Bisection, Newton, and Secant methods for finding")
    print("roots of the following nonlinear equations:")
    print("  f1(x) = x^2 - 4*sin(x)")
    print("  f2(x) = x^2 - 1")
    print("  f3(x) = x^3 - 3*x^2 + 3*x - 1")
    print("\nTermination Criteria:")
    print("  1. |x_{k+1} - x_k| < 1e-12 (consecutive iterations close)")
    print("  2. |f(x_k)| < 1e-12 (function value close to zero)")
    print("  3. Maximum of 50 iterations")
    print("=" * 80 + "\n")

    functions = [
        (f1, f1_prime, "f1(x) = x^2 - 4*sin(x)"),
        (f2, f2_prime, "f2(x) = x^2 - 1"),
        (f3, f3_prime, "f3(x) = x^3 - 3*x^2 + 3*x - 1")
    ]

    convergence_data = []

    for f, fprime, fname in functions:
        print("\n" + "=" * 80)
        print(f"ANALYZING {fname}")
        print("=" * 80)

        print("\n1. Bisection Method:")
        # Select intervals that bracket roots for each function
        intervals = [(-4, 1), (-2, 2), (-1, 1), (-2, 0), (0, 2)]
        best_bisect_result = None
        used_interval = None
        
        for a, b in intervals:
            print(f"  Trying interval [{a}, {b}]...")
            # Updated error tolerances from 1e-6 to 1e-12
            bisect_results = bisect_plot(a=a, b=b, M=100, err1=1e-12, epsilon=1e-12, f=f)
            if not isinstance(bisect_results, str):
                best_bisect_result = bisect_results
                used_interval = (a, b)
                print(f"  Root found in interval [{a}, {b}]")
                break
            print(f"  Failed: {bisect_results}")
        
        # Set up default values in case bisection fails
        bisect_x_vals = []
        bisect_fx_vals = []
        
        if best_bisect_result is None:
            print(f"  Bisection Method failed for all intervals.")
        else:
            bisect_x_vals, bisect_fx_vals = best_bisect_result
            # Calculate empirical convergence rate
            bisect_rate = calculate_convergence_rate(bisect_x_vals)
            print(f"  Empirical convergence rate: {bisect_rate}")
            print(f"  Number of iterations: {len(bisect_x_vals)}")
            print(f"  Root found: {bisect_x_vals[-1]}")
            print(f"  f(root): {bisect_fx_vals[-1]}")
        
        print("\n2. Newton's Method:")
        # Updated error tolerances from 1e-6 to 1e-12
        newton_results = newton_plot(f=f, fprime=fprime, x0=1.5, nmax=100, err1=1e-12, err2=1e-12, epsilon=1e-12)
        newton_x_vals, newton_fx_vals = newton_results
        # Show animation with increased precision
        newton_anim(f=f, fprime=fprime, x0=1.5, nmax=100, epsilon=1e-12)
        
        # Calculate empirical convergence rate
        newton_rate = calculate_convergence_rate(newton_x_vals)
        print(f"  Empirical convergence rate: {newton_rate}")
        print(f"  Number of iterations: {len(newton_x_vals)}")
        print(f"  Root found: {newton_x_vals[-1]}")
        print(f"  f(root): {newton_fx_vals[-1]}")
        
        print("\n3. Secant Method:")
        # Updated error tolerances from 1e-6 to 1e-12
        secant_results = secant_plot(f=f, x0=0, x1=2, nmax=100, err1=1e-12, err2=1e-12, epsilon=1e-12)
        secant_x_vals, secant_fx_vals = secant_results
        # Show animation with increased precision
        secant_anim(f=f, x0=0, x1=2, nmax=100, epsilon=1e-12)
        
        # Calculate empirical convergence rate
        secant_rate = calculate_convergence_rate(secant_x_vals)
        print(f"  Empirical convergence rate: {secant_rate}")
        print(f"  Number of iterations: {len(secant_x_vals)}")
        print(f"  Root found: {secant_x_vals[-1]}")
        print(f"  f(root): {secant_fx_vals[-1]}")
        
        # Plot all convergence graphs in one window
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"Convergence Analysis for {fname}", fontsize=16)
        
        # Only plot bisection graphs if we have data
        if len(bisect_x_vals) > 0:
            plot_convergence_graphs(bisect_x_vals, bisect_fx_vals, "Bisection Method", fname, 1)
            # Add to convergence data
            convergence_data.append((
                fname, 
                "Bisection Method", 
                len(bisect_x_vals), 
                bisect_x_vals[-1], 
                bisect_fx_vals[-1],
                bisect_rate, 
                "Linear (theoretical)"
            ))
        
        plot_convergence_graphs(newton_x_vals, newton_fx_vals, "Newton's Method", fname, 3)
        plot_convergence_graphs(secant_x_vals, secant_fx_vals, "Secant Method", fname, 5)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        plt.show()

        # Collect convergence data for all methods
        if len(bisect_x_vals) == 0:  # Don't add bisection data if it failed
            print(f"  No convergence data for Bisection Method (method failed)")
        
        convergence_data.append((
            fname, 
            "Newton's Method", 
            len(newton_x_vals), 
            newton_x_vals[-1], 
            newton_fx_vals[-1],
            newton_rate, 
            "Quadratic (theoretical)"
        ))
        
        convergence_data.append((
            fname, 
            "Secant Method", 
            len(secant_x_vals), 
            secant_x_vals[-1], 
            secant_fx_vals[-1],
            secant_rate, 
            "Superlinear (theoretical)"
        ))

    # Print comprehensive comparison table
    print("\n\n" + "=" * 100)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("=" * 100)
    print(f"{'Function':<20}{'Method':<18}{'Iterations':<12}{'Root':<15}{'f(Root)':<15}{'Empirical Rate':<20}{'Theoretical'}")
    print(f"{'-'*100}")
    
    for data in convergence_data:
        fname, method, iterations, root, froot, emp_rate, theory = data
        print(f"{fname[:20]:<20}{method:<18}{iterations:<12}{root:<15.8g}{froot:<15.8g}{str(emp_rate)[:20]:<20}{theory}")
    
    print("\nConclusions:")
    print("1. Termination Criteria: Using both |x_{k+1} - x_k| < ε and |f(x_k)| < ε with ε=1e-12 provides")
    print("   highly accurate roots, getting much closer to the actual mathematical solution.")
    print("2. Convergence Rates: The empirical rates generally match theoretical expectations:")
    print("   - Bisection: Linear convergence (reducing error by a constant factor)")
    print("   - Newton: Quadratic convergence (error squared with each iteration)")
    print("   - Secant: Superlinear convergence (rate of ~1.62)")
    print("3. Method Comparison:")
    print("   - Newton's method typically converges in fewer iterations when a good initial guess is available")
    print("   - Bisection is more reliable but slower, requiring a bracket containing the root")
    print("   - Secant method offers a good compromise, with fast convergence and no need for derivatives")
    print("4. With decreased error tolerances (1e-12), we achieve approximately 12 digits of precision")

    # After all analyses and comparisons are complete, offer animations
    print("\n\n" + "=" * 80)
    print("INTERACTIVE ANIMATIONS")
    print("=" * 80)
    manage_animations(f1, f2, f3, f1_prime, f2_prime, f3_prime)

    print("\nProgram complete. Thank you for using the Root-Finding Methods Comparison tool.")

if __name__ == "__main__":
    main()