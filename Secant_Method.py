import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def secant_method(f, x0, x1, nmax, err1, err2, epsilon):
    """
    Secant Method for finding roots of a function f.
    Arguments:
    f: The function.
    x0: Initial guess for the root.
    x1: Second guess for the root.
    nmax: Maximum number of iterations.
    err1: Convergence criterion for x.
    err2: Convergence criterion for f(x).
    epsilon: Small value to avoid division by zero.
    """
    fx0 = f(x0)
    fx1 = f(x1)
    print(f"x0: {x0}\nf(x0): {fx0}\nx1: {x1}\nf(x1): {fx1}")
    
    for i in range(1, nmax):
        if abs(fx1 - fx0) < epsilon:
            print("Small Difference in Function Values")
            return
        
        d = fx1 * (x1 - x0) / (fx1 - fx0)
        x0, x1 = x1, x1 - d
        fx0, fx1 = fx1, f(x1)
        print(f"i: {i}\nx1: {x1}\nf(x1): {fx1}")
        if abs(d) < err1 or abs(fx1) < err2:
            print("Converge")
            return f"Zero at ({x1}, {fx1})"

def secant_method_with_plot(f, x0, x1, nmax, err1, err2, epsilon):
    """
    Secant Method for finding roots with visualization support.
    """
    print("Starting Secant Method...\n")
    fx0 = f(x0)
    fx1 = f(x1)
    x_vals = [x0, x1]
    fx_vals = [fx0, fx1]
    
    print(f"Initial guesses:\nx0 = {x0}\nf(x0) = {fx0}\nx1 = {x1}\nf(x1) = {fx1}")
    
    for i in range(1, nmax + 1):
        if abs(fx1 - fx0) < epsilon:
            print(f"\nIteration {i}:\nSmall difference in function values encountered. Stopping computation.")
            break
        
        d = fx1 * (x1 - x0) / (fx1 - fx0)
        x0, x1 = x1, x1 - d
        fx0, fx1 = fx1, f(x1)
        x_vals.append(x1)
        fx_vals.append(fx1)
        print(f"\nIteration {i}:\n"
              f"x1 = {x1}\n"
              f"f(x1) = {fx1}\n"
              f"difference = {d}")
        
        if abs(d) < err1 or abs(fx1) < err2:
            print("\nConvergence achieved.")
            break
    
    return x_vals, fx_vals

def secant_animation(f, x0, x1, nmax, epsilon):
    # Initialize variables
    x_vals = [x0, x1]
    fx_vals = [f(x0), f(x1)]
    secants = []
    
    for _ in range(nmax):
        if abs(fx_vals[-1] - fx_vals[-2]) < epsilon:
            break
        d = fx_vals[-1] * (x_vals[-1] - x_vals[-2]) / (fx_vals[-1] - fx_vals[-2])
        x_new = x_vals[-1] - d
        secants.append((x_vals[-2], fx_vals[-2], x_vals[-1], fx_vals[-1]))
        x_vals.append(x_new)
        fx_vals.append(f(x_new))
    
    return x_vals, fx_vals, secants