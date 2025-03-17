import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# Your Original Function
def newton_method(f, fprime, x, nmax, err1, err2, epsilon):
    """
    Newton's Method for finding roots of a function f.
    Arguments:
    f: The function.
    fprime: The derivative of the function.
    x: Initial guess for the root.
    nmax: Maximum number of iterations.
    err1: Convergence criterion for x.
    err2: Convergence criterion for f(x).
    epsilon: Small value to avoid division by zero.
    """
    fx = f(x)
    print(f"x: {x}\nf(x): {fx}")
    
    for i in range(1, nmax):
        fp = fprime(x)
        if abs(fp) < epsilon:
            print("Small Derivative")
            return
        
        d = fx / fp
        x = x - d
        fx = f(x)
        print(f"i: {i}\nx: {x}\nf(x): {fx}")
        if abs(d) < err1 or abs(fx) < err2:
            print("Converge")
            return f"Zero at ({x}, {fx})"

# Enhanced Function with Better Output Formatting
def newton_method_with_plot(f, fprime, x0, nmax, err1, err2, epsilon):
    """
    Newton's Method for finding roots with visualization support.
    """
    print("Starting Newton's Method...\n")
    x = x0
    fx = f(x)
    x_vals = [x]
    fx_vals = [fx]
    
    print(f"Initial guess:\nx = {x}\nf(x) = {fx}")
    
    for i in range(1, nmax + 1):
        fp = fprime(x)
        if abs(fp) < epsilon:
            print(f"\nIteration {i}:\nSmall derivative encountered. Stopping computation.")
            break
        
        d = fx / fp
        x -= d
        fx = f(x)
        x_vals.append(x)
        fx_vals.append(fx)
        print(f"\nIteration {i}:\n"
              f"x = {x}\n"
              f"f(x) = {fx}\n"
              f"difference = {d}")
        
        if abs(d) < err1 or abs(fx) < err2:
            print("\nConvergence achieved.")
            break
    
    return x_vals, fx_vals

# Animation Function
def newton_animation(f, fprime, x0, nmax, epsilon):
    # Initialize variables
    x_vals = [x0]
    fx_vals = [f(x0)]
    tangents = []
    
    x = x0
    for _ in range(nmax):
        fp = fprime(x)
        if abs(fp) < epsilon:
            break
        d = f(x) / fp
        x_new = x - d
        tangents.append((x, f(x), fp))
        x = x_new
        x_vals.append(x)
        fx_vals.append(f(x))
    
    return x_vals, fx_vals, tangents
