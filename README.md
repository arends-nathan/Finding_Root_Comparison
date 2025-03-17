# Root-Finding Methods Comparison

## Overview

This program implements and compares three numerical methods for finding roots of nonlinear equations:

1. **Bisection Method** (linear convergence)
2. **Newton's Method** (quadratic convergence)
3. **Secant Method** (superlinear convergence)

The methods are tested on three different functions:

- f₁(x) = x² - 4·sin(x)
- f₂(x) = x² - 1
- f₃(x) = x³ - 3x² + 3x - 1

## Features

- Implementation of three classic root-finding algorithms
- Detailed convergence analysis with empirical rate calculation
- Interactive visualizations showing method behavior
- Step-by-step animations of each algorithm's progression
- Comprehensive performance comparisons

## How to Run

1. Execute `main.py` to run the full comparison:
   ```
   python main.py
   ```
2. Follow the on-screen prompts to view animations for specific methods and functions
3. Review the convergence graphs and performance data

## File Structure

- `main.py` - Primary program execution and comparison logic
- `Bisect_Method.py` - Implementation of the Bisection Method
- `Newton_Method.py` - Implementation of Newton's Method
- `Secant_Method.py` - Implementation of the Secant Method
- `Animation_Manager.py` - Handles visualization of the algorithms

## Termination Criteria

The program uses the following stopping conditions:

- |xₙ₊₁ - xₙ| < 10⁻¹² (consecutive iterations are sufficiently close)
- |f(xₙ)| < 10⁻¹² (function value is sufficiently close to zero)
- Maximum of 50 iterations reached

## Results

The program provides:

- Root approximations for each method
- Number of iterations required
- Final function value at the approximation
- Empirical convergence rate
- Visual comparison of convergence behavior

## Dependencies

- NumPy
- Matplotlib

## Conclusions

1. **Newton's Method** typically converges in fewer iterations when provided with a good initial guess
2. **Bisection Method** is more reliable but generally slower, requiring a bracket containing the root
3. **Secant Method** offers a good compromise, combining fast convergence without requiring derivatives
4. The empirical convergence rates confirm theoretical expectations:
   - Bisection: Linear (reducing error by a constant factor)
   - Newton: Quadratic (error squared with each iteration)
   - Secant: Superlinear (rate of approximately 1.62)
