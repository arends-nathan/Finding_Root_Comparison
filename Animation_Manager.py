"""
Animation Manager for Root-Finding Methods

This module handles user interactions for displaying animations of the
root-finding methods (Bisection, Newton, and Secant) for different functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Bisect_Method import bisect_animation
from Newton_Method import newton_animation
from Secant_Method import secant_animation

def display_function(f, fname, x_range):
    """Display the function graph with the specified range."""
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = [f(xi) for xi in x]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.title(f"{fname} Graph", fontsize=16)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    
    # Mark x-axis and y-axis
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.show()

def animate_bisection(f, a, b, M, epsilon, fname):
    """Create and display animation for the bisection method."""
    try:
        x_vals, fx_vals, intervals = bisect_animation(a, b, M, epsilon, f)
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot the function
        x_range = (min(a, b) - 1, max(a, b) + 1)
        x = np.linspace(x_range[0], x_range[1], 1000)
        y = [f(xi) for xi in x]
        ax.plot(x, y, 'b-', label=fname, linewidth=2)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='y=0')
        
        # Initialize interval line and point
        interval_line, = ax.plot([], [], 'g-', linewidth=2, label='Current Interval')
        point, = ax.plot([], [], 'ro', markersize=8, label='Current Point')
        
        # Add labels and legend
        ax.set_title(f"Bisection Method Animation - {fname}", fontsize=16)
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("f(x)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Set axis limits
        ax.set_xlim(x_range)
        y_min, y_max = min(y), max(y)
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)
        
        # Initialize text elements
        iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        value_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        
        def init():
            interval_line.set_data([], [])
            point.set_data([], [])
            iteration_text.set_text('')
            value_text.set_text('')
            return interval_line, point, iteration_text, value_text
        
        def update(frame):
            if frame < len(x_vals):
                # Update interval line
                if frame < len(intervals):
                    a_i, b_i = intervals[frame]
                    interval_x = [a_i, b_i]
                    interval_y = [0, 0]
                    interval_line.set_data(interval_x, interval_y)
                
                # Update point
                point.set_data([x_vals[frame]], [fx_vals[frame]])
                
                # Update text
                iteration_text.set_text(f'Iteration: {frame}')
                value_text.set_text(f'x = {x_vals[frame]:.8f}, f(x) = {fx_vals[frame]:.8f}')
            
            return interval_line, point, iteration_text, value_text
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(x_vals)+5, init_func=init, 
                             interval=1000, blit=True)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Animation error: {e}")
        return

def animate_newton(f, fprime, x0, nmax, epsilon, fname):
    """Create and display animation for Newton's method."""
    try:
        x_vals, fx_vals, tangents = newton_animation(f, fprime, x0, nmax, epsilon)
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Determine x range based on x_vals
        x_min, x_max = min(x_vals) - 1, max(x_vals) + 1
        
        # Plot the function
        x = np.linspace(x_min, x_max, 1000)
        y = [f(xi) for xi in x]
        ax.plot(x, y, 'b-', label=fname, linewidth=2)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='y=0')
        
        # Initialize tangent line and point
        tangent_line, = ax.plot([], [], 'g-', linewidth=2, label='Tangent Line')
        point, = ax.plot([], [], 'ro', markersize=8, label='Current Point')
        next_point, = ax.plot([], [], 'mo', markersize=8, label='Next Approximation')
        
        # Add labels and legend
        ax.set_title(f"Newton's Method Animation - {fname}", fontsize=16)
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("f(x)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Set axis limits
        ax.set_xlim(x_min, x_max)
        y_min, y_max = min(y), max(y)
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)
        
        # Initialize text elements
        iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        value_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        
        def init():
            tangent_line.set_data([], [])
            point.set_data([], [])
            next_point.set_data([], [])
            iteration_text.set_text('')
            value_text.set_text('')
            return tangent_line, point, next_point, iteration_text, value_text
        
        def update(frame):
            if frame < len(tangents):
                x_i, fx_i, fp_i = tangents[frame]
                
                # Calculate tangent line
                x_tangent = np.linspace(x_i - 1, x_i + 1, 100)
                y_tangent = [fx_i + fp_i * (xt - x_i) for xt in x_tangent]
                tangent_line.set_data(x_tangent, y_tangent)
                
                # Update points
                point.set_data([x_i], [fx_i])
                
                # Calculate next x value (where tangent intersects x-axis)
                next_x = x_i - fx_i / fp_i
                next_point.set_data([next_x], [0])
                
                # Update text
                iteration_text.set_text(f'Iteration: {frame}')
                value_text.set_text(f'x = {x_i:.8f}, f(x) = {fx_i:.8f}')
            
            return tangent_line, point, next_point, iteration_text, value_text
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(tangents)+5, init_func=init, 
                             interval=1000, blit=True)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Animation error: {e}")
        return

def animate_secant(f, x0, x1, nmax, epsilon, fname):
    """Create and display animation for the secant method."""
    try:
        x_vals, fx_vals, secants = secant_animation(f, x0, x1, nmax, epsilon)
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Determine x range based on x_vals
        x_min, x_max = min(x_vals) - 1, max(x_vals) + 1
        
        # Plot the function
        x = np.linspace(x_min, x_max, 1000)
        y = [f(xi) for xi in x]
        ax.plot(x, y, 'b-', label=fname, linewidth=2)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='y=0')
        
        # Initialize secant line and points
        secant_line, = ax.plot([], [], 'g-', linewidth=2, label='Secant Line')
        point1, = ax.plot([], [], 'ro', markersize=6, label='Previous Point')
        point2, = ax.plot([], [], 'mo', markersize=6, label='Current Point')
        next_point, = ax.plot([], [], 'go', markersize=8, label='Next Approximation')
        
        # Add labels and legend
        ax.set_title(f"Secant Method Animation - {fname}", fontsize=16)
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("f(x)", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Set axis limits
        ax.set_xlim(x_min, x_max)
        y_min, y_max = min(y), max(y)
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)
        
        # Initialize text elements
        iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        value_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        
        def init():
            secant_line.set_data([], [])
            point1.set_data([], [])
            point2.set_data([], [])
            next_point.set_data([], [])
            iteration_text.set_text('')
            value_text.set_text('')
            return secant_line, point1, point2, next_point, iteration_text, value_text
        
        def update(frame):
            if frame < len(secants):
                x0_i, fx0_i, x1_i, fx1_i = secants[frame]
                
                # Calculate secant line
                secant_line.set_data([x0_i, x1_i], [fx0_i, fx1_i])
                
                # Update points
                point1.set_data([x0_i], [fx0_i])
                point2.set_data([x1_i], [fx1_i])
                
                # Calculate next x value (where secant intersects x-axis)
                slope = (fx1_i - fx0_i) / (x1_i - x0_i)
                next_x = x1_i - fx1_i / slope if abs(slope) > 1e-10 else x1_i
                next_point.set_data([next_x], [0])
                
                # Update text
                iteration_text.set_text(f'Iteration: {frame}')
                value_text.set_text(f'x1 = {x1_i:.8f}, f(x1) = {fx1_i:.8f}')
            
            return secant_line, point1, point2, next_point, iteration_text, value_text
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(secants)+5, init_func=init, 
                             interval=1000, blit=True)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Animation error: {e}")
        return

def manage_animations(f1, f2, f3, f1_prime, f2_prime, f3_prime):
    """
    Main function to manage animations based on user input.
    
    Parameters:
    - f1, f2, f3: The three test functions
    - f1_prime, f2_prime, f3_prime: Their derivatives
    """
    functions = {
        '1': (f1, f1_prime, "f1(x) = x^2 - 4*sin(x)", (-3, 3)),
        '2': (f2, f2_prime, "f2(x) = x^2 - 1", (-2, 2)),
        '3': (f3, f3_prime, "f3(x) = x^3 - 3*x^2 + 3*x - 1", (-1, 2))
    }
    
    methods = {
        '1': "Bisection Method",
        '2': "Newton's Method",
        '3': "Secant Method"
    }
    
    while True:
        print("\n" + "=" * 50)
        print("ROOT-FINDING METHOD ANIMATION VIEWER")
        print("=" * 50)
        
        # Ask if user wants to see animations
        show_animation = input("\nWould you like to see method animations? (y/n): ").strip().lower()
        
        if show_animation != 'y':
            print("Returning to main program...")
            return
        
        # Select function
        print("\nSelect a function:")
        for key, (_, _, name, _) in functions.items():
            print(f"{key}: {name}")
        
        function_choice = input("\nEnter function number (1-3): ").strip()
        
        if function_choice not in functions:
            print("Invalid choice. Please try again.")
            continue
        
        f, fprime, fname, x_range = functions[function_choice]
        
        # Show function graph
        print(f"\nDisplaying graph for {fname}...")
        display_function(f, fname, x_range)
        
        # Select method
        print("\nSelect a method to animate:")
        for key, name in methods.items():
            print(f"{key}: {name}")
        
        method_choice = input("\nEnter method number (1-3): ").strip()
        
        if method_choice not in methods:
            print("Invalid choice. Please try again.")
            continue
        
        print(f"\nPreparing animation for {methods[method_choice]} on {fname}...")
        
        # Run the appropriate animation
        if method_choice == '1':  # Bisection
            # Select appropriate interval based on function
            if function_choice == '1':
                intervals = [(-4, 1), (1, 4)]
            elif function_choice == '2':
                intervals = [(-2, 0), (0, 2)]
            else:
                intervals = [(-1, 2)]
                
            print("\nSelect an interval:")
            for i, interval in enumerate(intervals):
                print(f"{i+1}: [{interval[0]}, {interval[1]}]")
            
            interval_choice = input("\nEnter interval number: ").strip()
            
            try:
                idx = int(interval_choice) - 1
                if 0 <= idx < len(intervals):
                    a, b = intervals[idx]
                    animate_bisection(f, a, b, 25, 1e-12, fname)
                else:
                    print("Invalid interval choice.")
            except ValueError:
                print("Please enter a valid number.")
                
        elif method_choice == '2':  # Newton
            # Select appropriate starting point based on function
            if function_choice == '1':
                x0_options = [0.5, 2.0]
            elif function_choice == '2':
                x0_options = [-0.5, 0.5]
            else:
                x0_options = [0.5, 1.5]
                
            print("\nSelect a starting point:")
            for i, x0 in enumerate(x0_options):
                print(f"{i+1}: x0 = {x0}")
            
            x0_choice = input("\nEnter starting point number: ").strip()
            
            try:
                idx = int(x0_choice) - 1
                if 0 <= idx < len(x0_options):
                    x0 = x0_options[idx]
                    animate_newton(f, fprime, x0, 25, 1e-12, fname)
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Please enter a valid number.")
                
        elif method_choice == '3':  # Secant
            # Select appropriate starting points based on function
            if function_choice == '1':
                point_options = [(0, 1), (1, 3)]
            elif function_choice == '2':
                point_options = [(-2, -1), (0, 2)]
            else:
                point_options = [(0, 0.5), (0.5, 1.5)]
                
            print("\nSelect starting points:")
            for i, (x0, x1) in enumerate(point_options):
                print(f"{i+1}: x0 = {x0}, x1 = {x1}")
            
            points_choice = input("\nEnter choice number: ").strip()
            
            try:
                idx = int(points_choice) - 1
                if 0 <= idx < len(point_options):
                    x0, x1 = point_options[idx]
                    animate_secant(f, x0, x1, 25, 1e-12, fname)
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Ask if user wants to continue
        continue_choice = input("\nWould you like to see another animation? (y/n): ").strip().lower()
        if continue_choice != 'y':
            print("Returning to main program...")
            break

if __name__ == "__main__":
    # If run directly, this will demonstrate the animation functions
    from main import f1, f2, f3, f1_prime, f2_prime, f3_prime
    manage_animations(f1, f2, f3, f1_prime, f2_prime, f3_prime)
