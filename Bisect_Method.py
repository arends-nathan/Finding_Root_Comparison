import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def sign(a) -> int:
    if a < 0:
        return -1
    else:
        return 1

def bisect_method(a, b, M, err1, epsilon, f):
    fa: float = f(a)
    fb: float = f(b)
    e: float = b - a
    print(f"a: {a}\nb: {b}\nf(a): {fa}\nf(b): {fb}")
    
    if sign(fa) == sign(fb):
        return "Same Sign"

    for i in range(1, M):
        e = e/2
        c = a + e
        fc = f(c)
        print (f"i: {i}\nc: {c}\nf(c): {fc}\ne: {e}")
        
        if abs(e) < err1 or abs(fc) < epsilon:
            return f"Root found at ({c},{fc})"

        if sign(fc) != sign(f(a)):
            b = c
            fb = fc     
        else:
            a = c
            fa = c

def bisect_method_with_plot(a, b, M, err1, epsilon, f):
    fa = f(a)
    fb = f(b)
    e = b - a
    x_vals = []
    fx_vals = []
    
    print(f"a: {a}\nb: {b}\nf(a): {fa}\nf(b): {fb}")
    
    if sign(fa) == sign(fb):
        return "Same Sign"

    for i in range(1, M + 1):
        e = e / 2
        c = a + e
        fc = f(c)
        x_vals.append(c)
        fx_vals.append(fc)
        print(f"i: {i}\nc: {c}\nf(c): {fc}\ne: {e}")
        
        if abs(e) < err1 or abs(fc) < epsilon:
            print("Convergence achieved.")
            break

        if sign(fc) != sign(f(a)):
            b = c
            fb = fc     
        else:
            a = c
            fa = fc
    
    return x_vals, fx_vals

def bisect_animation(a, b, M, epsilon, f):
    fa = f(a)
    fb = f(b)
    e = b - a
    x_vals = []
    fx_vals = []
    intervals = []
    
    if sign(fa) == sign(fb):
        return "Same Sign"

    for i in range(1, M + 1):
        e = e / 2
        c = a + e
        fc = f(c)
        x_vals.append(c)
        fx_vals.append(fc)
        intervals.append((a, b))
        
        if abs(e) < epsilon or abs(fc) < epsilon:
            break

        if sign(fc) != sign(f(a)):
            b = c
            fb = fc     
        else:
            a = c
            fa = fc
    
    return x_vals, fx_vals, intervals

