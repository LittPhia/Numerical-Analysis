# Adaptive Simpson Integration Method
import numpy as np

def adaptive_simpson(f, lower, upper, error, max_depth = 0x1000):
    ''' f must be callable, max_depth represent the max stack size'''
    
    def simpson_calculate(fl, fm, fr, half_ltor_len):
        return (fl + 4 * fm + fr) * half_ltor_len / 3
    
    interval_len = upper - lower
    INTEGRATE = 0
    STACK = []
    
    h = interval_len / 2
    mid = lower + h
    V = [lower, h, f(lower), f(mid), f(mid + h)]
    V.append(simpson_calculate(V[2], V[3], V[4], h))
    STACK.append(V)
    
    STACKSIZE = len(STACK)
    while(STACKSIZE != 0):
        if STACKSIZE > max_depth:
            print('WARNING: STACK MAX DEPTH REACHED')
            return INTEGRATE
        
        cur = STACKSIZE - 1
        while(cur >= 0):
            V = STACK[cur]
            left = V[0]
            mid = left + V[1]
            h = V[1] / 2
            fm1 = f(left + h)
            fm2 = f(mid + h)
            S1 = simpson_calculate(V[2], fm1, V[3], h)
            S2 = simpson_calculate(V[3], fm2, V[4], h)

            if abs(S1 + S2 - V[-1]) > 60 * error * h / interval_len:
                STACK.append([left, h, V[2], fm1, V[3], S1])
                STACK.append([mid, h, V[3], fm2, V[4], S2])
            else:
                INTEGRATE += (16 * (S1 + S2) - V[-1]) / 15
                
            STACK.pop(cur) 
            cur -= 1
            
        STACKSIZE = len(STACK)
    return INTEGRATE

        
def f(x):
    return np.e**x / (2 + np.log(0.1 + x) + np.cos(x))

print('%.16f' % adaptive_simpson(f, 0, 2, 1e-16, 0x100000))
