import numpy as np
import sympy as sp

################## Trapezoid method ##################
def Trapezoid(f, lower_bound, upper_bound, k):
    h = (upper_bound - lower_bound) / (2**k)
    s = (f(lower_bound) + f(upper_bound))/2
    
    for i in range(1, 2**k):
        s += f(lower_bound + i * h)
        
    return s * h


################### Simpson method ###################
def Simpson(f, lower_bound, upper_bound, k):
    h = (upper_bound - lower_bound) / (2**(k-1))
    s = f(lower_bound) + f(upper_bound)
    
    tmp = 0
    for i in range(1, 2**(k-1)):
        tmp += f(lower_bound + i * h)
    s += 2 * tmp
    
    tmp = 0
    for i in range(0, 2**(k-1)):
        tmp += f(lower_bound + i * h + h / 2)
    s += 4 * tmp

    return s * h / 6


######################## main ########################
x = sp.Symbol('x')
real = np.float64(sp.integrate(sp.sin(x), (x, 0, 4)))
print('Integrate[sin(x), 0, 4] = %.12f\n' % real)

print('Composite Trapezoid Method:')
error_old = 0.0
for k in range(0, 12):
    approx = Trapezoid(np.sin, 0, 4, k+1)
    error_now = np.abs(real - approx)
    if(k == 0):
        order = 0
    else:
        order = np.log(error_old / error_now) / np.log(2)
    print('I(k = %d) = %.12f \t Error = %.12f \t Order = %.4f' %(k+1, approx, error_now, order))
    error_old = error_now
    
print('Composite Simpson Method:')
error_old = 0.0
for k in range(0, 12):
    approx = Simpson(np.sin, 0, 4, k+1)
    error_now = np.abs(real - approx)
    if(k == 0):
        order = 0
    else:
        order = np.log(error_old / error_now) / np.log(2)
    print('I(k = %d) = %.12f \t Error = %.12f \t Order = %.4f' %(k+1, approx, error_now, order))
    error_old = error_now
