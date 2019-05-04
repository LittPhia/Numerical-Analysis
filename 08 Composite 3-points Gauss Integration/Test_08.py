import mpmath
import numpy

mpmath.mp.prec = 512

def f1(x):
    return mpmath.e**(- x * x)

def f2(x):
    return 1 / (1 + x * x)

def f3(x):
    return 1 / (2 + mpmath.cos(x))


################## Composite Trapezoid method ##################
def CTrapezoid(f, lower_bound, upper_bound, k):
    h = (upper_bound - lower_bound) / (2**k)
    s = (f(lower_bound) + f(upper_bound)) / 2

    for i in range(1, 2**k):
        s += f(lower_bound + i * h)

    return s * h

#################### Composite Gauss method ####################
def CGauss(f, lower_bound, upper_bound, k):
    h = (upper_bound - lower_bound) / (2**k)
    r = 0
    t0 = (1 - mpmath.sqrt(3/5)) * h / 2
    t1 = h / 2
    t2 = (1 + mpmath.sqrt(3/5)) * h / 2
    
    s = 0
    for i in range(2**k):
        s += f(lower_bound + i * h + t0)
    r += 5 * s
    
    s = 0
    for i in range(2**k):
        s += f(lower_bound + i * h + t1)
    r += 8 * s
    
    s = 0    
    for i in range(2**k):
        s += f(lower_bound + i * h + t2)
    r += 5 * s

    return r * h / 18

############################# main #############################

def print_info(integrate_method, func, lower, upper, M, real_val):
    error_old = 0
    for k in range(M):
        approx = integrate_method(func, lower, upper, k + 1)
        error_now = abs(real_val - approx)
        if(k == 0):
            order = 0
        else:
            order = mpmath.log(error_old / error_now) / mpmath.log(2)
        print('I(k = %d) = %.16f \t Error = %.12e \t Order = %.4f' %(k + 1, approx, error_now, order))
        error_old = error_now

    print()
    
######################## f1 ########################
lower = mpmath.mpf(0)
upper = mpmath.mpf(1)
result = mpmath.quad(f1, (lower, upper))

print('Integrate[E^(- x * x), {x, %f, %f}] = %.16f' % (lower, upper, result))
print('Composite Trapezoid method:')
print_info(CTrapezoid, f1, lower, upper, 7, result)
print('Composite Gauss method:')
print_info(CGauss, f1, lower, upper, 7, result)

######################## f2 ########################
lower = mpmath.mpf(0)
upper = mpmath.mpf(4)
result = mpmath.quad(f2, (lower, upper))

print('Integrate[1 / (1 + x * x), {x, %f, %f}] = %.16f' % (lower, upper, result))
print('Composite Trapezoid method:')
print_info(CTrapezoid, f2, lower, upper, 7, result)
print('Composite Gauss method:')
print_info(CGauss, f2, lower, upper, 7, result)

######################## f3 ########################
lower = mpmath.mpf(0)
upper = mpmath.mpf(2 * mpmath.pi)
result = mpmath.quad(f3, (lower, upper))

print('Integrate[1 / (2 + Cos[x]), {x, %f, %f}] = %.16f' % (lower, upper, result))
print('Composite Trapezoid method:')
print_info(CTrapezoid, f3, lower, upper, 7, result)
print('Composite Gauss method:')
print_info(CGauss, f3, lower, upper, 7, result)



####################### TESTS AREA #######################
mpmath.mp.prec = 2048
M = 18
print('\n####################### tests #######################\n')

def test(f_test, a, b):
    result = mpmath.quad(f_test, (a, b))
    print('Test Integrate = %.16f' %result)
    print('Composite Trapezoid method:')
    print_info(CTrapezoid, f_test, a, b, M, result)
    print('Composite Gauss method:')
    print_info(CGauss, f_test, a, b, M, result)

    
print('on x^8, 0 -> 1')
def ft(x):
    return x**8
test(ft, mpmath.mpf(0), mpmath.mpf(1))

print('on e^x, 0 -> 1')
def ft(x):
    return mpmath.e**x
test(ft, mpmath.mpf(0), mpmath.mpf(1))

print('on sin(x), 0 -> 1')
def ft(x):
    return mpmath.sin(x)
test(ft, mpmath.mpf(0), mpmath.mpf(1))

print('on arcsin(x), 0 -> 0.9')
def ft(x):
    return mpmath.asin(x)
test(ft, mpmath.mpf(0), mpmath.mpf(0.9))

print('on 1 / (1 + 2cos(x)), 0 -> 2 * numpy.pi')
test(f3, mpmath.mpf(0), 2 * numpy.pi)

print('on 1 / (1 + 2cos(x)), 0 -> 2 * mpmath.pi')
test(f3, mpmath.mpf(0), 2 * mpmath.pi)
