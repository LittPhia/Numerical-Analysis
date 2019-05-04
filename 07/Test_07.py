import numpy as np
import scipy.integrate as spint

def fa(x):
    if x == 0: return 1
    else: return np.sin(x) / x

def fb(x):
    if x == 0: return -1
    return (np.cos(x) - np.e**x) / np.sin(x)

def fc(t):
    if t == 0: return 0
    return  np.e**(-1/t) / t

################## Romberg method ##################
def romberg_table(f, lower_bound, upper_bound, M):
    ''' f must be calllable '''
    
    R = [[(upper_bound - lower_bound) * (f(lower_bound) + f(upper_bound)) / 2]]
    for n in range(M):
        s = 0
        h = (upper_bound - lower_bound) / 2**(n+1)
        for i in range(2**n):
            s += f(lower_bound + (2 * i + 1) * h)
        R.append([R[n][0] / 2 + s * h])
    
    for m in range(1, M + 1):
        for n in range(m, M + 1):
            R_nm = 4**m * R[n][m - 1] / (4**m - 1) - R[n - 1][m - 1] / (4**m - 1)
            R[n].append(R_nm)
    return R


def print_table(table):
    print(' n | ', end = '')
    for i in range(len(table)):
        print('    R(n, %d)     ' %i, end = '')
    print()
    for i in range(len(table)):
        print(' %d ' %i, end = '|')
        for j in range(len(table[i])):
            print(' %.12f' %table[i][j], end = '')
            if table[i][j] >= 0: print(end = ' ')
        print()
        
def print_error_table(table, real):
    print(' n | ', end = '')
    for i in range(len(table)):
        print('   R(n, %d)    ' %i, end = '')
    print()
    for i in range(len(table)):
        print(' %d ' %i, end = '|')
        for j in range(len(table[i])):
            print(' %e' %np.abs(real - table[i][j]), end = '')
            if table[i][j] >= 0: print(end = ' ')
        print()
        
###################### main ######################
print('Integrate[sin(x) / x, {x, 0, 1}] = %.12f \n'
      'Error of function scipy.integrate.quad %.12f' %spint.quad(fa, 0, 1))
t = romberg_table(fa, 0, 1, 6)
print('Romberg_Integrate[sin(x) / x, {x, 0, 1}] ---- M = 6')
print_table(t)
print()
print('Error of Romberg_Integrate[sin(x) / x, {x, 0, 1}] ---- M = 6')
print_error_table(t, spint.quad(fa, 0, 1)[0])
print('\n-----------------------------------------------------------------------------')


print('Integrate[(cos(x) - e**x) / sin(x), {x, -1, 1}] = %.12f \n'
      'Error of function scipy.integrate.quad %.12f' %spint.quad(fb, -1, 1))
t = romberg_table(fb, -1, 1, 6)
print('Romberg_Integrate[(cos(x) - e**x) / sin(x), {x, -1, 1}] ---- M = 6')
print_table(t)
print()
print('Error of Romberg_Integrate[(cos(x) - e**x) / sin(x), {x, -1, 1}] ---- M = 6')
print_error_table(t, spint.quad(fb, -1, 1)[0])
print('\n-----------------------------------------------------------------------------')


print('Integrate[1 / (x * e**x), {x, 0, Infinity}] = Integrate[e**(-1/t) / t, {x, 0, 1}] = %.12f \n'
      'Error of function scipy.integrate.quad %.12f' %spint.quad(fc, 0, 1))
t = romberg_table(fc, 0, 1, 6)
print('Romberg_Integrate[1 / (x * e**x), {x, 1, Infinity}] = Romberg_Integrate[e**(-1/t) / t, {x, 0, 1}] ---- M = 6')
print_table(t)
print()
print('Error of Romberg_Integrate[e**(-1/t) / t, {x, 0, 1}] ---- M = 6')
print_error_table(t, spint.quad(fc, 0, 1)[0])
print('\n-----------------------------------------------------------------------------')
