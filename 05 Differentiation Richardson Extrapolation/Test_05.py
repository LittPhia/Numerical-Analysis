import numpy as np

def richardson_table(f, x, h, M):
    ''' f must be callable. return the trianglar table of function f '''
    
    table = [[]]
    for i in range(M + 1):
        D_i0 = (f(x + h/2**i) - f(x - h/2**i)) / (2 * h / 2**i)
        table[0].append(D_i0)
        
    for j in range(1, M + 1):
        table.append([])
        for i in range(1, M - j + 2):
            D_ij = 4**j * table[j - 1][i] / (4**j - 1) - table[j - 1][i - 1] / (4**j - 1)
            table[j].append(D_ij)
            
    return table


def print_table(table):
    for i in range(len(table)):
        print('D(*, %d) = ' %i, end = '[')
        for j in range(len(table[i]) - 1):
            print('%.12f,' %table[i][j], end = ' ')
        print('%.12f]' % table[i][len(table[i]) - 1])
        
def g(x):
    return np.sin(x * x + x/ 3)

t = richardson_table(np.log, 3, 1, 3)
print('log(x), x = 3, h = 1, M = 3')
print_table(t)
print('error = %.24f\n' % np.abs(1/3 - t[-1][-1]))

t = richardson_table(np.tan, np.arcsin(0.8), 1, 4)
print('tan(x), x = arcsin 0.8, h = 1, M = 4')
print_table(t)
print('error = %.24f\n' % np.abs(25/9 - t[-1][-1]))

t = richardson_table(g, 0, 1, 5)
print('sin(x^2 + x / 3), x = 0, h = 1, M = 5')
print_table(t)
print('error = %.24f\n' % np.abs(1/3 - t[-1][-1]))

t = richardson_table(np.tan, np.arcsin(0.8), 0.25, 4)
print('tan(x), x = arcsin 0.8, h = 1, M = 4')
print_table(t)
print('error = %.24f\n' % np.abs(25/9 - t[-1][-1]))
