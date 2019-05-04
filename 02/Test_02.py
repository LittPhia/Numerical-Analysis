import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def Newton(f, Xs):
    def N(x):
        n = len(Xs)
        y = [f(Xs[i]) for i in range(n)]
        for i in range(1, n):
            for j in range(n - 1, i - 1, -1):
                y[j] = (y[j] - y[j-1]) / (Xs[j] - Xs[j-i])
        fx = y[n - 1]
        for i in range(n - 1, 0, -1):
            fx = y[i-1] + (x - Xs[i-1]) * fx;
        return fx
    return N

def f(x):
    return 1/(1 + x**2)

def EquiDistantPoints(lower, upper, N):
    return [upper - i * (upper - lower)/N for i in range(N + 1)]
    
def ChebyshevPoints(lower, upper, N):
    return [(lower - upper) * np.cos((2*i + 1)/(2*N + 2) * np.pi) / 2 for i in range(N + 1)]


continuousX = np.arange(-5, 5.01, 0.01)
discontinuousX = np.arange(-5, 5.1, 0.1)
cx = continuousX
dx = discontinuousX
fig = plt.figure()
plt.xlim(-5, 5)

Nrange = (5, 10, 20, 40)
for i in range(len(Nrange)):
    print('N = %d' %Nrange[i])
    subfigc = fig.add_subplot(241 + i)
    subfigd = fig.add_subplot(245 + i)
    subfigc.set_title('N = %d' %Nrange[i])
    
    subfigc.plot(cx, f(cx), c = 'r')
    subfigd.scatter(dx, f(dx), c = 'r', marker = '.')

    p1 = Newton(f, EquiDistantPoints(-5, 5, Nrange[i]))
    subfigc.plot(cx, p1(cx), c = 'y')
    subfigd.scatter(dx, p1(dx), c = 'y', marker = '.')
    
    m1 = max([abs(f(x) - p1(x)) for x in dx])
    print('Max Error of grid (1) : %.12f' %m1)
        
    p2 = Newton(f, ChebyshevPoints(-5, 5, Nrange[i]))
    subfigc.plot(cx, p2(cx), c = 'b')
    subfigd.scatter(dx, p2(dx), c = 'b', marker = '.')
    
    m2 = max([abs(f(x) - p2(x)) for x in dx])
    print('Max Error of grid (2) : %.12f' %m2)
    
    subfigc.legend(('original','grid (1)','grid (2)'))
    subfigd.legend(('original','grid (1)','grid (2)'))

plt.show()