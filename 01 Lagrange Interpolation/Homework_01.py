import numpy
import matplotlib
import matplotlib.pyplot

def Lagrange(f, Xs):
    def p(x):
        y = 0.0
        for i in range(len(Xs)):
            lt = 1.0
            for j in range(i):
                lt *= (x - Xs[j])/(Xs[i] - Xs[j])
            for j in range(i + 1, len(Xs)):
                lt *= (x - Xs[j])/(Xs[i] - Xs[j])
            y += f(Xs[i]) * lt
        return y
    return p

def f(x):
    return 1/(1 + x**2)

def EquiDistantPoints(lower, upper, N):
    return [upper - i * (upper - lower)/N for i in range(N + 1)]
    
def ChebyshevPoints(lower, upper, N):
    return [(lower - upper) * numpy.cos((2*i + 1)/(2*N + 2) * numpy.pi) / 2 for i in range(N + 1)]


continuousX = numpy.arange(-5, 5.01, 0.01)
discontinuousX = numpy.arange(-5, 5.1, 0.1)
cx = continuousX
dx = discontinuousX
fig = matplotlib.pyplot.figure()
matplotlib.pyplot.xlim(-5, 5)

Nrange = (5, 10, 20, 40)
for i in range(len(Nrange)):
    print('N = %d' %Nrange[i])
    subfigc = fig.add_subplot(241 + i)
    subfigd = fig.add_subplot(245 + i)
    subfigc.set_title('N = %d' %Nrange[i])
    
    subfigc.plot(cx, f(cx), c = 'r')
    subfigd.scatter(dx, f(dx), c = 'r', marker = '.')

    p1 = Lagrange(f, EquiDistantPoints(-5, 5, Nrange[i]))
    subfigc.plot(cx, p1(cx), c = 'y')
    subfigd.scatter(dx, p1(dx), c = 'y', marker = '.')
    
    m1 = max([abs(f(x) - p1(x)) for x in dx])
    print('Max Error of grid (1) : %.12f' %m1)
      
    p2 = Lagrange(f, ChebyshevPoints(-5, 5, Nrange[i]))
    subfigc.plot(cx, p2(cx), c = 'b')
    subfigd.scatter(dx, p2(dx), c = 'b', marker = '.')
    
    m2 = max([abs(f(x) - p2(x)) for x in dx])
    print('Max Error of grid (2) : %.12f' %m2)
    
    subfigc.legend(('original','grid (1)','grid (2)'))
    subfigd.legend(('original','grid (1)','grid (2)'))

matplotlib.pyplot.show()