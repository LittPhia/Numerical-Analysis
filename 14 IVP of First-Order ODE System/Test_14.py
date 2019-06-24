from InitialValueProblem import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# REQUIED EQUATION
def f(x, y):
    u = np.sin(y[0]) + np.cos(y[1] * x)
    v = None
    if x != 0:
        v = np.e**(-y[0] * x) + np.sin(y[1] * x) / x
    else:
        v =  np.e**(-y[0] * x) + y[1]
    return np.array([u, v])
  
t0 = -1
X, Y = 2.37, -3.48
tto = 4


# GET SOLUTION
print('Adaptive fifth-fourth order Runge-Kutta-Fehlberg.')

problem = InitialValueProblem(f, t0, np.array([X, Y]))
problem.RKF54a(0.001, tto + 0.001, 1e-16)


# PLOT
fig = plt.figure()

Ts = np.arange(t0, tto + 0.1, 0.1)
Xs = np.array([problem.value_at(t)[0] for t in Ts])
Ys = np.array([problem.value_at(t)[1] for t in Ts])

subfig_XYT = fig.add_subplot(221)
subfig_XY = fig.add_subplot(222)

subfig_XYT.grid()
subfig_XYT.set_title('Scattered figure of x-t and y-t', fontsize = 30)
subfig_XYT.scatter(Ts, Xs, c = 'r', marker = '.')
subfig_XYT.scatter(Ts, Ys, c = 'b', marker = '.')
subfig_XYT.legend(('x-t', 'y-t'))

subfig_XY.set_title('Scattered figure of x-y', fontsize = 30)
subfig_XY.scatter(Ys, Xs, c = 'g', marker = '.')
subfig_XY.legend(('y-x'))


Ts = np.arange(t0, tto + 0.002, 0.002)
Xs = np.array([problem.value_at(t)[0] for t in Ts])
Ys = np.array([problem.value_at(t)[1] for t in Ts])

subfig_XYT = fig.add_subplot(223)
subfig_XY = fig.add_subplot(224)

subfig_XYT.grid()
subfig_XYT.set_title('Smooth figure of x-t and y-t', fontsize = 30)
subfig_XYT.plot(Ts, Xs, c = 'r')
subfig_XYT.plot(Ts, Ys, c = 'b')
subfig_XYT.legend(('x-t', 'y-t'))

subfig_XY.set_title('Smooth figure of x-y', fontsize = 30)
subfig_XY.plot(Ys, Xs, c = 'g')
subfig_XY.legend(('y-x'))


plt.show()
