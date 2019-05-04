import numpy as np
import bisect
import matplotlib
import matplotlib.pyplot as plt

class Spline3:
    def __init__(self, f, Xs, d = 2, leftVal = 0, rightVal = 0):
        if d == 2:
            n = len(Xs) - 1
            Ys = f(np.array(Xs))
            h = [Xs[i + 1] - Xs[i] for i in range(n)]
            b = [6 * (Ys[i + 1] - Ys[i]) / h[i] for i in range(n)]
            
            u = [2 * (h[i] + h[i + 1]) for i in range(n - 1)]
            V = np.array([b[i + 1] - b[i] for i in range(n - 1)])
            V[0] -= h[0] * leftVal
            V[n - 2] -=  h[n - 1] * rightVal
            
            A = np.diag(u)
            for i in range(1, A.shape[0]):
                A[i, i - 1] = h[i]
                A[i - 1, i] = h[i]
            
            Z = np.linalg.solve(A, V)
            Z = np.insert(Z, 0, values = 0)
            Z = np.insert(Z, n, values = 0)
            
        elif d == 1:
            n = len(Xs) - 1
            Ys = f(np.array(Xs))
            h = [Xs[i + 1] - Xs[i] for i in range(n)]
            
            b = [6 * (Ys[i + 1] - Ys[i]) / h[i] for i in range(n)]
            b.insert(0, 6 * leftVal)
            b.insert(n + 1, 6 * rightVal)
            
            u = [2 * (h[i] + h[i + 1]) for i in range(n - 1)]
            u.insert(0, 2 * h[0])
            u.insert(n, 2 * h[n - 1])
            
            V = np.array([b[i + 1] - b[i] for i in range(n + 1)])        
            A = np.diag(u)
            for i in range(A.shape[0] - 1):
                A[i, i + 1] = h[i]
                A[i + 1, i] = h[i]

            Z = np.linalg.solve(A, V)

        self.t = tuple(Xs)
        self.y = tuple(Ys)
        self.h = tuple(h)
        self.z = tuple(Z)
        self.A = [(Z[i + 1] - Z[i]) / (6 * h[i]) for i in range(n)]
        self.B = [Z[i] / 2  for i in range(n)]
        self.C = [(Ys[i + 1] - Ys[i]) / h[i] - (Z[i + 1] / 6 + Z[i] / 3) * h[i] for i in range(n)]
        
    def get(self, x):
        i = bisect.bisect(self.t, x)
        if i == len(self.t):
            i -= 2
        elif i == len(self.t) - 1:
            i -= 1
        else:
            pass
        
        s = self.B[i] + (x - self.t[i]) * self.A[i]
        s = self.C[i] + (x - self.t[i]) * s
        return self.y[i] + (x - self.t[i]) * s
    
    def integrate(self):
        s = 0
        for i in range(len(self.t) - 1):
            s += (self.y[i] + self.y[i + 1]) * self.h[i] / 2 - (self.z[i] + self.z[i + 1]) * self.h[i]**3 / 24
        return s
            
        
def f(x):
    return np.e**x


left = 0
right = 1
continuousX = np.arange(left, right + 0.001, 0.001)
cx = continuousX

Nrange = (10, 20, 40, 80)
fig = plt.figure()

error_new_1 = 0
error_old_1 = 0
error_new_2 = 0
error_old_2 = 0

for i in range(len(Nrange)):
    print('N = %d' %Nrange[i])
    discontinuousX = np.arange(left, right + (right - left) / Nrange[i], (right - left) / Nrange[i])
    dx = discontinuousX
    
    subfigc = fig.add_subplot(241 + i)
    subfigd = fig.add_subplot(245 + i)
    subfigc.set_title('N = %d' %Nrange[i])
    
    subfigc.plot(cx, f(cx), c = 'r')
    subfigd.scatter(dx, f(dx), c = 'r', marker = '.')
    
    
    p1 = Spline3(f, dx, d = 2, leftVal = 0, rightVal = 0)
    subfigc.plot(cx, np.array([p1.get(x) for x in cx]), c = 'y')
    subfigd.scatter(dx, np.array([p1.get(x) for x in dx]), c = 'y', marker = '.')
    
    error_new_1 = max([abs(f(x) - p1.get(x)) for x in np.delete(dx, 0) - 0.5 / Nrange[i]])
    print('Method (1) Error = %.12f' %error_new_1, end = '\t')
    if i > 0:
        print('Order = %.12f' %(np.log(error_old_1 / error_new_1) / np.log(Nrange[i] / Nrange[i - 1])))
    else:
        print()
    error_old_1 = error_new_1
    
    print('Integrate[S1(x), {t_n, t_0}] = %.12f' %p1.integrate())
    
    p2 = Spline3(f, dx, d = 1, leftVal = 1, rightVal = np.e)
    subfigc.plot(cx, np.array([p2.get(x) for x in cx]), c = 'b')
    subfigd.scatter(dx, np.array([p2.get(x) for x in dx]), c = 'b', marker = '.')
    
    error_new_2 = max([abs(f(x) - p2.get(x)) for x in np.delete(dx, 0) - 0.5 / Nrange[i]])
    print('Method (2) Error = %.12f' %error_new_2, end = '\t')
    if i > 0:
        print('Order = %.12f' %(np.log(error_old_2 / error_new_2) / np.log(Nrange[i] / Nrange[i - 1])))
    else:
        print()
    error_old_2 = error_new_2
    
    print('Integrate[S2(x), {t_n, t_0}] = %.12f' %p2.integrate())
    print(end = '\n')
        
    subfigc.legend(('original','method (1)','method (2)'))
    subfigd.legend(('original','method (1)','method (2)'))

plt.show()