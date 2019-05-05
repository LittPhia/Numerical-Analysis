import numpy as np
import matplotlib.pyplot as plt


class InitialValueProblem:
    def __init__(self, f, x0, y0):
        self.f = f
        self.x0 = x0
        self.y0 = y0
        self.nsol = None
    
    def RungeKuttaMethod(self, h, x_to_where):
        f = self.f
        x = self.x0
        y = self.y0
        self.nsol = [[x], [y]]
        
        # Since it is not an adaptive algorithm, we simply use h
        h = np.sign(x_to_where - x) * abs(h)
        
        while(True):
            if x == x_to_where:
                print('Runge-Kutta Method ended successfully')
                return
            
            if abs(x_to_where - x) < abs(h):
                h = x_to_where - x
                
            try:
                F1 = h * f(x, y)
                F2 = h * f(x + h / 2, y + F1 / 2)
                F3 = h * f(x + h / 2, y + F2 / 2)
                F4 = h * f(x + h, y + F3)

                y += (F1 + 2 * F2 + 2 * F3 + F4) / 6
                x += h
                
            except Exception as e:
                print(e)
                print('Runge-Kutta Method ended forcefully after point (%.16f, %.16f)' %(self.nsol[0][-1], self.nsol[1][-1]))
                return
            
            else:
                self.nsol[0].append(x)
                self.nsol[1].append(y)


lower = 0
upper = 5
h = 0.01
while(True):
    try:
        Lambda = np.float64(input('Input lambda = '))
    except Exception as e:
        print(e)
        continue

    def f(x, y):
        return Lambda * y + np.cos(x)  - Lambda * np.sin(x)
    def y(x):
        return np.sin(x)

    problem = InitialValueProblem(f, lower, 0)
    print('Solving Equation...')
    problem.RungeKuttaMethod(h, upper)

    max_error = 0
    maxe_point = lower
    Y = []
    for i in range(len(problem.nsol[0])):
        Y.append(y(problem.nsol[0][i]))
        error = abs(Y[-1] - problem.nsol[1][i])
        if error > max_error:
            maxe_point = problem.nsol[0][i]
            max_error = error
    print('Max Error = %.12E\n'
          'Reached at %.12f' % (max_error, maxe_point))

    plt.plot(problem.nsol[0], problem.nsol[1], c = 'r')
    plt.plot(problem.nsol[0], Y, c = 'b')
    plt.title('lambda = %f' %Lambda)
    plt.legend(('Numerical', 'Analytical'))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()
