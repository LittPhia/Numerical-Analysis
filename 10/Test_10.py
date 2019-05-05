import numpy as np
import matplotlib.pyplot as plt
import bisect
import warnings
import traceback


class InitialValueProblem:
    def __init__(self, f, x0, y0):
        self.f = f
        self.x0 = x0
        self.y0 = y0
        self.sol = None
    
    def RKF54(self, h0, error_each_step):
        warnings.filterwarnings('error')
        f = self.f
        h = h0
        x = self.x0
        y = self.y0
        self.sol = [[x], [y]]
        
        F1 = h0 * f(x, y)
        F2 = h0 * f(x + h0 / 4, y + F1 / 4)
        F3 = h0 * f(x + h0 * 3/8, y + F1 * 3/32 + F2 * 9/32)
        F4 = h0 * f(x + h0 * 12/13, y + F1 * 1932/2197 - F2 * 7200/2197 + F3 * 7296/2197)
        F5 = h0 * f(x + h0, y + F1 * 439/216 - F2 * 8 + F3 * 3680/513 - F4 * 845/4104)
        F6 = h0 * f(x + h0 / 2, y - F1 * 8/27 + F2 * 2 - F3 * 3544/2565 + F4 * 1859/4104 - F5 * 11/40)
        error = abs(F1 / 360 - F3 * 128/4275 - F4 * 2197/75240 + F5 / 50 + F6 * 2/55)
        h = 0.9 * h0 * np.power(error_each_step / error, 1 / 6)
        # Not update (x, y) here because we do not fully trust h0
        # Instead we just use h0 to calculate the stride of first step

        while(True):
            try:
                F1 = h * f(x, y)
                F2 = h * f(x + h / 4, y + F1 / 4)
                F3 = h * f(x + h * 3/8, y + F1 * 3/32 + F2 * 9/32)
                F4 = h * f(x + h * 12/13, y + F1 * 1932/2197 - F2 * 7200/2197 + F3 * 7296/2197)
                F5 = h * f(x + h, y + F1 * 439/216 - F2 * 8 + F3 * 3680/513 - F4 * 845/4104)
                F6 = h * f(x + h / 2, y - F1 * 8/27 + F2 * 2 - F3 * 3544/2565 + F4 * 1859/4104 - F5 * 11/40)

                y += F1 * 16/135 + F3 * 6656/12825 + F4 * 28561/56430 - F5 * 9/50 + F6 * 2/55
                x += h
                error = abs(F1 / 360 - F3 * 128/4275 - F4 * 2197/75240 + F5 / 50 + F6 * 2/55)
                h = 0.9 * h * np.power(error_each_step / error, 1 / 6)
                if h <= 0:
                    raise Exception
                
            except Exception as e:
                print(traceback.format_exc())

                print('Adaptive RKF54 method ended forcefully after point (%.32f, %.32f)' %(self.sol[0][-1], self.sol[1][-1]))
                return
            
            else:
                self.sol[0].append(x)
                self.sol[1].append(y)

        
    def sol_xupper(self):
        if self.sol is None:
            raise Warning('WARNING: Unsolved Equation')
        else:
            return self.sol[0][-1]
        
        
    def y(self, x):
        if self.sol is None:
            raise Exception('ERROR: Equation Unsolved')
        
        i = bisect.bisect(self.sol[0], x)
        if i == len(self.sol[0]) or x < self.x0:
            raise Exception('ERROR: Sampling point out of range')
            
        elif i == len(self.sol[0]) - 1:
            return self.sol[1][i]
        
        else:
            x1 = self.sol[0][i]
            x2 = self.sol[0][i + 1]
            y1 = self.sol[1][i]
            y2 = self.sol[1][i + 1]
            return (y2 - y1) / (x2 - x1) * (x - x1) + y1


        
def f(x, y):
    return np.e**(y * x) + np.cos(y - x)


problem = InitialValueProblem(f, 1, 3)
print('Solving Equation...')
problem.RKF54(0.01, 1e-16)
print('Solution now reachable in [1, %.12f]\n' % problem.sol_xupper())

warnings.filterwarnings('ignore')
plt.plot(problem.sol[0][0:4096], problem.sol[1][0:4096])
plt.grid()
plt.show()

while(True):
    print('Input sampling point = ', end = '')
    try:
        x = np.float64(input())
        print('y(%.12f) = %.12f' %(x, problem.y(x)))
    except Exception as e:
        print(e)
