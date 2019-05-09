import scipy as sp
import scipy.optimize


class InitialValueProblem:
    class Solution:
        def __init__(self):
            self.sol = []
            
        def X(self, index):
            return self.sol[index][0]
        
        def Y(self, index):
            return self.sol[index][1]
        
        def append(self, xt, yt):
            self.sol.append((xt, yt))
            
        def size(self):
            return len(self.sol)
        
            
    def __init__(self, f, x0, y0):
        self.f = f
        self.x0 = x0
        self.y0 = y0
        self.sol = InitialValueProblem.Solution()
    
    def RungeKuttaMethod(self, h, xto):
        f = self.f
        x = None
        y = None
        if self.sol.size() == 0:
            x = self.x0
            y = self.y0
            self.sol.append(x, y)
        else:
            x = self.sol.X(-1)
            y = self.sol.Y(-1)
        
        h = sp.sign(xto - x) * abs(h)
        
        while(sp.sign(xto - x) == sp.sign(h)):
            if abs(xto - x) < abs(h): h = xto - x
                
            F1 = h * f(x, y)
            F2 = h * f(x + h / 4, y + F1 / 4)
            F3 = h * f(x + h * 3/8, y + F1 * 3/32 + F2 * 9/32)
            F4 = h * f(x + h * 12/13, y + F1 * 1932/2197 - F2 * 7200/2197 + F3 * 7296/2197)
            F5 = h * f(x + h, y + F1 * 439/216 - F2 * 8 + F3 * 3680/513 - F4 * 845/4104)
            F6 = h * f(x + h / 2, y - F1 * 8/27 + F2 * 2 - F3 * 3544/2565 + F4 * 1859/4104 - F5 * 11/40)

            y += F1 * 16/135 + F3 * 6656/12825 + F4 * 28561/56430 - F5 * 9/50 + F6 * 2/55
            x += h
            
            self.sol.append(x, y)
            
        return self.sol.Y(-1)
    
    def AdamsBashforthMethod(self, h, xto):
        f = self.f
        x = None
        if self.sol.size() == 0:
            x = self.x0
            self.sol.append(x, self.y0)
        else:
            x = self.sol.X(-1)
        
        h = sp.sign(xto - x) * abs(h)
                
        self.RungeKuttaMethod(h, x + 4 * h)
        x = self.sol.X(-1)
        
        while(sp.sign(xto - x) == sp.sign(h)):
            if abs(xto - x) < abs(h): h = xto - x
                    
            y = self.sol.Y(-1) + h * (
                    1901 * f(self.sol.X(-1), self.sol.Y(-1))
                    - 2774 * f(self.sol.X(-2), self.sol.Y(-2))
                    + 2616 * f(self.sol.X(-3), self.sol.Y(-3))
                    - 1274 * f(self.sol.X(-4), self.sol.Y(-4))
                    + 251 * f(self.sol.X(-5), self.sol.Y(-5))
                ) / 720
            x += h
            
            self.sol.append(x, y)
            
        return self.sol.Y(-1)
    
    def last_point(self):
        return (self.sol.X(-1), self.sol.Y(-1))

    
    
# REQUIED EQUATION
def y1(x):
    def eqn(x, y):
        return y**2 - x**2 + 2 * sp.e**y - 2 * sp.e**(-x)
    return sp.optimize.fsolve(lambda y : eqn(x, y) - eqn(x0, y0), y0 - 1)[0]

def f1(x, y):
    return (x - sp.e**(-x)) / (y + sp.e**y)


# ANOTHER FUCTION
def y2(x):
    return sp.sin(x)
    
def f2(x, y):
    return y + sp.cos(x) - sp.sin(x)

    
# PRINT APPROX-ERROR-ORDER LIST
real = None

def solve_then_print_aeolist(f, x0, y0, xto):
    print('Adams-Bashforth :')
    error_old = 0
    for k in range(3, 9):
        problem = InitialValueProblem(f, x0, y0)
        problem.AdamsBashforthMethod((xto - x0) / 2**k, xto)
        x, y = problem.last_point()

        error_new = abs(real - y)
        if error_new == 0:
            print('[k = %d]  y(%f) = %.12f, Error = %.12e '
                  '// Max precision reached' % (k, xto, y, error_new))
        else:
            order = sp.log(error_old / error_new) / sp.log(2)
            print('[k = %d]  y(%f) = %.12f, Error = %.12e, Order = %.4f' % (k, xto, y, error_new, order))

        error_old = error_new
    
    print()
    
    
# INITIALIZATION
x0 = 0
y0 = 0
xto = 1
real = y1(xto)
print('\nEQN:\n'
      '  f(x, y) = (x - e^(-x)) / (y + e^y)\n'
      '  y(0) = 0\n'
     'Real : y1(%f) = %.12f\n' % (xto, real))

solve_then_print_aeolist(f1, x0, y0, xto)

x0 = 0
y0 = 1
xto = 1
real = y1(xto)
print('\nEQN:\n'
      '  f(x, y) = (x - e^(-x)) / (y + e^y)\n'
      '  y(0) = 0\n'
     'Real : y1(%f) = %.12f\n' % (xto, real))

solve_then_print_aeolist(f1, x0, y0, xto)

x0 = 0
y0 = 0
xto = 1
real = y2(xto)
print('\nEQN:\n'
      '  f(x, y) = y + cos(x) - sin(x)\n'
      '  y(0) = 0\n'
      'Real : y2(%f) = %.12f\n' % (xto, real))

solve_then_print_aeolist(f2, x0, y0, xto)
