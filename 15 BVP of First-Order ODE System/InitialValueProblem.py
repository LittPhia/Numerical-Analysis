import numpy as np
import bisect

class InitialValueProblem:
    '''
        Current version 1.0.2.
        This class is designed for solving n-dim differential equations with the form: 
            y'(x) = f(x, y)
            y(x0) = y0
        x0 should be an n-elements np.array or just single element (when n == 1). y0 is a single element.
        y(x) and f(x, y) are supposed to be functions which return 1-dim value and compatible with x == x0.
        
        Implemented methods:
            Runge-Kutta 4
            Runge-Kutta 5
            Runge-Kutta-Fehlberg 54 (Adaptive)
            Adams-Bashforth 5
            Adams-Bashforth-Moulton 5
            
        Litt Phia, last edited at 6/21/2019.
    '''
    
    class Solution:
        def __init__(self):
            self.sol_ = []
            
        def X(self, index):
            return self.sol_[index][0]
        
        def Y(self, index):
            return self.sol_[index][1]
        
        def Xs(self):
            return [xy[0] for xy in self.sol_]
        
        def Ys(self):
            return [xy[1] for xy in self.sol_]
        
        def append(self, xt, yt):
            self.sol_.append((xt, yt))
            
        def size(self):
            return len(self.sol_)
        
        
    def __init__(self, f, x0, y0):
        self.f = f
        self.x0 = x0
        self.y0 = y0
        self.sol = InitialValueProblem.Solution()
    
    
    def clear(self):
        del self.sol
        self.sol = InitialValueProblem.Solution()

        
    def RK4(self, h, xto):
        ''' Fourth order Runge-Kutta method '''
        f = self.f
        x = None
        y = None
        if self.sol.size() == 0:
            x = self.x0
            y = np.array(self.y0)
            self.sol.append(x, np.array(y))
        else:
            x = self.sol.X(-1)
            y = np.array(self.sol.Y(-1))
                
        h = np.sign(xto - x) * abs(h)

        while(np.sign(xto - x) == np.sign(h)):
            if abs(xto - x) < abs(h): h = xto - x
            if h == 0: return
            
            F1 = h * f(x, y)
            F2 = h * f(x + h / 2, y + F1 / 2)
            F3 = h * f(x + h / 2, y + F2 / 2)
            F4 = h * f(x + h, y + F3)

            y += (F1 + 2 * F2 + 2 * F3 + F4) / 6
            x += h
            self.sol.append(x, np.array(y))

    
    def RK5(self, h, xto):
        ''' Fifth order Runge-Kutta method '''
        f = self.f
        x = None
        y = None
        if self.sol.size() == 0:
            x = self.x0
            y = np.array(self.y0)
            self.sol.append(x, np.array(y))
        else:
            x = self.sol.X(-1)
            y = np.array(self.sol.Y(-1))
        
        h = np.sign(xto - x) * abs(h)
        
        while(np.sign(xto - x) == np.sign(h)):
            if abs(xto - x) < abs(h): h = xto - x
            if h == 0: return
            
            F1 = h * f(x, y)
            F2 = h * f(x + h / 4, y + F1 / 4)
            F3 = h * f(x + h * 3/8, y + F1 * 3/32 + F2 * 9/32)
            F4 = h * f(x + h * 12/13, y + F1 * 1932/2197 - F2 * 7200/2197 + F3 * 7296/2197)
            F5 = h * f(x + h, y + F1 * 439/216 - F2 * 8 + F3 * 3680/513 - F4 * 845/4104)
            F6 = h * f(x + h / 2, y - F1 * 8/27 + F2 * 2 - F3 * 3544/2565 + F4 * 1859/4104 - F5 * 11/40)

            y += F1 * 16/135 + F3 * 6656/12825 + F4 * 28561/56430 - F5 * 9/50 + F6 * 2/55
            x += h
            self.sol.append(x, np.array(y))
                 
      
    def RKF54a(self, h0, xto, e):
        ''' Adaptive fifth-fourth order Runge-Kutta-Fehlberg method '''
        f = self.f
        x = None
        y = None
        if self.sol.size() == 0:
            x = self.x0
            y = np.array(self.y0)
            self.sol.append(x, np.array(y))
        else:
            x = self.sol.X(-1)
            y = np.array(self.sol.Y(-1))
        
        h = np.sign(xto - x) * abs(h0)

        while(np.sign(xto - x) == np.sign(h)):
            if abs(xto - x) < abs(h): h = xto - x
            if h == 0: return
            
            F1 = h * f(x, y)
            F2 = h * f(x + h / 4, y + F1 / 4)
            F3 = h * f(x + h * 3/8, y + F1 * 3/32 + F2 * 9/32)
            F4 = h * f(x + h * 12/13, y + F1 * 1932/2197 - F2 * 7200/2197 + F3 * 7296/2197)
            F5 = h * f(x + h, y + F1 * 439/216 - F2 * 8 + F3 * 3680/513 - F4 * 845/4104)
            F6 = h * f(x + h / 2, y - F1 * 8/27 + F2 * 2 - F3 * 3544/2565 + F4 * 1859/4104 - F5 * 11/40)

            error = abs(F1 / 360 - F3 * 128/4275 - F4 * 2197/75240 + F5 / 50 + F6 * 2/55)
            
            if (max(np.array(error)) > 2 * e):
                h = 0.6 * h * np.power(e / max(np.array(error)), 1 / 6)
                continue
            
            y += F1 * 16/135 + F3 * 6656/12825 + F4 * 28561/56430 - F5 * 9/50 + F6 * 2/55
            x += h
            self.sol.append(x, np.array(y))
            
            if max(np.array(error)) < e / 2**7: h *= 2
            else: h = 0.9 * h * np.power(e / max(np.array(error)), 1 / 6)
    
    
    def AB5(self, h, xto):
        ''' Fifth order Adams-Bashforth method. Using RK5 to get first 5 points. '''
        f = self.f
        x = None 
        if self.sol.size() == 0:
            x = self.x0
            self.sol.append(x, np.array(self.y0))
        else:
            x = self.sol.X(-1)
        
        h = np.sign(xto - x) * abs(h)
                
        self.RK5(h, x + 4 * h)
        x = self.sol.X(-1)
        
        while(np.sign(xto - x) == np.sign(h)):
            if abs(xto - x) < abs(h): h = xto - x
            if h == 0: return
            
            y = self.sol.Y(-1) + h * (
                    1901 * f(self.sol.X(-1), self.sol.Y(-1))
                    - 2774 * f(self.sol.X(-2), self.sol.Y(-2))
                    + 2616 * f(self.sol.X(-3), self.sol.Y(-3))
                    - 1274 * f(self.sol.X(-4), self.sol.Y(-4))
                    + 251 * f(self.sol.X(-5), self.sol.Y(-5))
                ) / 720
            x += h
            
            self.sol.append(x, np.array(y))
       
    
    def ABM5EC(self, h, xto):
        ''' Fifth order Adams-Bashforth-Moulton Estimate-Correction method. Using RK5 to get first 5 points. '''
        f = self.f
        x = None 
        if self.sol.size() == 0:
            x = self.x0
            self.sol.append(x, np.array(self.y0))
        else:
            x = self.sol.X(-1)
        
        h = np.sign(xto - x) * abs(h)
                
        self.RK5(h, x + 4 * h)
        x = self.sol.X(-1)
        
        while(np.sign(xto - x) == np.sign(h)):
            if abs(xto - x) < abs(h): h = xto - x
            if h == 0: return
            
            # Estimate
            y = self.sol.Y(-1) + h * (
                    1901 * f(self.sol.X(-1), self.sol.Y(-1))
                    - 2774 * f(self.sol.X(-2), self.sol.Y(-2))
                    + 2616 * f(self.sol.X(-3), self.sol.Y(-3))
                    - 1274 * f(self.sol.X(-4), self.sol.Y(-4))
                    + 251 * f(self.sol.X(-5), self.sol.Y(-5))
                ) / 720
            
            # Correction
            y = self.sol.Y(-1) + h * (
                    251 * f(x + h, y)
                     + 646 * f(self.sol.X(-1), self.sol.Y(-1))
                    - 264 * f(self.sol.X(-2), self.sol.Y(-2))
                    + 106 * f(self.sol.X(-3), self.sol.Y(-3))
                    - 19 * f(self.sol.X(-4), self.sol.Y(-4))
                ) / 720
            
            x += h
            self.sol.append(x, np.array(y))
            
    
    def value_at(self, x):
        if self.sol.size() == 0:
            raise Exception('ERROR: Equation Unsolved')

        if not(self.sol.X(-1) <= x <= self.x0 or self.sol.X(-1) >= x >= self.x0):
            raise Exception('ERROR: Sampling point out of range')
            
        i = bisect.bisect(self.sol.Xs(), x)

        if i == 0: return self.sol.Y(0)
        if i == self.sol.size(): return self.sol.Y(-1)
                
        x1 = self.sol.X(i - 1)
        x2 = self.sol.X(i)
        y1 = self.sol.Y(i - 1)
        y2 = self.sol.Y(i)
        return (y2 - y1) / (x2 - x1) * (x - x1) + y1
        
    
    def last_point(self):
        return (self.sol.X(-1), self.sol.Y(-1))
