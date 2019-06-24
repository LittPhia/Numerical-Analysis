import numpy as np
import numpy.linalg
import InitialValueProblem
import bisect


class BoundaryValueProblem:
    '''
        Current version 1.0.1.
        This class is designed for solving n-dim differential equations with the form: 
            x''(t) = f(t, x, x')
            x(a) = alpha
            x(b) = beta
        a, b should be n-elements np.arrays or just single elements (when n == 1).
        alpha, beta are single elements.
        x(t) and f(t, x, x') are supposed to be functions which return 1-dim value and compatible with t == a, t == b.
        
        Implemented methods:
            Mutiple Shooting Method
            Finite Difference Method
            
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
        
        
    def __init__(self, f, a, alpha, b, beta):
        self.f = f
        self.a = a
        self.alpha = alpha
        self.b = b
        self.beta = beta
        self.sol = BoundaryValueProblem.Solution()
    
    
    def clear(self):
        del self.sol
        self.sol = BoundaryValueProblem.Solution()

        
    def SM(self, h, e):
        ''' Shooting Method '''
        
        def f(t, y):
            return np.array([y[1], self.f(t, y[0], y[1])])
        
        if abs(h) <= abs(self.b - self.a) / 4:
            h *= 4
        
        z_1 = 0.0
        subproblem = InitialValueProblem.InitialValueProblem(f, self.a, np.array([self.alpha, z_1]))
        subproblem.RK4(h, self.b)
        phi_1 = subproblem.last_point()[1][0] - self.beta;
        
        z_2 = 1.0
        while abs(phi_1) > abs(4 * e):
            subproblem = InitialValueProblem.InitialValueProblem(f, self.a, np.array([self.alpha, z_2]))
            subproblem.RK4(h, self.b)
            phi_2 = subproblem.last_point()[1][0] - self.beta;
            
            if phi_1 == phi_2:
                z_2 += 1.0
            else:
                z_3 = z_2 - ((z_2 - z_1) / (phi_2 - phi_1)) * phi_2
                z_2 = z_3
                z_1 = z_2
                phi_1 = phi_2
            
        h /= 4
        while abs(phi_1) > abs(e):
            subproblem = InitialValueProblem.InitialValueProblem(f, self.a, np.array([self.alpha, z_2]))
            subproblem.RK4(h, self.b)
            phi_2 = subproblem.last_point()[1][0] - self.beta;
            
            z_3 = z_2 - ((z_2 - z_1) / (phi_2 - phi_1)) * phi_2
            z_2 = z_3
            z_1 = z_2
            phi_1 = phi_2

        for subsol in subproblem.sol.sol_:
            self.sol.append(subsol[0], subsol[1][0])
            
    
    def FDM(self, u, v, w, n):
        ''' Finite Difference Method. Using linear approximation { f(t, x, x') == u(t) + v(t) x' + w(t) x'' } '''
        
        h = (self.b - self.a) / (n + 1)
        A = np.eye(n, n)
        b = np.eye(n, 1)
        
        for i in range(n - 1):  
            A[i + 1][i] = - h * w(self.a + (i + 2) * h) / 2 - 1
        for i in range(n):
            A[i][i] = 2 + h * h * v(self.a + (i + 1) * h)
        for i in range(n - 1):
            A[i][i + 1] = h * w(self.a + (i + 1) * h) / 2 - 1
            
        for i in range(n):
            b[i] = - h * h * u(self.a + (i + 1) * h)
        b[0] += (h * w(self.a + h) / 2 + 1) * self.alpha
        b[n - 1] -= (h * w(self.b - h) / 2 - 1) * self.beta
        
        s = list(np.linalg.solve(A, b))
        for i in range(len(s)): s[i] = s[i][0]
            
        s = [self.alpha] + s + [self.beta]
        for i in range(len(s)):
            self.sol.append(self.a + i*h, s[i])
        
    
    def CMCBS(self, n):
        ''' Collocation Method with cubic B-spline '''
        
    
    
    def value_at(self, x):
        if self.sol.size() == 0:
            raise Exception('ERROR: Equation Unsolved')

        if not(self.a <= x <= self.b or self.b >= x >= self.a):
            raise Exception('ERROR: Sampling point out of range')
            
        i = bisect.bisect(self.sol.Xs(), x)        
        if i == 0: return self.sol.Y(0)
        if i == self.sol.size(): return self.sol.Y(-1)
        
        x1 = self.sol.X(i - 1)
        x2 = self.sol.X(i)
        y1 = self.sol.Y(i - 1)
        y2 = self.sol.Y(i)
        return (y2 - y1) / (x2 - x1) * (x - x1) + y1
        