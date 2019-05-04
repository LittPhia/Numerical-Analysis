import numpy as np

class continued_fraction:
    def __init__(self, a, b):
        '''a, b must be callable with an interger parameter.'''
        self.a = a
        self.b = b
        self.A = [0, a(1)]
        self.B = [1, b(1)]
        
    def calculate(self, n):
        '''return y = a(0) / (b(0) + a(1) / (b(1) + a(2) / (b(2) + ... ))), iteration will continue to order n'''
        if(n + 1 > len(self.A)):
            for i in range(len(self.A), n + 1):
                self.A.append(self.b(i) * self.A[i-1] + self.a(i) * self.A[i-2])
                self.B.append(self.b(i) * self.B[i-1] + self.a(i) * self.B[i-2])
        else: pass
        return self.A[n] / self.B[n]

def a(n):
    return n**2 / 3
def b(n):
    return 2 * n + 1

f = continued_fraction(a, b)
for i in range(10):
    print(1 / np.sqrt(3) / (1 + f.calculate(i)))
