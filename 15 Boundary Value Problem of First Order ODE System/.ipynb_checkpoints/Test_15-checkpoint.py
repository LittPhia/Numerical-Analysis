from InitialValueProblem import *
from BoundaryValueProblem import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# REQUIED EQUATION
def f(t, x, xd1):
    return x * x + t
  
problem = BoundaryValueProblem(f, 0, 0, 1, 0)
problem.SM(1e-5, 1e-12)

# PLOT
Ts = np.arange(0, 1 + 0.01, 0.01)
Xs = np.array([problem.value_at(t) for t in Ts])

plt.grid()
plt.title('Figure of x-t', fontsize = 24)
plt.plot(Ts, Xs, c = 'r')

plt.show()
