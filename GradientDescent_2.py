#tim gia tri cuc tieu cua ham so g(x) = (1/3)x^3-x
import numpy as np

def grad(x):
    return x**2-1
def cost(x):
    return (1/3)*(x**3)-x 
def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) <1e-3:
            break 
        x.append(x_new)
    return (x, it)
x0 = 3.0
eta = 0.1
(x1, it1) = myGD1(x0, eta)
print("Solution x1 = %f, cost = %f, after %d interations" %(x1[-1], cost(x1[-1]), it1))