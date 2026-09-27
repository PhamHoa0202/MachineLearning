import numpy as np

def grad(x):
    return( 2*x-4)
def cost(x):
    return (x**2-4*x+5)
def myGD1(x0, eta):
    x =[x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) <1e-3:
            break
        x.append(x_new)
    return(x, it)
(x1, it1) = myGD1(.5, .2)
print("Solution x1 = %f, cost = %f, after %d interations" % (x1[-1], cost(x1[-1]), it1))