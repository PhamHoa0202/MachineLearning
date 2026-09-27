#Tim gia tri cuc tieu cua ham so f(x) = x^2-2
import numpy as np
def grad(x):
    return 2*x;
def cost(x):
    return x**2-2;
def myGD1(x0, eta):
    x=[x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new))<1e-3:
            break
        x.append(x_new)
    return(x,it)    

x0=5.0
eta =0.1
(x1, it1) = myGD1(x0,eta)
print("Solution x1 = %f, cost = %f, after %d interations" %(x1[-1], cost(x1[-1]), it1))
