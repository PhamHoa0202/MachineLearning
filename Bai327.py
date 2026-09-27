import numpy as np

w = np.array([1,2,-10])
x = np.array([3,4,1])
y_true = -1 # nhãn thực tế

w_dot_x = np.dot(w,x)
print(f"Ket qua w^T*x = {w_dot_x}")

y_pred = 1 if w_dot_x >=0 else -1
print(f"Nhan du doan = {y_pred}")

is_mis = (y_pred != y_true)
print(f"Diem dl bi phan lop sai {'co' if is_mis else 'khong'}")
# Buoc cap nhat trong so neu bị phan lop sai
if is_mis:
    learning_rate = 1.0
    w_new = w + learning_rate*y_true*x
    print(f"Trong so moi w_new = {w_new}")