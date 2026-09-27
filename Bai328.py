import numpy as np

w = np.array([-2, 1, 0])
x = np.array([2, 3, 1])
y = 1
eta = 1 
w_dot_x = np.dot(w, x)
y_pred = 1 if w_dot_x >= 0 else -1

is_mis = (y_pred != y)
print(f"w^T*x ban dau = {w_dot_x}")
print(f"Nhan du doan: {y_pred} | Nhan thuc te: {y}")
print(f"Diem dl bi phan lop sai? {'Co' if is_mis else 'Khong'}")

if is_mis:
    w_new = w + eta * y * x
    print(f"Tron so w moi {w_new}")
    w_new_dot_x = np.dot(w_new, x)
    y_pred_new = 1 if w_new_dot_x >= 0 else -1
    print(f"Gia tri w_new^T*x sau cap nhat = {w_new_dot_x}")
    print(f"Nhan du doan moi: {y_pred_new}")