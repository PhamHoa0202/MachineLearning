import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

data = { 
    "DienTich": [30,40,50,60,70,80,90,100,110,120,
                 45,55,75,95,105,65,85,35,115,125],

    "PhongNgu": [1,2,2,2,3,3,4,3,4,5,
                 1,3,2,4,3,2,4,1,5,4],

    "KhoangCach": [4,5,6,7,7,8,9,10,10,11,
                   5,6,8,9,11,7,9,4,10,12],

    "Gia": [8,9,11,12,15,16,19,20,22,25,
            10,12,16,21,24,14,18,8.5,23,26]
}
df = pd.DataFrame(data)
print(df)

x = df[["DienTich", "PhongNgu", "KhoangCach"]]
y = df["Gia"]

#Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("\n===== TRAIN / TEST =====")
print("Train:", len(X_train))
print("Test :", len(X_test))

#
model_linear = LinearRegression()

model_linear.fit(X_train, y_train)

y_train_pred = model_linear.predict(X_train)
y_test_pred = model_linear.predict(X_test)

print("\n Hồi quy")

print("Train R2:",
      r2_score(y_train, y_train_pred))

print("Test R2:",
      r2_score(y_test, y_test_pred))

print("Train RMSE:",
      np.sqrt(mean_squared_error(y_train, y_train_pred)))

print("Test RMSE:",
      np.sqrt(mean_squared_error(y_test, y_test_pred)))

# ==========================================
# 5. TAO MO HINH POLYNOMIAL
# ==========================================

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Tao mo hinh Polynomial bac 10
model_overfit = Pipeline([
    ("poly", PolynomialFeatures(degree=10)),
    ("linear", LinearRegression())
])

# Huấn luyện mô hình
model_overfit.fit(X_train, y_train)

# Dự đoán tập Train và Test
y_train_overfit = model_overfit.predict(X_train)
y_test_overfit = model_overfit.predict(X_test)

print("\n===== POLYNOMIAL DEGREE 10 =====")

print("Train R2:",
      r2_score(y_train, y_train_overfit))

print("Test R2:",
      r2_score(y_test, y_test_overfit))

print("Train RMSE:",
      np.sqrt(mean_squared_error(y_train, y_train_overfit)))

print("Test RMSE:",
      np.sqrt(mean_squared_error(y_test, y_test_overfit)))