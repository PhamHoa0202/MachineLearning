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
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print("\nTRAIN / TEST")
print("Train:", len(X_train))
print("Test :", len(X_test))
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)
y_train_pred = model_linear.predict(X_train)
y_test_pred = model_linear.predict(X_test)

print("\n Hồi quy")
print("Train R2:",r2_score(y_train, y_train_pred))
print("Test R2:", r2_score(y_test, y_test_pred))
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))

# TAO MO HINH POLYNOMIAL
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Tao mo hinh Polynomial bac 10
model_overfit = Pipeline([ ("poly", PolynomialFeatures(degree=10)), ("linear", LinearRegression())])
# Huấn luyện mô hình
model_overfit.fit(X_train, y_train)
# Dự đoán tập Train và Test
y_train_overfit = model_overfit.predict(X_train)
y_test_overfit = model_overfit.predict(X_test)

print("\nPOLYNOMIAL DEGREE 10")
print("Train R2:", r2_score(y_train, y_train_overfit))
print("Test R2:", r2_score(y_test, y_test_overfit))
print("Train RMSE:",np.sqrt(mean_squared_error(y_train, y_train_overfit)))
print("Test RMSE:",np.sqrt(mean_squared_error(y_test, y_test_overfit)))

print("\nSO SANH CAC BAC")
for degree in [1, 2, 3, 5, 10]:
    model = Pipeline([("poly", PolynomialFeatures(degree=degree)), ("linear", LinearRegression())])
    # Huấn luyện
    model.fit(X_train, y_train)
    # Dự đoán
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    # Tính R2
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    # Tính RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred) )
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    print(
        "Degree:", degree,
        "| Train R2:", round(train_r2, 4),
        "| Test R2:", round(test_r2, 4),
        "| Train RMSE:", round(train_rmse, 4),
        "| Test RMSE:", round(test_rmse, 4)
    )
#K-FOLD CROSS VALIDATION
from sklearn.model_selection import KFold, cross_val_score
# Chia dữ liệu thành 5 phần
kf = KFold(n_splits=5,shuffle=True, random_state=4 )
print("\nFOLD CROSS VALIDATION")
for degree in [1, 2, 3, 5, 10]:
    model = Pipeline([("poly", PolynomialFeatures(degree=degree)), ("linear", LinearRegression())])
    # Cross Validation
    scores = cross_val_score( model, x, y, cv=kf, scoring="neg_root_mean_squared_error")
    # Đổi dấu vì sklearn trả về giá trị âm
    rmse = -scores
    print("Degree:", degree, "| RMSE tung Fold:", np.round(rmse, 4),  "| RMSE trung binh:",round(rmse.mean(), 4))
#TIM MO HINH TOT NHAT
from sklearn.model_selection import GridSearchCV
model = Pipeline([ ("poly", PolynomialFeatures()),("linear", LinearRegression())])

# Các bậc muốn thử
param_grid = { "poly__degree": [1, 2, 3, 5, 10]}

# GridSearch + K-Fold
grid = GridSearchCV( model, param_grid, cv=kf, scoring="neg_root_mean_squared_error")

# Huấn luyện
grid.fit(x, y)
print("\n===== MO HINH TOT NHAT =====")
print("Degree tot nhat:",grid.best_params_["poly__degree"])
print( "RMSE tot nhat:", round(-grid.best_score_, 4))
#Du DOAN CAN NHA MOI
new_house = pd.DataFrame({"DienTich": [50],"PhongNgu": [2],"KhoangCach": [8]})

# Lấy mô hình tốt nhất
best_model = grid.best_estimator_
# Huấn luyện lại trên toàn bộ dữ liệu
best_model.fit(x, y)
# Dự đoán
prediction = best_model.predict(new_house)
print("\n===== DU DOAN GIA NHA =====")
print("Dien tich   :", 50)
print("Phong ngu   :", 2)
print("Khoang cach :", 8)

print("Gia du doan :",round(prediction[0], 4))