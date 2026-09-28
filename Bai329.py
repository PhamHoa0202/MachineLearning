import numpy as np

class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=42):
        self.eta = eta           
        self.n_iter = n_iter        
        self.random_state = random_state
        self.w = None         

    def fit(self, X, y):
        print("Huan luyen mo hinh Perceptron")
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        
        for _ in range(self.n_iter):
            for xi, target in zip(X, y):
                y_pred = self._predict_single(xi)
                if y_pred != target:
                    update = self.eta * target
                    self.w[1:] += update * xi  
                    self.w[0] += update     
        return self

    def _net_input(self, X):
        print("Tich vo huong w^T * x + b")
        return np.dot(X, self.w[1:]) + self.w[0]

    def _predict_single(self, xi):
        print("Du bao nhan cho mau dl")
        return 1 if self._net_input(xi) >= 0.0 else -1

    def predict(self, X):
        print(" Du bao nhan cho tap dl X: mang 2D kich thuoc (n_samples, n_features)")
        return np.where(self._net_input(X) >= 0.0, 1, -1)