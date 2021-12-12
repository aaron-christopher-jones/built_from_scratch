import numpy as np


class LinearRegression:
    def __init__(self, learning_rate, tol_err):
        self.learning_rate = learning_rate
        self.tol_err = tol_err

    def _func_loss(self, betas):
        Z = self.ytrain - np.dot(self.Xtrain, betas)
        return (1 / self.n) * np.dot(Z.T, Z)

    def _func_gradient(self, betas):
        Z = self.ytrain - np.dot(self.Xtrain, betas)
        return -(2 / self.n) * np.dot(self.Xtrain.T, Z)

    def fit(self, X, y):
        self.Xtrain = X
        self.ytrain = y.reshape(len(y), 1)
        self.n, self.m = X.shape

        betas = np.array([0] * self.m).reshape(self.m, 1)
        lrs = np.array([self.learning_rate] * self.m).reshape(self.m, 1)
        mse = self._func_loss(betas=betas)

        while True:
            betas = betas - lrs * self._func_gradient(betas=betas)
            
            mse_old = mse
            mse = self._func_loss(betas=betas)

            err = abs(mse_old - mse)
            if err <= self.tol_err:
                break
        
        self.coefs = betas
        
    def predict(self, X):
        self.yhat = np.dot(X, self.coefs)
        return self.yhat
