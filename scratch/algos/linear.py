import numpy as np

from scratch.abstract import AbstractModel


class LinearRegression(AbstractModel):
    def __init__(self, learning_rate, tol_err, max_iters):
        self.learning_rate = learning_rate

        self.tol_err = tol_err
        self.max_iters = max_iters

    @staticmethod
    def _loss(y, yhat, n):
        z = y - yhat
        return (1 / n) * np.dot(z.T, z)

    def fit(self, X, y):
        n, m = X.shape

        self.weights = np.zeros(m)
        self.bias = 0
        self.loss = []

        self.iters = 0
        while True:
            yhat = np.dot(X, self.weights) + self.bias
            # compute gradient
            dw = (2 / n) * np.dot(X.T, (yhat - y))
            db = (2 / n) * np.sum(yhat - y)
            # update weights and bias parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            # calculate and compare tolerances
            self.loss.append(self._loss(y=y, yhat=yhat, n=n))
            if self.iters > 0:
                err = abs(self.loss[self.iters - 1] - self.loss[self.iters])
                if err <= self.tol_err or self.iters == self.max_iters:
                    break
                else:
                    self.iters += 1
            else:
                self.iters += 1
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
