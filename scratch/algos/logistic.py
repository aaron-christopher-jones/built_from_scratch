import numpy as np

from scratch.abstract import AbstractModel


class LogisticRegression(AbstractModel):
    def __init__(self, learning_rate, tol_err, max_iters):
        self.learning_rate = learning_rate

        self.tol_err = tol_err
        self.max_iters = max_iters
    
    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _loss(y, yhat):
        return (-y * np.log(yhat) - (1 - y) * np.log(1 - yhat)).mean()

    def fit(self, X, y):
        n, m = X.shape

        self.weights = np.zeros(m)
        self.bias = 0
        self.loss = []

        self.iters = 0
        while True:
            z = np.dot(X, self.weights) + self.bias
            yhat = self._sigmoid(z=z)
            # compute gradient
            dw = (1 / n) * np.dot(X.T, (yhat - y))
            db = (1 / n) * np.sum(yhat - y)
            # update weights and bias parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            # calculate and compare tolerances
            self.loss.append(self._loss(y=y, yhat=yhat))
            if self.iters > 0:
                err = abs(self.loss[self.iters - 1] - self.loss[self.iters])
                if err <= self.tol_err or self.iters == self.max_iters:
                    break
                else:
                    self.iters += 1
            else:
                self.iters += 1

    def predict(self, X, threshold):
        z = np.dot(X, self.weights) + self.bias
        y_prob = self._sigmoid(z=z)
        y_class = [1 if i >= threshold else 0 for i in y_prob]
        return np.array(y_prob), np.array(y_class)
