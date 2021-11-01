import matplotlib.pyplot as plt
import numpy as np

from scratch.algos import linear_regression as lr
from scratch.utils import evaluation as eval


# build test dataset
n = 500
eps = np.random.normal(loc=0, scale=1, size=n).reshape(n, 1)
x0 = np.array([1] * n).reshape(n, 1)
x1 = np.random.normal(loc=0, scale=5.6, size=n).reshape(n, 1)
x2 = np.random.normal(loc=2.3, scale=0.6, size=n).reshape(n, 1)
x3 = np.random.gamma(shape=1.1, scale=2, size=n).reshape(n, 1)
x4 = np.random.beta(a=3, b=3, size=n).reshape(n, 1)
X = np.concatenate((x0, x1, x2, x3, x4), axis=1)
y = (7.2 * x0) + (0.41 * x1) + (1.93 * x2) + (2.11 * x3) + (4.2 * x4) + eps
Xtrain = X[:400]
ytrain = y[:400]
Xvalid = X[400:]
yvalid = y[400:]

# run the linear regression model
m = lr.LinearRegression(
    y=ytrain, 
    X=Xtrain, 
    learning_rate=0.01, 
    tol_err=1e-06
)
m.fit()

yhat = m.predict(Xvalid=Xvalid)

print("Coefficients: {}".format(m.coefs))
print("MAE%: {}".format(eval.metric_maep(y=yvalid, yhat=yhat)))
print("RMSE: {}".format(eval.metric_rmse(y=yvalid, yhat=yhat)))

plt.plot(yhat, yvalid, "o")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.plot(yhat, "-", label="predict")
plt.plot(yvalid, "-", label="actual")
plt.legend()
plt.show()
