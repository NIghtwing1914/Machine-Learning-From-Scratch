import numpy as np
import matplotlib.pyplot as plt

n_samples = 100
m = 2
X = np.random.rand(n_samples, m)
y = 4 + 3 * X + np.random.randn(n_samples,m)

X_test = np.random.rand(n_samples,m)
y_test = 4 + 3 * X_test + np.random.randn(n_samples,m)
w = np.zeros((m,1))
b= 0

n_iterations = 1000

lr=0.01

for i in range(n_iterations):
    y_pred = np.dot(X,w)+b
    error = y_pred - y

    dw = 1/n_samples * np.dot(X.T, error)
    db = 1/n_samples * np.sum(error)

    w = w - lr * dw
    b = b - lr * db

y_pred = np.dot(X_test,w.T)+b
error = y_pred - y_test
mse = np.mean(np.square(error))
print(mse)

plt.plot(X_test, y_test, 'ro')
plt.plot(X_test, y_pred, 'bo')
plt.show()

