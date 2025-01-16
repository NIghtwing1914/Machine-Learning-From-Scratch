import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_iterations = 1000

def sigmoid(x):
    return 1/(1+np.exp(-x))

n,m = X_train.shape

w = np.zeros((m,1))
b = 0

lr = 0.001
print(y_train.shape)
for i in range(n_iterations):

    y_pred = sigmoid(np.dot(X_train,w)+b)
    error = y_pred - y_train.reshape(-1, 1)

    dw = 1/n * np.dot(X_train.T,error)
    db = 1/n * np.sum(error)

    w = w - lr*dw
    b = b - lr*db

linear_pred = np.dot(X_test,w)+b
sigmoid_pred = sigmoid(linear_pred)
print(sigmoid_pred)
# Convert sigmoid predictions to binary values
y_pred_binary = [1 if i > 0.5 else 0 for i in sigmoid_pred]

print(y_pred_binary)
print(accuracy_score(y_test, y_pred_binary))
