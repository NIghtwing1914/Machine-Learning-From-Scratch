from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = datasets.load_breast_cancer()
X,y = data.data , data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classes = np.unique(y_train)
mean = np.zeros((len(classes), X_train.shape[1]), dtype=np.float64)
var = np.zeros((len(classes), X_train.shape[1]), dtype=np.float64)
priors = np.zeros(len(classes), dtype=np.float64)

for idx, c in enumerate(classes):
    X_c = X_train[y_train == c]
    mean[idx, :] = X_c.mean(axis=0)
    var[idx, :] = X_c.var(axis=0)
    priors[idx] = X_c.shape[0] / float(X_train.shape[0])

def predict(X):
    y_pred = [_predict(x) for x in X]
    return np.array(y_pred)

def _predict(x):
    posteriors = []
    
    for idx, c in enumerate(classes):
        prior = np.log(priors[idx])
        class_conditional = np.sum(np.log(_pdf(idx, x)))
        posterior = prior + class_conditional
        posteriors.append(posterior)
    
    return classes[np.argmax(posteriors)]

def _pdf(class_idx, x):
    mean_class = mean[class_idx]
    var_class = var[class_idx]
    numerator = np.exp(- (x - mean_class) ** 2 / (2 * var_class))
    denominator = np.sqrt(2 * np.pi * var_class)
    return numerator / denominator

y_pred = predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
