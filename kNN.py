import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter

iris = datasets.load_iris()
X,y = iris.data , iris.target

k=3

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

y_pred =[]
for x in X_test:
    distances = [euclidean_distance(x,x1) for x1 in X_train]
    indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in indices]
    predicted = Counter(k_nearest_labels).most_common()
    y_pred.append(predicted[0][0])


print(y_pred)
acc = np.sum(y_pred == y_test)/len(y_test)
print(acc)



