import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import numpy as np
import random

class LogisticRegression():
    
    def __init__(self, method, learning_rate, num_iterations, track_loss):
        self.method = method.upper()
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.track_loss = track_loss
        
    def update_weights_with_GD(self,X, y):
        y_pred = self.predict(self.X)
        self.weights = self.weights - self.learning_rate * np.dot(self.X.T, y_pred-self.y) / len(self.X)
        return self.weights
    
    def update_weights_with_SGD(self, X, y):
        y_pred = self.predict(self.X)
        sample = random.randint(0, len(self.y)-1)
        self.weights = self.weights - self.learning_rate * (np.dot((y_pred[sample] - self.y[sample]), self.X[sample, :]))
        return self.weights
    
    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))
    
    def cost_function(self, X, y, weights):
        n = len(self.X)
        y_pred = self.predict(self.X)
        cost = (-1/n)*(np.dot(self.y,np.log(y_pred)) + np.dot((1-self.y),(1-y_pred)))

        return cost        
        
    def fit(self, X, y):
        
        self.y = y
        self.X = X
        self.weights = np.zeros((X.shape[1],))

        cost_history = np.zeros((self.num_iterations))
        cost = 0

        for i in range(self.num_iterations):
            if cost!=np.inf:
                if self.method == 'GD':
                    self.weights = self.update_weights_with_GD(X, y)
                elif self.method == 'SGD':
                    self.weights = self.update_weights_with_SGD(X, y)
                else:
                    raise Exception('Problem with method')
                    
                cost = self.cost_function(self.weights, self.X, self.y)
                cost_history[i] = cost
        if self.track_loss:        
            return self.weights, cost_history
        else:
            return self.weights
    
    def predict(self, X):
        z = np.dot(X, self.weights)
        return self.sigmoid(z)
    
def predict_prob(self, X_test=None):
    if X_test is None:
        self.y_pred = np.dot(self.X, self.w)
        self.y_pred = 1 / (1 + np.exp(-self.y_pred))
        return self
    else:
        X_test = np.c_[X_test, np.ones(X_test.shape[0])]
        self.y_pred = np.dot(X_test, self.weights)
        return 1 / (1 + np.exp(-self.y_pred))
    

    def score(self, y, y_pred):
        return sum((y_pred-self.y)**2)/ len(self.y)
        
# Let's test this on the iris data
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

log_reg_sgd = LogisticRegression(method='SGD', learning_rate=0.1, num_iterations=1000, track_loss=True)
cost_histroy = log_reg_sgd.fit(X_train,y_train)
y_pred = log_reg_sgd.predict(X_test)

# plot the cost
plt.plot(cost_histroy[1])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# plot the elbow
log_reg_gd = LogisticRegression(method='GD', learning_rate=0.1, num_iterations=1000, track_loss=True)
cost_histroy = log_reg_gd.fit(X_train,y_train)
y_pred = log_reg_gd.predict(X_test)

plt.plot(cost_histroy[1])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
