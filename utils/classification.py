from __future__ import absolute_import

import numpy as np
import random
from utils.display import printProgressBar

class log_reg:

    def __init__(self, eta=0.001):
        self.beta = None
        self.eta = eta

    def gradient(self, x_i, y_i):
        n = len(x_i)
        x = np.ones(n+1)
        for i, val in enumerate(x_i):
            x[i+1] = val
        return (x*(y_i - self.Pr(x_i)))[0]

    def fit(self, X, Lb, epochs=5, itermax=2):
        n, m = X.shape
        dim = len(X[0])

        self.beta = np.zeros((1, dim+1))
        for i in range(dim+1):
            self.beta[0][i] = -random.randrange(40)/10
        for p in range(epochs):
            #printProgressBar(i, epochs-1, str(i+1) + '/' + str(epochs) + str(' epoch'))
            grad_val = np.zeros(dim + 1)
            self.beta = self.beta + self.eta*(grad_val)
            r = np.array([self.gradient(X[i], Lb[i]) for i in range(n)])
            grad_val = np.sum(r, axis=0)
            #Gradient Descent
            for j in range(itermax):
                self.beta = self.beta + self.eta*(grad_val)
                r = np.array([self.gradient(X[i], Lb[i]) for i in range(n)])
                grad_val = np.sum(r, axis=0)
        #print(self.beta)

    def Pr(self, x_i):
        d = len(x_i) + 1# we add 1 to x_i for the coefficient beta_0
        x = np.ones((1, d))
        for i in range(1, d):
            x[0][i] = x_i[i-1]
        return 1/(1+np.exp(-np.dot(self.beta, x.T)))

    def accuracy(self, X, Y, p):
        n = len(X)
        acc = 0
        for i in range(n):
            test = 0
            if self.Pr(X[i]) > p:
                test = 1# We are in the first class
            else:
                test = 0
            if test == Y[i]:
                acc += 1
        return acc/n
