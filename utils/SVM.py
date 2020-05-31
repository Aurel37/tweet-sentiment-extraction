import math 
import numpy as np
from random import randint

def quadra_error(w, l, X, Y):
    n = len(X)
    s1 = l * (np.linalg.norm(w)**2)/2
    s2 = 0
    for i in range(n):
        s2 += (Y[i] - w.T@X[i])**2    
    return s1 + s2/n

def grad_regression(X, Y, i, w):
    print(w.shape)
    print(X.shape)
    print(Y.shape)
    return -2*(Y[i] - w.T@X[i])*X[i]


class SAGRegression():
    
    def __init__(self, lambada=0.001, eta=0.01):
        
        self.lambada = lambada 
        self.eta = eta
        self.w = None 
        self.b = None
        
    def fit(self, X, Y, epochs=10):
        
        # First, on rajoute des 1 au X 
        n, m = X.shape
        Xbis = np.ones((n, m + 1))
        print(Xbis.shape)
        Xbis[:, :m] = np.copy(X)
        print(Xbis.shape)
        Loss = []
        #. On gère le warm start
        if self.w is None:
            w = np.random.random_sample(m+1)
        else:
            w = np.zeros(m+1)
            w[:-1] = self.w
            w[-1] = self.b
        d = np.zeros(w.shape)
        y = np.zeros(Xbis.shape)
        for i in range(n*epochs):
            print(i/epochs)
            ir = np.random.randint(0, n-1)
            d -= y[ir]
            y[ir] = grad_regression(Xbis, Y, ir, w)
            d += y[ir]
            w = w  - self.eta*self.lambada * np.concatenate((w[:-1],0), axis=None) - (self.eta/n)* d
            self.w = w[:-1]
            self.b = w[-1]
            Loss.append(quadra_error(w, self.lambada, Xbis, Y))
        self.w = w[:-1]
        self.b = w[-1]
        return self.w, self.b, Loss

    
        
    def predict(self, X): 
        n = len(X)
        Y = np.zeros(n)
        for i in range(n):
            Y[i] = self.w.T@X[i] + self.b
        return Y
