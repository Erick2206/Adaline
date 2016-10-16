import numpy as np

class Adaline(object):
    def __init__(self,eta=0.0001,n_iter=50):
        self.eta=eta
        self.n_iter=n_iter

    def fit(self,X,y):
        self.weights=np.zeros(1+X.shape[1])
        self.costs=[]

        for i in range(self.n_iter):
            output=self.net_input(X)
            errors=(y-output)
            self.weights[1:] +=self.eta* X.T.dot(errors)
            self.weights[0] +=self.eta*errors.sum()
            cost=(errors**2).sum()/2.0
            self.costs.append(cost)

    def net_input(self,X):
        return np.dot(X,self.weights[1:]) + self.weights[0]

    def activation(self,X):
        return self.net_input(X)

    def predict(self,X):
        return np.where(self.activation(X) >= 0.0,1,-1)
