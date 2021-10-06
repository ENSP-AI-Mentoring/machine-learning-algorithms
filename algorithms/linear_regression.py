import numpy as np


class LinearRegression:
    def __init__(self, eps=1e-4):
        self.eps = eps
    
    def compute_loss(self):
        y_pred = self.predict(self.x)
        sse = sum((i-j)**2 for i, j in zip(self.y, y_pred))
        self.sigma = (sse / (len(self.x)-2)) ** (0.5)
        return self.sigma
    
    def train(self, train_x:list, train_y:list):
        self.x = train_x.copy()
        self.y = train_y.copy()

        mean_x = sum(train_x)
        mean_y = sum(train_y)
        mean_xy = sum([i*j for i, j in zip(train_x, train_y)])
        mean_xx = sum([i*i for i in train_x])
        n = len(train_y)

        self.sxy = mean_xy - mean_x*mean_y/n
        self.sxx = mean_xx - mean_x**2/n

        self.beta1 = self.sxy / self.sxx
        self.beta0 = (mean_y - mean_x*self.beta1) / n
        return self

    def predict(self, test_x):
        preds = [i*self.beta1+self.beta0 for i in test_x]
        return preds.copy()