import numpy as np


class LinearRegression:
    def __init__(self, eps=1e-4):
        self.eps = eps
    
    def train(self, train_x:list, train_y:list):
        mean_x = sum(train_x)
        mean_y = sum(train_y)
        mean_xy = sum([i*j for i, j in zip(train_x, train_y)])
        mean_xx = sum([i*i for i in train_x])
        n = len(train_y)

        self.beta1 = (mean_xy - mean_x*mean_y/n) / (mean_xx - mean_x**2/n)
        self.beta0 = (mean_y - mean_x*self.beta1) / n