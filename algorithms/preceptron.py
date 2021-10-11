import numpy as np


class PerceptronClassifier:
    def __init__(self) -> None:
        np.random.seed(41)

    def train(self, train_x, train_y):
        encoder = {0: -1, 1: 1, -1: -1}
        train_y = np.array([encoder[i] for i in train_y])
        X = np.c_(range(len(train_y)), train_x)
        np.random.shuffle(X)
        train_x = X[:, 1:]
        train_y = train_y[X[:, 0]]
        W = np.random.randn(train_x.shape[0], 1)
        for x, y in zip(train_x, train_y):
            y_predict = np.sign(np.dot(W, x))
            if y != y_predict:
                W = W + y * x.T
        self.W = W
        return self

    def predict(self, test_x):
        preds = np.sign(np.dot(test_x, self.W))
        preds = np.sign(preds)
        return preds
