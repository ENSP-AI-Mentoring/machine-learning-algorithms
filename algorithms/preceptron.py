import numpy as np


class PerceptronClassifier:
    def __init__(self) -> None:
        np.random.seed(41)

    def train(self, train_x, train_y):
        encoder = {0: -1, 1: 1, -1: -1}
        train_y = np.array(train_y).reshape((-1,))
        train_y = np.array([encoder[i] for i in train_y])
        X = np.c_[list(range(len(train_y))), train_x]
        np.random.shuffle(X)
        train_x = X[:, 1:]
        train_x = np.c_[np.ones((len(train_y), 1)), train_x]
        train_y = train_y[[int(i) for i in X[:, 0]]]
        W = np.random.randn(train_x.shape[1], 1)
        for x, y in zip(train_x, train_y):
            y_predict = np.sign(np.dot(x, W))[0]
            if y != y_predict:
                W = W + y * x.reshape((-1, 1))
        self.W = W
        return self

    def predict(self, test_x):
        test_x = np.c_[np.ones((test_x.shape[0], 1)), test_x]
        preds = np.sign(np.dot(test_x, self.W))
        preds = np.sign(preds).reshape((-1,))
        return preds
