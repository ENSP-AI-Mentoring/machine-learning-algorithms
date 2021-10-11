import numpy as np

from algorithms.preceptron import PerceptronClassifier

def test_perceptron_classifier():
    X = np.random.randn(100, 1)
    y = np.sign(X)
    model = PerceptronClassifier().train(X, y)

    test = np.random.randn(10, 1)
    y_test = np.sign(test).reshape((-1,))

    y_predict = model.predict(test)
    print(y_test)
    print(y_predict)