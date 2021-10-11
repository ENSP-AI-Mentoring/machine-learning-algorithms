import numpy as np

from algorithms.linear_regression import LinearRegression, MultiLinearRegression

def test_linear_regression_algorithm():
    x = np.random.randn(30).tolist()
    y = [i*2+3+j for i, j in zip(x, (np.random.randn(30)/50).tolist())]

    test_x = np.random.randn(10).tolist()
    test_y = [i*2+3 for i in test_x]

    model = LinearRegression().train(x, y)
    print(test_y)
    print(model.predict(test_x))

    print(model.compute_loss())

def test_multi_linear_regression_algorithm():
    x = np.random.randn(30, 5)
    
    y = np.dot(x, np.random.randn(5, 1)) + np.random.randn(30, 1)/100

    test_x = np.random.randn(10, 5)
    test_y = np.dot(x, np.random.randn(5, 1))

    model = MultiLinearRegression().train(x, y)
    print(test_y)
    print(model.predict(test_x))

    print(model.compute_loss())