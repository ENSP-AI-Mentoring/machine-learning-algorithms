import numpy as np

from algorithms.linear_regression import LinearRegression

def test_linear_regression_algorithm():
    x = np.random.randn(30).tolist()
    y = [i*2+3+np.random.rand()/100 for i in x]

    test_x = np.random.randn(10).tolist()
    test_y = [i*2+3 for i in test_x]

    model = LinearRegression().train(x, y)
    print(test_y)
    print(model.predict(test_x))
    