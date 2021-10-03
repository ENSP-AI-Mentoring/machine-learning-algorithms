
from algorithms.bayes_classification import count_on_object
import numpy as np

def test_count_on_object():
    x = [1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 5]
    y = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    espected = np.array([
        [3/6, 0],
        [1/6, 0],
        [2/6, 0],
        [0, 4/5],
        [0, 1/5],
    ]) + 1e-6
    espected = np.log(espected)
    data = count_on_object(x, y, eps=1e-6)
    assert np.mean(espected == data)==1