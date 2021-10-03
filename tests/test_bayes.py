
from algorithms.bayes_classification import count_on_object


def test_count_on_object():
    x = [1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 5]
    y = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    data = [
        [3, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [3, 0],
        [3, 0],
    ]
    data = count_on_object(x, y, eps=1e-6)