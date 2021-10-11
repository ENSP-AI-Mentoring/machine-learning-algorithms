import numpy as np

from algorithms.bayes_classification import likelihoods_table
from algorithms.bayes_classification import prior_probability


def test_count_on_object():
    x = [1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 5]
    y = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    espected = (
        np.array(
            [
                [3 / 6, 0],
                [1 / 6, 0],
                [2 / 6, 0],
                [0, 4 / 5],
                [0, 1 / 5],
            ]
        )
        + 1e-6
    )
    espected = np.log(espected)
    data = likelihoods_table(x, y, eps=1e-6)
    assert np.mean(espected == data.values) == 1


def test_prior_probability():
    y = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    espected = np.log(np.array([6 / 11, 5 / 11]))
    counted = prior_probability(y)
    print(counted)
    print(espected)
    assert np.mean(espected == counted.values) == 1
