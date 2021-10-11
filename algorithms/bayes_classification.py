import collections

import numpy as np
import pandas as pd


def likelihoods_table(x, y, eps=1e-6):
    """
    For each category in x, count probability of each class in y

    Return :
        n distinct category of x * m distinct classes
    """
    count = collections.defaultdict(dict)
    for i, j in zip(x, y):
        count[i][j] = count[i].get(j, 0) + 1
    df = pd.DataFrame(count)
    df.fillna(0, inplace=True)
    df = df.T
    col_sum = df.sum(axis=0)
    df = df.divide(col_sum) + eps
    df: pd.DataFrame = np.log(df)
    return df


def prior_probability(y):
    count = pd.Series(y)
    count = count.value_counts(normalize=True)
    count: pd.Series = np.log(count)
    return count


class NaiveBayes:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def train(self, train_x: np.array, train_y):
        self.likelihoods = [
            likelihoods_table(train_x[:, i], train_y, self.eps).to_dict()
            for i in range(train_x.shape[1])
        ]
        self.prior = prior_probability(train_y).to_dict()
        return self

    def _posterior_proba(self, test_x: np.array, target):
        predictions = np.array(
            [
                [likelihood[target].get(line, np.log(self.eps)) for line in lines]
                for likelihood, lines in zip(self.likelihoods, test_x.T)
            ]
        )
        predictions = predictions.sum(axis=1) + self.prior[target]
        return predictions

    def predict(self, test_x):
        self.probs = {
            target: self._posterior_proba(test_x, target) for target in self.prior
        }
        self.probs = pd.DataFrame(self.probs)
        self.classes = np.argmax(self.probs.values, axis=1)
        return self.classes
