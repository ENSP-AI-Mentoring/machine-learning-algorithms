import collections
import numpy as np
import pandas as pd

def likehoods_table(x, y, eps=1e-6):
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
    df = np.log(df)
    return df

def prior_probability(y):
    count = pd.Series(y)
    count = count.value_counts(normalize=True)
    return np.log(count)