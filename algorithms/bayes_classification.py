import collections
import numpy as np
import pandas as pd

def count_on_object(x, y, eps=1e-6):
    """
        For each category in x, count probability of each class in y

        Return :
            n distinct category of x * m distinct classes
    """
    x = list(x)
    y = list(y)
    count = collections.defaultdict(collections.defaultdict(int))
    for i, j in zip(x, y):
        count[i][j] += 1
    df = pd.DataFrame(count) + eps
    df.fillna(0, inplace=True)
    col_sum = df.sum(axis=1)
    df = df/col_sum
    df = np.log(df)
    return df