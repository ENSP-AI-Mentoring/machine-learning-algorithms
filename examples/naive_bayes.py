from algorithms.bayes_classification import NaiveBayes
import os
import pandas as pd

def test_naive_bayes_algorithms():
    data = pd.read_csv(os.getcwd() + "./dataset/dataset.csv")
    train = data[:-10]
    test = data[-10:]
    y = train["Risk"]
    X = train.drop(columns=["Risk"])
    model = NaiveBayes()
    model.train(X.values, y.values)
    predict = model.predict(test[X.columns].values)
    print(predict)
    print(test["Risk"].values)