import numpy as np
import pandas as pd
from functions import load_parquet
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


if __name__ == "__main__":
    df = load_parquet("/ceph/aavocone/Datasets/3_large.parquet")

    df.drop(["class","cladd"], axis=1, inplace=True)
    print(df.columns)


    x = df[df.columns[:-1]]
    y = df["signal"]
    xtrain,xval,ytrain,yval = train_test_split(x, y, test_size = 0.33, stratify = y)
    xtrain,xtest,ytrain,ytest = train_test_split(xtrain, ytrain, test_size = 0.5, stratify= ytrain)

    over_sample = RandomOverSampler(sampling_strategy = 0.2)
    xtrain, ytrain = over_sample.fit_resample(xtrain,ytrain)
    xtest, ytest = over_sample.fit_resample(xtest,ytest)
    xval, yval = over_sample.fit_resample(xval,yval)

    print("Length training set:             ",len(ytrain))
    print("Number of signals in training:   ",sum(ytrain))
    print("Length testing set:              ",len(ytest))
    print("Number of signals in testing:    ",sum(ytest))
    print("Length validation set:           ",len(yval))
    print("Number of signals in validation: ",sum(yval))


    estimator = 1000
    weight = (len(ytest)-sum(ytest))/sum(ytest)

    print(f"test{estimator}\n")

    model     = XGBClassifier(  n_estimators = estimator, learning_rate = 0.2,
                                eval_metric = "logloss", scale_pos_weight = weight, use_label_encoder =False,
                                verbosity=0, n_jobs = 30, early_stopping_rounds=20
                            )
                            
    model.fit(xtrain,ytrain, eval_set=[(xtrain,ytrain),(xval,yval)])
    model.save_model(f"/work/aavocone/models/3_0_model{estimator}_over.txt")