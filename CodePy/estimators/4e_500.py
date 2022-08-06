print("4_500 output:\n")

import numpy as np
import pandas as pd
import xgboost as xgb
import pyarrow as pa
import pyarrow.parquet as pq
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


if __name__ == "__main__":


    print("Reading files:")
    df = pq.read_table("/ceph/aavocone/Datasets/4_large.parquet")
    df = df.to_pandas()

    #test train split
    X = df[df.columns[:-1]]    #exclude "signal" "classification"
    print(X.columns)
    y = df["signal"]            
    print("test2\n")
    xtrain,xtest,ytrain,ytest = train_test_split(X, y, test_size = 0.33, stratify = y)
    xtrain,xval,ytrain,yval = train_test_split(xtrain, ytrain, test_size = 0.5)
    print("test3\n")
    print("Number of columns:",len(xtrain.columns))


    weight = (len(ytest)-sum(ytest))/sum(ytest)
    print("test4\n")

    estimator = 500


    print(f"test{estimator}\n")

    model     = XGBClassifier(  n_estimators = estimator, learning_rate = 0.2,
                                eval_metric = "logloss", scale_pos_weight = weight, use_label_encoder =False,
                                verbosity=0, n_jobs = 30, early_stopping_rounds=20
                            )
                            
    model.fit(xtrain,ytrain, eval_set=[(xtrain,ytrain),(xval,yval)])
    model.save_model(f"/ceph/aavocone/models/new_4_6_model{estimator}.txt")