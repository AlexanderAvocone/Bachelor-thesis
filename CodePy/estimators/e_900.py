print("e.py output:\n")

print("Importing numpy:")
import numpy as np
print("completed!\n")
print("Importing pandas:")
import pandas as pd
print("completed!\n")
print("Importing xgb:")
import xgboost as xgb
print("completed!\n")
print("Importing XGBC:")
from xgboost import XGBClassifier
print("completed!\n")
print("Importing train_test_split:")
from sklearn.model_selection import train_test_split
print("completed!\n")

import pyarrow as pa
import pyarrow.parquet as pq


if __name__ == "__main__":


    print("Reading files:")
    df = pq.read_table("/ceph/aavocone/Datasets/3_large.parquet")
    df = df.to_pandas()


    #test train split
    X = df[df.columns[:-3]]    #exclude "signal" "classification"
    y = df["signal"]            
    print("test2\n")
    xtrain,xtest,ytrain,ytest = train_test_split(X, y, test_size = 0.33, stratify = y)
    xtrain,xval,ytrain,yval = train_test_split(xtrain, ytrain, test_size = 0.5)
    print("test3\n")
    print("Number of columns:",len(xtrain.columns))


    weight = (len(ytest)-sum(ytest))/sum(ytest)
    print("test4\n")


 
    estimator = 900


    print(f"test{estimator}\n")

    model     = XGBClassifier(  n_estimators = estimator, learning_rate = 0.2,
                                eval_metric = "logloss", scale_pos_weight = weight, use_label_encoder =False,
                                verbosity=0, n_jobs = 30
                            )
                            
    model.fit(xtrain,ytrain, eval_set=[(xval,yval)])
    model.save_model(f"/work/aavocone/models/3_0_model{estimator}.txt")