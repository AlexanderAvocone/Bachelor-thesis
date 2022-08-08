import numpy as np
import pandas as pd
from functions import load_parquet
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import xgboost as xgb


if __name__ == "__main__":
    df = load_parquet("/ceph/aavocone/Datasets/3_large.parquet")

    df.drop(["class","cladd"], axis=1, inplace=True)
    print(df.columns)


    x = df[df.columns[:-1]]
    y = df["signal"]

    under_sample = RandomUnderSampler(sampling_strategy = 0.1)
    xunder, yunder = under_sample.fit_resample(x,y)
    print("Length undersampled set:     ",len(xunder))
    print("Number of signals :          ",sum(yunder))


    xtrain,xval,ytrain,yval = train_test_split(xunder, yunder, test_size = 0.33, stratify = y)
    xtrain,xtest,ytrain,ytest = train_test_split(xtrain, ytrain, test_size = 0.5, stratify= ytrain)


    estimator = 1000
    weight = (len(ytest)-sum(ytest))/sum(ytest)
    
    print(f"test{estimator}\n")
    

    model     = xgb.XGBClassifier(  n_estimators = estimator, learning_rate = 0.2,
                                eval_metric = "logloss", scale_pos_weight = weight, use_label_encoder =False,
                                verbosity=0, n_jobs = 30, early_stopping_rounds=20
                            )
                            
    model.fit(xtrain,ytrain, eval_set=[(xtrain,ytrain),(xval,yval)])
    model.save_model(f"/work/aavocone/models/3_0_model{estimator}_under.txt")