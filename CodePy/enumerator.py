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


if __name__ == "__main__":


    print("Reading files:")
    data        = pd.read_hdf("/ceph/aavocone/Datasets/signal3_large.h5", "df")
    charged     = pd.read_hdf("/ceph/aavocone/Datasets/charged_large.h5", "df")
    mixed       = pd.read_hdf("/ceph/aavocone/Datasets/mixed_large.h5", "df")
    uu          = pd.read_hdf("/ceph/aavocone/Datasets/uu_large.h5", "df")
    cc          = pd.read_hdf("/ceph/aavocone/Datasets/cc_large.h5", "df")
    dd          = pd.read_hdf("/ceph/aavocone/Datasets/dd_large.h5", "df")
    ss          = pd.read_hdf("/ceph/aavocone/Datasets/ss_large.h5", "df")
    print("completed!\n")
    sets =[data,charged,mixed,uu,cc,dd,ss]
    df = pd.concat(sets)


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


    model500 = xgb.XGBClassifier()
    model500.load_model("/work/aavocone/models/model500.txt")

    for estimator in [600, 700, 800, 900]:
        print(f"test{estimator}\n")

        model     = XGBClassifier(  n_estimators = estimator, learning_rate = 0.2,
                                    eval_metric = "logloss", scale_pos_weight = weight, use_label_encoder =False,
                                    verbosity=0, n_jobs = 40
                                )
                                
        model.fit(xtrain,ytrain, eval_set=[(xval,yval)], xgb_model=model500)
        model.save_model(f"/ceph/aavocone/models/3_0_model{estimator}.txt")
    

