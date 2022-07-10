import uproot
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt




#root to np array
def root_to_np(data,index):
    x = []
    for data_set in index:
        x.append(data[data_set].array(library="np"))
    return x

#manually classify background = 0 and signal = 1
def manual_classification(data, true_or_false):
    data.append(len(data[1])*[true_or_false])

def root_to_DF(List_Signal_and_Background, list_of_classification, selected_variables):
    #List_Signal_and_Background = [signal, background1, background2, ....]

    # 2 list needed to save the transformed values 
    list_root_to_np = []
    list_np_to_DF = []

    #define dtypes for creating DF. The classification part converts all dtypes from float to object
    #probably because it is the only columns with objects
    define_dtype = ["float64"]*len(selected_variables) 

    #root to numpy
    for value in range(len(List_Signal_and_Background)):
        list_root_to_np.append(root_to_np(List_Signal_and_Background[value],selected_variables))
    print("Convert to np.array successfull.")



    #classification
    print(f"starting to create DF[classification] with values: {list_of_classification}")
    define_dtype += ["object"]

 

    selected_variables.append("classification")
    for i,v in enumerate(list_root_to_np):
        manual_classification(v,list_of_classification[i])
   

        


    #get elements from list_root_to_np, transform them to a DF and save it in list_np_to_DF
    for ii,vv in enumerate(list_root_to_np):
        list_np_to_DF.append(pd.DataFrame(np.transpose(vv), columns = selected_variables))

    dataframe = pd.concat(list_np_to_DF)
    print("Convert to pd.DF successfull.")
    print("All dtypes are now objects.")
    dataframe.index = range(0,len(dataframe))


    #change dtypes
    for n,c in enumerate(dataframe.columns):
        dataframe[c] = dataframe[c].astype(define_dtype[n])
    print("Change dtypes for every column successfull.")

    #MC truth and (signal,background) = (1,0)
    dataframe["signal"] = dataframe["classification"].apply(lambda x: 1 if x=="data" else 0)
    print("Classification with singal = 1 and background = 0 successfully implemented.")
    dataframe.drop(dataframe[(dataframe["B_sig_isSignalAcceptMissingNeutrino"]==0.0)&(dataframe["signal"] == 1)].index, inplace = True)
    print("MC truth matching completed.")



    print("Proccess completed.")

    return dataframe

def efficiency(yprob,ytest):
    bin_edges = np.linspace(0,1,101)
    s_eff = []
    b_eff = []

#----------------SIGNAL..............................................
    #hist 
    s_hist= yprob*ytest
    s_hist = s_hist[s_hist!=0]      #overwrites s_hist with an array with no 0 values
    counts,_ = np.histogram(s_hist,bins = bin_edges)
    for i in range(len(bin_edges)):
        s_eff.append(sum(counts[i:])/sum(counts))
    s_eff = np.array(s_eff)


#----------------BACKGROUND.............................................
    #hist 
    b_hist = yprob*(1-ytest)
    b_hist = b_hist[b_hist != 0]     #removes the 0 values

    #efficiency
    counts,_ = np.histogram(b_hist,bins = bin_edges)
    
    for i in range(len(bin_edges)):
        b_eff.append(sum(counts[i:])/sum(counts))
    b_eff = np.array(b_eff)
    return s_hist, b_hist,s_eff, b_eff, bin_edges

def hist_sig_back(bhist,shist,bin_edges,save_path):
    plt.figure(figsize=(9,6))
    plt.hist(bhist,bins = bin_edges, density=True, histtype="step", label = "Background", color = "g")
    plt.hist(shist,bins = bin_edges, density=True, histtype="step", label = "signal", color = "r", alpha = 0.8)
    plt.title(f"normalized histogram with n = {estimator} trees \nsample weight {round(pos_weight,2)} \nsample size = large")
    plt.ylabel("Entries / ({:.2f} unit)".format(bin_edges[1]-bin_edges[0]), fontsize = 15)
    plt.xlabel("xgb probability", fontsize = 20)
    plt.legend()
    if save_path != 0:
        plt.savefig(f"{save_path}.jpeg")
    else:
        pass
    plt.show() 

def hist_overtrain(bhist,bhist_train,shist,shist_train,estimator,bin_edges,save_path):
    
    #Test for overtraining
    plt.figure(figsize=(9,6))
    plt.hist(bhist_train,bins = bin_edges, density=True, histtype="step", label = "train set", color = "g")
    plt.hist(bhist,bins = bin_edges, density=True, histtype="step", label = "test set", color = "r", alpha = 0.5)
    plt.title(f"normalized background histogram for {estimator}\nsample size = large")
    plt.ylabel("Entries / ({:.2f} unit)".format(bin_edges[1]-bin_edges[0]), fontsize = 15)
    plt.xlabel("xgb probability", fontsize = 20)
    plt.legend()
    if save_path != 0:
        plt.savefig(f"{save_path}_background.jpeg")
    else:
        pass
    plt.show() 

    #signal
    plt.figure(figsize=(9,6))
    plt.hist(shist_train,bins = bin_edges, density=True, histtype="step", label = "train set", color = "g")
    plt.hist(shist,bins = bin_edges, density=True, histtype="step", label = "test set", color = "r", alpha = 0.5)
    plt.title(f"normalized signal histogram with n = {estimator} trees \nsample weight {round(pos_weight,2)} \nsample size = large")
    plt.ylabel("Entries / ({:.2f} unit)".format(bin_edges[1]-bin_edges[0]), fontsize = 15)
    plt.xlabel("xgb probability", fontsize = 20)
    plt.legend()
    if save_path != 0:
        plt.savefig(f"{save_path}_signal.jpeg")
    else:
        pass
    plt.show()

def hist_eff(beff,seff,shist,save_path):
    plt.figure(figsize=(9,6))
    plt.plot(bin_edges, 1-beff, label = "$1-\epsilon_B$", color ="g")
    plt.xlabel("xgb probability", fontsize = 20)
    plt.ylabel("background rejection", fontsize = 15)
    plt.title(f"background rejection", fontsize = 15)
    plt.legend()
    if save_path != 0:
        plt.savefig(f"{save_path}_background.jpeg")
    else:
        pass
    plt.show() 

    plt.figure(figsize=(9,6))
    plt.plot(bin_edges, seff, label = "$\epsilon_S$", color = "r")
    plt.xlabel("xgb probability", fontsize = 20)
    plt.ylabel("signal efficiency",fontsize=15)
    plt.title(f"signal efficiency\nsample size = large", fontsize = 15)
    plt.legend()
    if save_path != 0:
        plt.savefig(f"{save_path}_signal.jpeg")
    else:
        pass
    plt.show() 

def hist_roc(yprob):
    FP1,TP1,t1 = roc_curve(ytest, yprob[:,0], pos_label=0) 
    FP2,TP2,t2 = roc_curve(ytest, yprob[:,1], pos_label=1)

    plt.figure(figsize=(9,6))
    plt.plot(FP1, TP1)
    plt.title(f"ROC background with n = {estimator}\nsample size = large", fontsize = 15)
    plt.xlabel("FP rate ", fontsize = 20)
    plt.ylabel("TP rate ", fontsize = 15)
    if save_path != 0:
        plt.savefig(f"{save_path}_background.jpeg")
    else:
        pass
    plt.show()

    #ROC for signal
    plt.figure(figsize=(9,6))
    plt.plot(FP2, TP2, label = "signal")
    plt.title(f"ROC signal with n = {estimator}\nsample size = large", fontsize = 15)
    plt.xlabel("FP rate", fontsize = 20)
    plt.ylabel("TP rate", fontsize = 15)
    if save_path != 0:
        plt.savefig(f"{save_path}_signal.jpeg")
    else:
        pass
    plt.show()