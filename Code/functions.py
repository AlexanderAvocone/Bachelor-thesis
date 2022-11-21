import uproot
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq


def efficiency(yprob, ytest):
    bin_edges = np.linspace(0,1,101)
    s_eff = []
    b_eff = []

    #----------------SIGNAL..............................................
    #hist 
    s_hist= yprob*ytest
    s_hist = s_hist[s_hist!=0]      #overwrites s_hist with an array with no 0 values
    counts,_ = np.histogram(s_hist,bins = bin_edges)

    for i in range(len(bin_edges)):
        s_eff.append(sum(counts[i:])/33333)
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

def stacked_hist(original_df,yprob,ytest,save_path="/work/aavocone/stacked_hist.pdf",binning=np.linspace(0,1,101)):
    #----------------Explenation----------------------------------------------------------------------------------------
    # 1.) getting the background hist, copied from efficiency()
    # 2.) turn histogram into DF to use pd.cut(target_to_be_binned , bin_edges, labels) for binning the histogram into the 
    #     background types 
    # 3.) use df.index for the target_to_be_binned
    # 4.) use df["class"].value_counts() to get the length of each background type
    # 5.) use .sort_index to get a sorted list arranged like the load-in (signal,charged,mixed,uu,...)
    # 6.) create bin__edges by summing up the length of the background types, start with bin_edges = [-1] because
    #     pd.cut() starts with the binning at (bin_edges + 1)

    #get background histogram
    labels = ["signal","charged","mixed","uu","cc","dd","ss"]
    colors = ("darkblue","slateblue","mediumorchid","indianred","goldenrod","mediumseagreen") #try cubehelix



    background_hist= yprob*(1-ytest)                        #yprob * 0 = 0 ---> 1-ytest to turn background 0 to 1
    background_hist = background_hist[background_hist!=0]   #new list with out 0 values 


    #creating df to use pd.cut() 
    background_hist = pd.DataFrame({"signal":background_hist})

    #getting the lengths and summing up for bin_edges
    length_of_bkg = original_df["class"].value_counts().sort_index()
    bin_edges = [-1]                                            
    for i in length_of_bkg:
        bin_edges.append(i + max(bin_edges))
    labels = ["signal","charged","mixed","uu","cc","dd","ss"]


    #pd.cut()
    background_hist["type of bkg"] =  pd.cut(background_hist.index, bin_edges,labels=labels)

    # get the different histograms
    stacked_hist = []
    for i in labels[1:]:
        stacked_hist.append(background_hist.signal[background_hist["type of bkg"]==i])
    #plot histogram
    plt.figure(figsize=(9,6))
    plt.hist(stacked_hist, stacked = True, density=True, bins = binning, label = labels[1:], alpha = 0.7, color = colors)
    plt.xlabel("xgboost probability", fontsize=20)
    plt.ylabel("Entries / ({:.2f} unit)".format(binning[1]-binning[0]), fontsize = 15)
    plt.title("Normalized stacked background histogram for n = 500", fontsize = 15)
    plt.legend()
    plt.savefig(save_path, format="pdf",bbox_inches="tight")
    plt.show()

def PFOM(seff,beff,bhist,n):
    return seff/(np.sqrt(beff*sum(bhist))+n/2)

def load_parquet(data_path_as_parquet):
    return pq.read_table(data_path_as_parquet).to_pandas()



    
    
