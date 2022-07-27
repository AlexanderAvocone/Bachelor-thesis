print("Import uproot:")
import uproot
print("completed!\n")
print("Import uproot:")
import numpy as np
print("completed!\n")
print("Import uproot:")
import pandas as pd
print("completed!\n")

if __name__ == "__main__":
        
    print("Test index")
    index       = [ "B_sig_K_dr","B_sig_K_dz","B_sig_CleoConeCS_3_ROE", "thrustAxisCosTheta","aplanarity","sphericity",
                    "harmonicMomentThrust0","harmonicMomentThrust1","harmonicMomentThrust2","harmonicMomentThrust3","harmonicMomentThrust4",
                    "foxWolframR1","foxWolframR2","foxWolframR3","foxWolframR4", "B_sig_isSignalAcceptMissingNeutrino"]
    print("Test data")
    data        = uproot.open("/ceph/aavocone/Data/processed_simulation_B_K_a_nunu_ma_3_GeV_100000_events_nobdtcut.root:tree_Bsig;1").arrays(index, library ="pd")
    data["signal"]  = np.ones(len(data))
    data["class"]   = np.ones(len(data))
    data.drop(data[data["B_sig_isSignalAcceptMissingNeutrino"]==0.0].index, inplace = True)
    data.drop("B_sig_isSignalAcceptMissingNeutrino", axis=1, inplace= True)

    print("Test save signal to hdf5")
    data.to_hdf("/ceph/aavocone/Datasets/signal3_large.h5", key = "df", mode="w")
