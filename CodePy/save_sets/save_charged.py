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
                    "foxWolframR1","foxWolframR2","foxWolframR3","foxWolframR4"]
    print("Test charged")
    charged     = uproot.open('/ceph/aavocone/Data/kplus_v34_kshort_v34_100invfb_test_nobdtcut_16812.root:tree_Bsig;1').arrays(index, library ="pd")
    charged["signal"]  = np.zeros(len(charged))
  
    

    print("Test save carged to hdf5")
    charged.to_hdf("/ceph/aavocone/Datasets/charged_large.h5", key = "df", mode="w")
