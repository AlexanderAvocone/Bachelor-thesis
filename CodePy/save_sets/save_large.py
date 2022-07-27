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
    print("Test charged")
    charged     = uproot.open('/ceph/aavocone/Data/kplus_v34_kshort_v34_100invfb_test_nobdtcut_16812.root:tree_Bsig;1').arrays(index, library ="pd")
    print("Test mixed")
    mixed       = uproot.open('/ceph/aavocone/Data/kplus_v34_kshort_v34_100invfb_test_nobdtcut_16817.root:tree_Bsig;1').arrays(index, library ="pd")
    print("Test uu")
    uu          = uproot.open('/ceph/aavocone/Data/kplus_v34_kshort_v34_100invfb_test_nobdtcut_16802.root:tree_Bsig;1').arrays(index, library ="pd")
    print("Test cc")
    cc          = uproot.open('/ceph/aavocone/Data/kplus_v34_kshort_v34_100invfb_test_nobdtcut_16792.root:tree_Bsig;1').arrays(index, library ="pd")
    print("Test dd")
    dd          = uproot.open('/ceph/aavocone/Data/kplus_v34_kshort_v34_100invfb_test_nobdtcut_16797.root:tree_Bsig;1').arrays(index, library ="pd")
    print("Test ss")
    ss          = uproot.open('/ceph/aavocone/Data/kplus_v34_kshort_v34_100invfb_test_nobdtcut_16807.root:tree_Bsig;1').arrays(index, library ="pd")
    print("Test create df[signal] and df[class]")

    sets =[data,charged,mixed,uu,cc,dd,ss]
    l=0
    for i,v in enumerate(sets):
        if i==0:
            v["signal"]  = np.ones(len(v))
            
        else:
            v["signal"] = np.zeros(len(v))
        v["class"] = np.ones(len(v))*i
        l += len(v)
    data.drop(data[data["B_sig_isSignalAcceptMissingNeutrino"]==0.0].index, inplace = True)
    df = pd.concat(sets)
    print("Test save to hdf5")
    df.to_hdf("/ceph/aavocone/large_set.h5", key = "df", mode="w")