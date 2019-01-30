import os, sys, gc
import numpy as np
import pandas as pd

rse   = ['run','subrun','event']
rsev  = ['run','subrun','event','vtxid']
rsec  = ['run','subrun','event','cvtxid']
rserv = ['run','subrun','event','roid','vtxid']

RSE   = list(rse)
RSEV  = list(rsev)
RSEVC = list(rsec)
RSERV = list(rserv)

#
# Drop if column name ends in _y
#
def drop_q(df):
    to_drop = [x for x in df if x.endswith('_q')]
    df.drop(to_drop, axis=1, inplace=True)

#
# Define, given value, how to get prob. from PDF
#
def nearest_id(spectrum,value):
    return np.argmin(np.abs(spectrum - value))

def nearest_id_v(spectrum_v,value_v):
    return np.array([np.argmin(np.abs(spectrum[0] - value)) for spectrum, value in zip(spectrum_v,value_v)])

#
# print whole data frame out
#
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')
    

#
# concat dataframe
#
def concat_pkls(flist_v,OUT_NAME="out"):
    df_v = []
    row_ctr = 0

    for ix,f in enumerate(flist_v):
        try:
            df = pd.read_pickle(f)
        except:
            print "\033[91m" + "ERROR @file=%s" % os.path.basename(f) + "\033[0m"
            continue
                    
        df_v.append(df)

        row_ctr += int(df_v[-1].index.size)
        print "\r(%02d/%02d)... read %d rows" % (ix,len(flist_v),row_ctr),
        sys.stdout.flush()
        
    if len(df_v)==0: 
        print "nothing to see..."
        return False
    
    df = pd.concat(df_v,ignore_index=True)
    print "...concat"
    SS = OUT_NAME + ".pkl"
    df.to_pickle(SS)
    dsk  = os.path.getsize(SS)
    dsk /= (1024.0 * 1024.0)
    print "...saved {:.02f} MB".format(dsk)
    del df
    gc.collect()
    print "...reaped"
    return True

kUnknown                   =   0
kCCQE                      =   1
kNCQE                      =   2
kResCCNuProtonPiPlus       =   3
kResCCNuNeutronPi0         =   4
kResCCNuNeutronPiPlus      =   5
kResNCNuProtonPi0          =   6
kResNCNuProtonPiPlus       =   7
kResNCNuNeutronPi0         =   8
kResNCNuNeutronPiMinus     =   9
kResCCNuBarNeutronPiMinus  =   10
kResCCNuBarProtonPi0       =   11
kResCCNuBarProtonPiMinus   =   12
kResNCNuBarProtonPi0       =   13
kResNCNuBarProtonPiPlus    =   14
kResNCNuBarNeutronPi0      =   15
kResNCNuBarNeutronPiMinus  =   16
kResCCNuDeltaPlusPiPlus    =   17
kResCCNuDelta2PlusPiMinus  =   21
kResCCNuBarDelta0PiMinus   =   28
kResCCNuBarDeltaMinusPiPlus=   32
kResCCNuProtonRhoPlus      =   39
kResCCNuNeutronRhoPlus     =   41
kResCCNuBarNeutronRhoMinus =   46
kResCCNuBarNeutronRho0     =   48
kResCCNuSigmaPlusKaonPlus  =   53
kResCCNuSigmaPlusKaon0     =   55
kResCCNuBarSigmaMinusKaon0 =   60
kResCCNuBarSigma0Kaon0     =   62
kResCCNuProtonEta          =   67
kResCCNuBarNeutronEta      =   70
kResCCNuKaonPlusLambda0    =   73
kResCCNuBarKaon0Lambda0    =   76
kResCCNuProtonPiPlusPiMinus=   79
kResCCNuProtonPi0Pi0       =   80
kResCCNuBarNeutronPiPlusPiMinus =   85
kResCCNuBarNeutronPi0Pi0   =   86
kResCCNuBarProtonPi0Pi0    =   90
kCCDIS                     =   91
kNCDIS                     =   92
kUnUsed1                   =   93
kUnUsed2                   =   94
kCCQEHyperon               =   95
kNCCOH                     =   96
kCCCOH                     =   97
kNuElectronElastic         =   98
kInverseMuDecay            =   99

mode = {}
mode[kUnknown]                    = "kMEC"
mode[kCCQE]                       = "kCCQE"
mode[kNCQE]                       = "kNCQE"
mode[kResCCNuProtonPiPlus]        = "kResCCNuProtonPiPlus"
mode[kResCCNuNeutronPi0]          = "kResCCNuNeutronPi0"
mode[kResCCNuNeutronPiPlus]       = "kResCCNuNeutronPiPlus"
mode[kResNCNuProtonPi0]           = "kResNCNuProtonPi0"
mode[kResNCNuProtonPiPlus]        = "kResNCNuProtonPiPlus"
mode[kResNCNuNeutronPi0]          = "kResNCNuNeutronPi0"
mode[kResNCNuNeutronPiMinus]      = "kResNCNuNeutronPiMinus"
mode[kResCCNuBarNeutronPiMinus]   = "kResCCNuBarNeutronPiMinus"
mode[kResCCNuBarProtonPi0]        = "kResCCNuBarProtonPi0"
mode[kResCCNuBarProtonPiMinus]    = "kResCCNuBarProtonPiMinus"
mode[kResNCNuBarProtonPi0]        = "kResNCNuBarProtonPi0"
mode[kResNCNuBarProtonPiPlus]     = "kResNCNuBarProtonPiPlus"
mode[kResNCNuBarNeutronPi0]       = "kResNCNuBarNeutronPi0"
mode[kResNCNuBarNeutronPiMinus]   = "kResNCNuBarNeutronPiMinus"
mode[kResCCNuDeltaPlusPiPlus]     = "kResCCNuDeltaPlusPiPlus"
mode[kResCCNuDelta2PlusPiMinus]   = "kResCCNuDelta2PlusPiMinus"
mode[kResCCNuBarDelta0PiMinus]    = "kResCCNuBarDelta0PiMinus"
mode[kResCCNuBarDeltaMinusPiPlus] = "kResCCNuBarDeltaMinusPiPlus"
mode[kResCCNuProtonRhoPlus]       = "kResCCNuProtonRhoPlus"
mode[kResCCNuNeutronRhoPlus]      = "kResCCNuNeutronRhoPlus"
mode[kResCCNuBarNeutronRhoMinus]  = "kResCCNuBarNeutronRhoMinus"
mode[kResCCNuBarNeutronRho0]      = "kResCCNuBarNeutronRho0"
mode[kResCCNuSigmaPlusKaonPlus]   = "kResCCNuSigmaPlusKaonPlus"
mode[kResCCNuSigmaPlusKaon0]      = "kResCCNuSigmaPlusKaon0"
mode[kResCCNuBarSigmaMinusKaon0]  = "kResCCNuBarSigmaMinusKaon0"
mode[kResCCNuBarSigma0Kaon0]      = "kResCCNuBarSigma0Kaon0"
mode[kResCCNuProtonEta]           = "kResCCNuProtonEta"
mode[kResCCNuBarNeutronEta]       = "kResCCNuBarNeutronEta"
mode[kResCCNuKaonPlusLambda0]     = "kResCCNuKaonPlusLambda0"
mode[kResCCNuBarKaon0Lambda0]     = "kResCCNuBarKaon0Lambda0"
mode[kResCCNuProtonPiPlusPiMinus] = "kResCCNuProtonPiPlusPiMinus"
mode[kResCCNuProtonPi0Pi0]        = "kResCCNuProtonPi0Pi0"
mode[kResCCNuBarNeutronPiPlusPiMinus]  = "kResCCNuBarNeutronPiPlusPiMinus"
mode[kResCCNuBarNeutronPi0Pi0]    = "kResCCNuBarNeutronPi0Pi0"
mode[kResCCNuBarProtonPi0Pi0]     = "kResCCNuBarProtonPi0Pi0"
mode[kCCDIS]                      = "kCCDIS"
mode[kNCDIS]                      = "kNCDIS"
mode[kUnUsed1]                    = "kUnUsed1"
mode[kUnUsed2]                    = "kUnUsed2"
mode[kCCQEHyperon]                = "kCCQEHyperon"
mode[kNCCOH]                      = "kNCCOH"
mode[kCCCOH]                      = "kCCCOH"
mode[kNuElectronElastic]          = "kNuElectronElastic"
mode[kInverseMuDecay]             = "kInverseMuDecay"
