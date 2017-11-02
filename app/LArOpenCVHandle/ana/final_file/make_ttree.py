import os,sys,gc

if len(sys.argv) != 5:
    print 
    print "LL_PKL   = str(sys.argv[1])"
    print "SHR_ROOT = str(sys.argv[2])"
    print "TRK_ROOT = str(sys.argv[3])"
    print "OUTDIR   = str(sys.argv[4])" 
    print 
    sys.exit(1)

LL_PKL   = str(sys.argv[1])
SHR_ROOT = str(sys.argv[2])
TRK_ROOT = str(sys.argv[3])
OUTDIR   = str(sys.argv[4]) 
NUM      = int(os.path.basename(SHR_ROOT).split(".")[0].split("_")[-1])
RSE = ['run','subrun','event']


BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

import ROOT
import numpy as np
import root_numpy as rn
import pandas as pd
from rootdata import ROOTData

#
# read LL
#
LL_df = pd.read_pickle(LL_PKL)

#
# read SHR_ROOT
#
SHR_df = pd.DataFrame(rn.root2array(SHR_ROOT,treename="fShowerTree_nueshowers"))
TRK_df = pd.DataFrame(rn.root2array(TRK_ROOT,treename="_recoTree"))

#
# combine
#
LL_df.reset_index(inplace=True)
SHR_df.reset_index(inplace=True)
TRK_df.reset_index(inplace=True)

LL_df.set_index(RSE,inplace=True)
SHR_df.set_index(RSE,inplace=True)
TRK_df.set_index(RSE,inplace=True)

comb_df = pd.concat([LL_df,SHR_df,TRK_df],axis=1,join_axes=[LL_df.index])

FOUT = os.path.join(OUTDIR,"nue_analysis_%d.root" % NUM)
tf = ROOT.TFile.Open(FOUT,"RECREATE")
print "OPEN %s"%FOUT
tf.cd()

rd = ROOTData()
tree = ROOT.TTree("nue_ana_tree","")
rd.init_tree(tree)

for index,row in comb_df.iterrows():
    rd.reset()

    print "@(r,s,e)=",index

    rd.run[0]    = index[0]
    rd.subrun[0] = index[1]
    rd.event[0]  = index[2]

    rd.LL[0] = row['LL']
    
    #
    # shower vars
    #
    energyU = row['reco_energy_U']
    energyV = row['reco_energy_V']
    energyY = row['reco_energy_Y']
    
    if energyY>0.00001:
        rd.reco_shower_E[0] = energyY
    else:
        rd.reco_shower_E[0] = (energyU + energyV) / 2.0
    print row['reco_dedx_Y']
    rd.reco_shower_dEdx[0] = row['reco_dedx_Y']
    rd.reco_shower_dX[0]   = row['mc_dcosx']
    rd.reco_shower_dY[0]   = row['mc_dcosy']
    rd.reco_shower_dZ[0]   = row['mc_dcosz']
    rd.reco_shower_good[0] = 0

    #
    # track vars
    #

    if row["E_proton_v"].values.size > 0: 
        rd.reco_track_E_p[0]  = row['E_proton_v'].values[0]

    if row["E_muon_v"].values.size > 0:
        rd.reco_track_E_m[0]  = row['E_muon_v'].values[0]

    if row['Length_v'].values.size > 0:
        rd.reco_track_len[0]  = row['Length_v'].values[0]

    if row['Avg_Ion_v'].values.size > 0:
        rd.reco_track_ion[0]  = row['Avg_Ion_v'].values[0]

    rd.reco_track_good[0] = row['GoodVertex']

    #
    # position
    #
    
    rd.reco_X[0] = row['x']
    rd.reco_Y[0] = row['y']
    rd.reco_Z[0] = row['z']

    tree.Fill()

tree.Write()
tf.Close()
                   




