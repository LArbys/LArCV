import os,sys,gc

if len(sys.argv) != 3:
    print 
    print "LL_PKL   = str(sys.argv[1])"
    print "OUTDIR   = str(sys.argv[2])" 
    print 
    sys.exit(1)

LL_PKL   = str(sys.argv[1])
OUTDIR   = str(sys.argv[2]) 
NUM      = int(os.path.basename(LL_PKL).split(".")[0].split("_")[-1])

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
print "Reading LL..."
LL_df = pd.read_pickle(LL_PKL)
print "... read"

print "Maximizing @ LL_dist..."
LL_sort_df = LL_df.sort_values(["LL_dist"],ascending=False).groupby(RSE).head(1).copy()
print "... maximized"

del LL_df
gc.collect()

FOUT = os.path.join(OUTDIR,"nue_analysis_%d.root" % NUM)
tf = ROOT.TFile.Open(FOUT,"RECREATE")
print "OPEN %s"%FOUT
tf.cd()

rd = ROOTData()
tree = ROOT.TTree("nue_ana_tree","")
rd.init_tree(tree)

ix=-1
for index,row in LL_sort_df.iterrows():
    ix += 1
    rd.reset()

    rd.run[0]    = row['run']
    rd.subrun[0] = row['subrun']
    rd.event[0]  = row['event']

    print "@id=%03d @(r,s,e)=(%d,%d,%d)"%(ix,row['run'],row['subrun'],row['event'])

    if row['locv_num_vertex'] == 0:
        tree.Fill()
        print "no vertex... skip!"
        continue
        
    if np.isnan(row['LL_dist']):
        tree.Fill()
        print "invalid LL... skip!"
        continue

    if row['LL_dist'] > 0:
        rd.reco_selected[0] = 1

    rd.LL_dist[0] = row['LL_dist']
    rd.LLc_e[0]   = row['L_ec_e']
    rd.LLc_p[0]   = row['L_pc_p']
    rd.LLe_e[0]   = row['LLe']
    rd.LLe_p[0]   = row['LLp']
    
    rd.reco_X[0] = row['locv_x']
    rd.reco_Y[0] = row['locv_y']
    rd.reco_Z[0] = row['locv_z']

    tree.Fill()

tree.Write()
tf.Close()
                   




