import os,sys,gc

if len(sys.argv) != 4:
    print 
    print "LL_PKL = str(sys.argv[1])"
    print "IS_MC  = bool(sys.argv[2])"
    print "OUTDIR = str(sys.argv[3])" 
    print 
    sys.exit(1)

LL_PKL   = str(sys.argv[1])
IS_MC    = bool(sys.argv[2])
OUTDIR   = str(sys.argv[3]) 
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

    if IS_MC == True:
        # fill MC
        rd.true_vertex[0]  = float(row['locv_parentX']);
        rd.true_vertex[1]  = float(row['locv_parentY']);
        rd.true_vertex[2]  = float(row['locv_parentZ']);
        rd.selected1L1P[0] = int(row['locv_selected1L1P']);
        rd.nu_pdg[0]       = int(row['locv_parentPDG']);
        rd.true_nu_E[0]    = int(row['locv_energyInit']); 
        
        rd.inter_type[0] = int(row['anashr2_mcinfoInteractionType']); 
        rd.inter_mode[0] = int(row['anashr2_mcinfoMode']); 
        
        rd.true_proton_E[0]   = float(row['locv_dep_sum_proton'])
        rd.true_electron_E[0] = float(row['anashr2_mc_energy'])
    
        #rd.true_proton_dR[0] = ;
        
        rd.true_electron_dR[0] = float(row['anashr2_mc_dcosx']);
        rd.true_electron_dR[1] = float(row['anashr2_mc_dcosy']);
        rd.true_electron_dR[2] = float(row['anashr2_mc_dcosz']);
        
        rd.true_proton_theta[0]   = float(row['proton_beam_angle']);
        rd.true_electron_theta[0] = float(row['lepton_beam_angle']);

        rd.true_proton_phi[0]   = float(row['proton_planar_angle']);
        rd.true_electron_phi[0] = float(row['lepton_planar_angle']);
        
        rd.true_opening_angle[0] = float(row['opening_angle']);
        rd.true_proton_ylen[0]   = float(row['proton_yplane_len']);
        

    # fill common
    rd.num_croi[0]   = int(row['locv_number_croi']);

    if row['locv_num_vertex'] == 0 or np.isnan(row['locv_num_vertex']):
        rd.num_vertex[0] = int(0)
        tree.Fill()
        print "no vertex... skip!"
        continue


    rd.num_vertex[0] = int(row['locv_num_vertex']);
        
    if np.isnan(row['LL_dist']):
        tree.Fill()
        print "invalid LL... skip!"
        continue

    rd.vertex_id[0]  = int(row['vtxid']);

    rd.scedr[0]      = float(row['locv_scedr']);

    if IS_MC == True:
        rd.reco_mc_proton_E[0]   = float(row['reco_mc_track_energy']);
        rd.reco_mc_electron_E[0] = float(row['reco_mc_shower_energy']);
        rd.reco_mc_total_E[0]    = float(row['reco_mc_total_energy']);

    # fill LL
    if row['LL_dist'] > 0:
        rd.reco_selected[0] = int(1)
    else:
        rd.reco_selected[0] = int(0)
        
    rd.LL_dist[0] = float(row['LL_dist']);
    rd.LLc_e[0]   = float(row['L_ec_e']);
    rd.LLc_p[0]   = float(row['L_pc_p']);
    rd.LLe_e[0]   = float(row['LLe']);
    rd.LLe_p[0]   = float(row['LLp']);

    # fill reco
    rd.reco_proton_E[0]     = float(row['reco_LL_proton_energy'])
    #self.reco_proton_len[0]  = ;
    #self.reco_proton_ion[0]  = ;
    #self.reco_proton_good[0] = ;

    rd.reco_electron_E[0]     = float(row['reco_LL_electron_energy']);
    #self.reco_electron_dEdx[0] = ;
    #self.reco_electron_dR[0]   = ;

    rd.reco_total_E[0] = float(row['reco_LL_total_energy']);

    rd.reco_vertex[0] = float(row['locv_x'])
    rd.reco_vertex[1] = float(row['locv_y'])
    rd.reco_vertex[2] = float(row['locv_z'])

    tree.Fill()

tree.Write()
tf.Close()



