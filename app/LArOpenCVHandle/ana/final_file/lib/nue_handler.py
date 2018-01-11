import os,sys,gc
import numpy as np
import pandas as pd

from common import *
from handler import Handler
from rootdata import ROOTData

class NueHandler(Handler):
    
    def __init__(self,tree):
        super(NueHandler,self).__init__("NueHandler")
        self.df = pd.DataFrame()
        self.no_vertex = False
        self.no_ll = False
        self.tree = tree
        self.rd = ROOTData()
        self.rd.init_nue_tree(self.tree)

    def reshape(self,inputfile) :

        #
        # read LL
        #
        print "Reading LL..."
        LL_df = pd.read_pickle(inputfile)
        print "... read"
        
        LL_sort_df = pd.DataFrame()
        
        no_vertex = False
        no_ll = False
        
        print "Check if no vertex..."
        if "locv_num_vertex" not in LL_df.columns:
            print "No vertex found in file!"
            LL_sort_df = LL_df.groupby(RSE).head(1).copy()
            no_vertex = True
            print "... handled"

        elif "LL_dist" not in LL_df.columns:
            print "Vertex exists but none valid in file!"
            LL_sort_df = LL_df.groupby(RSE).head(1).copy()
            no_ll = True
            print "... handled"
        else:
            print "Maximizing @ LL_dist..."
            LL_sort_df = LL_df.sort_values(["LL_dist"],ascending=False).groupby(RSE).head(1).copy()
            LL_sort_df.sort_values(by=RSE,inplace=True)
            print "... maximized"
    
        print "... checked"
        
        self.df = LL_sort_df.copy()
        self.no_vertex = no_vertex
        self.no_ll = no_ll

        del LL_df
        del LL_sort_df

        # ensure the rse is int64
        self.df[RSE] = self.df[RSE].astype(np.int64)

        gc.collect()

        
    def fill(self,run,subrun,event,ismc):
        self.rd.reset()

        row = self.df.query("run==@run&subrun==@subrun&event==@event")
        if row.index.size == 0: return False
        row = row.iloc[0]

        self.rd.run[0]    = int(row['run'])
        self.rd.subrun[0] = int(row['subrun'])
        self.rd.event[0]  = int(row['event'])
        
        print "@(r,s,e)=(%d,%d,%d)"%(row['run'],row['subrun'],row['event'])
        
        if ismc == True:
            # fill MC
            self.rd.true_vertex[0]  = float(row['locv_parentX']);
            self.rd.true_vertex[1]  = float(row['locv_parentY']);
            self.rd.true_vertex[2]  = float(row['locv_parentZ']);
            self.rd.selected1L1P[0] = int(row['locv_selected1L1P']);
            self.rd.nu_pdg[0]       = int(row['locv_parentPDG']);
            self.rd.true_nu_E[0]    = int(row['locv_energyInit']); 
            
            self.rd.inter_type[0] = int(row['anashr2_mcinfoInteractionType']); 
            self.rd.inter_mode[0] = int(row['anashr2_mcinfoMode']); 
            
            self.rd.true_proton_E[0]   = float(row['locv_dep_sum_proton'])
            self.rd.true_electron_E[0] = float(row['anashr2_mc_energy'])
            
            #self.rd.true_proton_dR[0] = ;
            
            self.rd.true_electron_dR[0] = float(row['anashr2_mc_dcosx']);
            self.rd.true_electron_dR[1] = float(row['anashr2_mc_dcosy']);
            self.rd.true_electron_dR[2] = float(row['anashr2_mc_dcosz']);
            
            self.rd.true_proton_theta[0]   = float(row['proton_beam_angle']);
            self.rd.true_electron_theta[0] = float(row['lepton_beam_angle']);
            
            self.rd.true_proton_phi[0]   = float(row['proton_planar_angle']);
            self.rd.true_electron_phi[0] = float(row['lepton_planar_angle']);
            
            self.rd.true_opening_angle[0] = float(row['opening_angle']);
            self.rd.true_proton_ylen[0]   = float(row['proton_yplane_len']);
            

        # fill common
        self.rd.num_croi[0]   = int(row['locv_number_croi']);
        
        if self.no_vertex == True:
            self.rd.num_vertex[0] = int(0)
            self.tree.Fill()
            self.rd.reset()
            print "empty file... skip!"
            return True        

        if row['locv_num_vertex'] == 0 or np.isnan(row['locv_num_vertex']):
            self.rd.num_vertex[0] = int(0)
            self.tree.Fill()
            self.rd.reset()
            print "no vertex... skip!"
            return True        

        self.rd.num_vertex[0] = int(row['locv_num_vertex']);

        if self.no_ll == True:
            self.tree.Fill()
            self.rd.reset()
            print "no LL vertex in file... skip!"
            return True
            
        if np.isnan(row['LL_dist']):
            self.tree.Fill()
            self.rd.reset()
            print "invalid LL... skip!"
            return True
            
        self.rd.vertex_id[0]  = int(row['vtxid']);
        self.rd.scedr[0]      = float(row['locv_scedr']);

        if ismc == True:
            self.rd.reco_mc_proton_E[0]   = float(row['reco_mc_track_energy']);
            self.rd.reco_mc_electron_E[0] = float(row['reco_mc_shower_energy']);
            self.rd.reco_mc_total_E[0]    = float(row['reco_mc_total_energy']);

        # fill LL
        if row['LL_dist'] > 0:
            self.rd.reco_selected[0] = int(1)
        else:
            self.rd.reco_selected[0] = int(0)

        self.rd.reco_vertex[0] = float(row['locv_x'])
        self.rd.reco_vertex[1] = float(row['locv_y'])
        self.rd.reco_vertex[2] = float(row['locv_z'])
        
        self.rd.LL_dist[0] = float(row['LL_dist']);
        self.rd.LLc_e[0]   = float(row['L_ec_e']);
        self.rd.LLc_p[0]   = float(row['L_pc_p']);
        self.rd.LLe_e[0]   = float(row['LLe']);
        self.rd.LLe_p[0]   = float(row['LLp']);
        
        # fill reco
        self.rd.reco_proton_E[0]   = float(row['reco_LL_proton_energy'])
        self.rd.reco_electron_E[0] = float(row['reco_LL_electron_energy']);
        self.rd.reco_total_E[0]    = float(row['reco_LL_total_energy']);

        self.tree.Fill()
        self.rd.reset()
        return True

