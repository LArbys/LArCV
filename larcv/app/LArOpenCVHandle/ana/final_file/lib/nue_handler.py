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
            
            self.rd.true_proton_E[0] = float(row['locv_dep_sum_proton'])
            self.rd.true_lepton_E[0] = float(row['anashr2_mc_energy'])
            
            self.rd.true_proton_P[0]   = float(row['proton_momentum_X'])
            self.rd.true_lepton_P[0]   = float(row['lepton_momentum_X'])

            self.rd.true_proton_P[1]   = float(row['proton_momentum_Y'])
            self.rd.true_lepton_P[1]   = float(row['lepton_momentum_Y'])

            self.rd.true_proton_P[2]   = float(row['proton_momentum_Z'])
            self.rd.true_lepton_P[2]   = float(row['lepton_momentum_Z'])            
            
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

        if ismc == True:
            self.rd.scedr[0] = float(row['locv_scedr']);

            if self.rd.scedr[0] < 5:
                self.rd.reco_close[0] = int(1)
            else:
                self.rd.reco_close[0] = int(0)
        
            if int(row['locv_vtx_on_nu']) >= 2:
                self.rd.reco_on_nu[0] = int(1)
            else:
                self.rd.reco_on_nu[0] = int(0)

        if ismc == True:
            self.rd.reco_mc_proton_E[0]  = float(row['reco_mc_track_energy']);
            self.rd.reco_mc_lepton_E[0]  = float(row['reco_mc_shower_energy']);
            self.rd.reco_mc_total_E[0]   = float(row['reco_mc_total_energy']);

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
        self.rd.reco_proton_id[0]    = int(row['reco_LL_proton_id'])
        self.rd.reco_proton_E[0]     = float(row['reco_LL_proton_energy'])
        self.rd.reco_proton_theta[0] = float(row['reco_LL_proton_theta'])
        self.rd.reco_proton_phi[0]   = float(row['reco_LL_proton_phi'])
        self.rd.reco_proton_dEdx[0]  = float(row['reco_LL_proton_dEdx'])
        self.rd.reco_proton_length[0]= float(row['reco_LL_proton_length'])
        self.rd.reco_proton_mean_pixel_dist[0] = float(row['reco_LL_proton_mean_pixel_dist'])
        self.rd.reco_proton_width[0]       = float(row['reco_LL_proton_width'])
        self.rd.reco_proton_area[0]        = float(row['reco_LL_proton_area'])
        self.rd.reco_proton_qsum[0]        = float(row['reco_LL_proton_qsum'])
        self.rd.reco_proton_shower_frac[0] = float(row['reco_LL_proton_shower_frac'])

        self.rd.reco_electron_id[0]     = int(row['reco_LL_electron_id'])
        self.rd.reco_electron_E[0]      = float(row['reco_LL_electron_energy']);
        self.rd.reco_electron_theta[0]  = float(row['reco_LL_electron_theta'])
        self.rd.reco_electron_phi[0]    = float(row['reco_LL_electron_phi'])
        self.rd.reco_electron_dEdx[0]   = float(row['reco_LL_electron_dEdx'])
        self.rd.reco_electron_length[0] = float(row['reco_LL_electron_length'])
        self.rd.reco_electron_mean_pixel_dist[0] = float(row['reco_LL_electron_mean_pixel_dist'])
        self.rd.reco_electron_width[0]       = float(row['reco_LL_electron_width'])
        self.rd.reco_electron_area[0]        = float(row['reco_LL_electron_area'])
        self.rd.reco_electron_qsum[0]        = float(row['reco_LL_electron_qsum'])
        self.rd.reco_electron_shower_frac[0] = float(row['reco_LL_electron_shower_frac'])

        self.rd.reco_total_E[0]    = float(row['reco_LL_total_energy']);

        # fill PID
        self.rd.inferred[0]      = int(1)

        for pl in xrange(3):
            self.rd.eminus_score[pl]  = float(row['anapid_eminus_score'][pl])
            self.rd.gamma_score[pl]   = float(row['anapid_gamma_score'][pl])
            self.rd.muon_score[pl]    = float(row['anapid_muon_score'][pl])
            self.rd.pion_score[pl]    = float(row['anapid_pion_score'][pl])
            self.rd.proton_score[pl]  = float(row['anapid_proton_score'][pl])

            self.rd.eminus_score_vtx[pl]  = float(row['anapid_eminus_score_vtx'][pl])
            self.rd.gamma_score_vtx[pl]   = float(row['anapid_gamma_score_vtx'][pl])
            self.rd.muon_score_vtx[pl]    = float(row['anapid_muon_score_vtx'][pl])
            self.rd.pion_score_vtx[pl]    = float(row['anapid_pion_score_vtx'][pl])
            self.rd.proton_score_vtx[pl]  = float(row['anapid_proton_score_vtx'][pl])

            self.rd.nue_perceptron[pl] = float(row['anapid2_perceptron'][pl])

        self.tree.Fill()
        self.rd.reset()

        return True

