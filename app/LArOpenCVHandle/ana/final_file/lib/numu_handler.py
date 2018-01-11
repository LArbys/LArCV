import os,sys,gc
import numpy as np
import pandas as pd

from common import *
from handler import Handler
from rootdata import ROOTData

class NumuHandler(Handler):
    
    def __init__(self,tree):
        super(NumuHandler,self).__init__("NumuHandler")
        self.df = pd.DataFrame()
        self.no_vertex = False
        self.no_ll = False
        self.tree = tree
        self.rd = ROOTData()
        self.rd.init_numu_tree(self.tree)

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

        elif "numu_CosmicLL" not in LL_df.columns:
            print "Vertex exists but none valid in file!"
            LL_sort_df = LL_df.groupby(RSE).head(1).copy()
            no_ll = True
            print "... handled"
        else:
            print "Maximizing @ numu_CosmicLL..."
            LL_sort_df = LL_df.sort_values(["numu_CosmicLL"],ascending=False).groupby(RSE).head(1).copy()
            LL_sort_df.sort_values(by=RSE,inplace=True)
            print "... maximized"
    
        print "... checked"
        
        self.df = LL_sort_df.copy()
        self.no_vertex = no_vertex
        self.no_ll = no_ll

        del LL_df
        del LL_sort_df

        gc.collect()

    def fill(self,run,subrun,event,ismc):
        self.rd.reset()
        
        row = self.df.query("run==@run&subrun==@subrun&event==@event")
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
            
            self.rd.true_proton_E[0]   = float(row['locv_dep_sum_proton'])
            
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

        if np.isnan(row['numu_CosmicLL']):
            self.tree.Fill()
            self.rd.reset()
            print "invalid LL... skip!"
            return True

        self.rd.vertex_id[0] = int(row['vtxid']);
        self.rd.scedr[0]     = float(row['locv_scedr']);

        # fill LL
        if row['numu_CosmicLL'] > 0:
            self.rd.reco_selected[0] = int(1)
        else:
            self.rd.reco_selected[0] = int(0)
        
        self.rd.reco_vertex[0] = float(row['locv_x'])
        self.rd.reco_vertex[1] = float(row['locv_y'])
        self.rd.reco_vertex[2] = float(row['locv_z'])
        
        self.rd.CosmicLL[0]      = float(row["numu_CosmicLL"])
        self.rd.NuBkgLL[0]       = float(row["numu_NuBkgLL"])
        self.rd.PassCuts[0]      = int(row["numu_PassCuts"])
        self.rd.VtxAlgo[0]       = int(row["numu_VtxAlgo"])
        self.rd.NTracks[0]       = int(row["numu_NTracks"])
        self.rd.N5cmTracks[0]    = int(row["numu_N5cmTracks"])
        self.rd.InFiducial[0]    = int(row["numu_InFiducial"])
        self.rd.Good3DReco[0]    = int(row["numu_Good3DReco"])
        self.rd.AnythingRecod[0] = int(row["numu_AnythingRecod"])

        self.tree.Fill()
        self.rd.reset()
        return True



