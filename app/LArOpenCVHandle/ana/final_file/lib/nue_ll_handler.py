import os,sys,gc
import numpy as np
import pandas as pd
import root_numpy as rn

from common import *
from combine import *
from handler import Handler
from rootdata import ROOTData

class NueLLHandler(Handler):
    
    def __init__(self,tree):
        super(NueLLHandler,self).__init__("NueLLHandler")
        self.df = pd.DataFrame()
        self.no_vertex = False
        self.no_pass = False
        self.tree = tree
        self.rd = ROOTData()
        self.rd.init_nue_ll_tree(self.tree)

    def reshape(self,inputfile,mcinfo) :

        print
        print "--> @nue_ll_handler reshape"

        df = pd.DataFrame()
        df3 = pd.DataFrame()
        df_sort = pd.DataFrame()

        print "inputfile=",inputfile
        df = pd.read_pickle(inputfile)

        print "Check if no vertex..."
        if "locv_num_vertex" not in df.columns:
            print "No vertex found in file!"
            df_sort = df.groupby(RSE).head(1).copy()
            df_sort.sort_values(by=RSE,inplace=True)
            self.df = df_sort.copy()
            self.df[RSE] = self.df[RSE].astype(np.int64)
            self.no_vertex = True
            del df
            del df3
            del df_sort
            gc.collect()
            return

        # add in the MC information
        if mcinfo is not None:
            df3 = pd.DataFrame(rn.root2array(mcinfo,treename="EventMCINFO_DL"))
            df3 = df3[['run','subrun','event','parentSCEX','parentSCEY','parentSCEZ']]
            df3.set_index(RSE,inplace=True)
            df3 = df3.add_prefix("mcinfo_")
            df.set_index(RSE,inplace=True)

            df = df.join(df3)

            df.reset_index(inplace=True)

            df["mcinfo_scedr"] = df.apply(bless_nue_scedr,axis=1)

        print "Sort by ll_selected ..."
        df_sort = df.sort_values(["ll_selected"],ascending=False).groupby(RSE).head(1).copy()
        df_sort.sort_values(by=RSE,inplace=True)
        print "... done"
    
        self.df = df_sort.copy()
        self.df[RSE] = self.df[RSE].astype(np.int64)

        del df
        del df3
        del df_sort

        gc.collect()
        print "--> done"
        print 
        return

    def fill(self,run,subrun,event,ismc):
        self.rd.reset()

        row = self.df.query("run==@run&subrun==@subrun&event==@event")
        if row.index.size == 0: 
            return False

        row = row.iloc[0]

        self.rd.run[0]    = int(row['run'])
        self.rd.subrun[0] = int(row['subrun'])
        self.rd.event[0]  = int(row['event'])
        
        print "@(r,s,e)=(%d,%d,%d)"%(row['run'],row['subrun'],row['event'])
        
        # fill common
        self.rd.num_croi[0] = int(row['locv_number_croi']);
        self.rd.num_vertex[0] = int(0)
        
        if self.no_vertex == True:
            self.tree.Fill()
            print "--> empty file"
            return True        

        if row['locv_num_vertex'] == 0 or np.isnan(row['locv_num_vertex']):
            self.tree.Fill()
            print "--> no vertex"
            return True        

        self.rd.num_vertex[0] = int(row['locv_num_vertex']);
        self.rd.vertex_id[0] = int(row['vtxid']);

        if ismc == True:
            self.rd.scedr[0] = float(row['mcinfo_scedr']);

            if self.rd.scedr[0] < 5:
                self.rd.reco_close[0] = int(1)
            else:
                self.rd.reco_close[0] = int(0)

        self.rd.reco_passed_cuts[0] = int(0)
        if row['passed_ll_precuts'] == 1:
            self.rd.reco_passed_cuts[0] = int(1)

        self.rd.reco_selected[0] = int(0)
        if row['ll_selected'] == 1:
            self.rd.reco_selected[0] = int(1)

        self.rd.reco_vertex[0] = float(row['nueid_vertex_x'])
        self.rd.reco_vertex[1] = float(row['nueid_vertex_y'])
        self.rd.reco_vertex[2] = float(row['nueid_vertex_z'])
        
        self.rd.flash_chi2[0] = float(row['flash_chi2']);

        if row['LLem_LL'] == row['LLem_LL']:
            self.rd.LLem[0] = float(row['LLem_LL'])

        if row['LLpc_LL'] == row['LLpc_LL']:
            self.rd.LLpc[0] = float(row['LLpc_LL'])

        # fill reco
        self.rd.reco_proton_id[0]   = int(row['pid'])
        self.rd.reco_electron_id[0] = int(row['eid'])
        self.rd.reco_proton_E[0]    = float(row['reco_proton_energy'])
        self.rd.reco_electron_E[0]  = float(row['reco_electron_energy'])
        self.rd.reco_total_E[0]     = float(row['reco_energy']);

        self.tree.Fill()

        return True

