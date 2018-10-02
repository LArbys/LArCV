import os,sys,gc
import numpy as np
import pandas as pd
import root_numpy as rn

from common import *
from combine import *
from handler import Handler
from rootdata import ROOTData

class NueCutHandler(Handler):
    
    def __init__(self,tree):
        super(NueCutHandler,self).__init__("NueCutHandler")
        self.df = pd.DataFrame()
        self.no_vertex = False
        self.tree = tree
        self.rd = ROOTData()
        self.rd.init_nue_cut_tree(self.tree)

    def reshape(self,inputfile,mcinfo) :
        
        print
        print "--> @nue_cut_handler reshape"

        df = pd.DataFrame()
        df3 = pd.DataFrame()        
        df_sort = pd.DataFrame()

        print "inputfile=",inputfile
        df = pd.read_pickle(inputfile)

        print "Check if no vertex"
        if "locv_num_vertex" not in df.columns:
            print "No vertex found in file"
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

        print "Sort by selected ..."
        df_sort = df.sort_values(["selected"],ascending=False).groupby(RSE).head(1).copy()
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
        if row['passed_cuts'] == 1:
            self.rd.reco_passed_cuts[0] = int(1)

        self.rd.reco_selected[0] = int(0)
        if row['selected'] == 1:
            self.rd.reco_selected[0] = int(1)

        self.rd.reco_vertex[0] = float(row['nueid_vertex_x'])
        self.rd.reco_vertex[1] = float(row['nueid_vertex_y'])
        self.rd.reco_vertex[2] = float(row['nueid_vertex_z'])
        
        self.rd.flash_chi2[0] = float(row['flash_chi2']);
        
        # fill reco
        self.rd.reco_proton_id[0]   = int(row['pid'])
        self.rd.reco_electron_id[0] = int(row['eid'])
        self.rd.reco_proton_E[0]    = float(row['reco_proton_energy'])
        self.rd.reco_electron_E[0]  = float(row['reco_electron_energy'])
        self.rd.reco_total_E[0]     = float(row['reco_energy']);

        self.tree.Fill()

        return True

