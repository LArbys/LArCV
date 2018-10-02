import os,sys,gc
import numpy as np
import pandas as pd
import root_numpy as rn

from common import *
from combine import *
from handler import Handler
from rootdata import ROOTData

class NumuHandler(Handler):
    
    def __init__(self,tree):
        super(NumuHandler,self).__init__("NumuHandler")
        self.df = pd.DataFrame()
        self.no_vertex = False
        self.no_pass = False
        self.tree = tree
        self.rd = ROOTData()
        self.rd.init_numu_tree(self.tree)

    def reshape(self,ifile0,ifile1,ifile2,ifile3=None) :
        print
        print "--> @numu_handler reshape"
        
        df = pd.DataFrame()
        df_sort = pd.DataFrame()
        
        print "read @ifile0=",ifile0
        df0 = pd.read_pickle(ifile0)

        print "read @ifile1=",ifile1
        try:
            df1 = pd.DataFrame(rn.root2array(ifile1))
        except IOError:
            print "Empty ttree in this file"
            df = combine_vertex_numu(df0)
            df_sort = df.groupby(RSE).head(1).copy()
            df_sort.sort_values(by=RSE,inplace=True)
            self.df = df_sort.copy()
            self.df[RSE] = self.df[RSE].astype(np.int64)
            self.no_vertex = True
            del df
            del df_sort
            gc.collect()
            return

        print "read @ifile2=",ifile2
        df2 = pd.DataFrame(rn.root2array(ifile2))

        df1.rename(columns={"vtx_id" : "vtxid"},inplace=True)

        for df_ in [df1,df2]:
            df_.rename(columns={"_run"    : "run"},inplace=True)
            df_.rename(columns={"_subrun" : "subrun"},inplace=True)
            df_.rename(columns={"_event"  : "event"},inplace=True)
            df_['run']    = df_['run'].astype(np.int32)
            df_['subrun'] = df_['subrun'].astype(np.int32)
            df_['event']  = df_['event'].astype(np.int32)
            df_['vtxid']  = df_['vtxid'].astype(np.int32)
        
        # join first 2 on vtxid
        df1.set_index(RSEV,inplace=True)
        df2.set_index(RSEV,inplace=True)
        
        df1 = df1.add_prefix("anatrk_")
        df2 = df2.add_prefix("numu_")
        
        df = df1.join(df2)
        
        df.reset_index(inplace=True)

        # add in the MC information
        if ifile3 is not None:
            df3 = pd.DataFrame(rn.root2array(ifile3,treename="EventMCINFO_DL"))
            df3 = df3[['run','subrun','event','parentSCEX','parentSCEY','parentSCEZ']]
            df3.set_index(RSE,inplace=True)
            df3 = df3.add_prefix("mcinfo_")
            df.set_index(RSE,inplace=True)

            df = df.join(df3)

            df.reset_index(inplace=True)

            df["mcinfo_scedr"] = df.apply(bless_numu_scedr,axis=1)

        # join on the comb vertex df
        df = combine_vertex_numu(df0,df)

        print "Maximizing @ CosmicLL..."
        df_sort = df.sort_values(["numu_CosmicLL"],ascending=False).groupby(RSE).head(1).copy()
        df_sort.sort_values(by=RSE,inplace=True)
        print "... maximized"
        
        self.df = df_sort.copy()
        self.df[RSE] = self.df[RSE].astype(np.int64)

        del df
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
        
        # fill common
        self.rd.num_croi[0]   = int(row['locv_number_croi']);
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
        
        # fill LL
        if float(row['numu_CosmicLL']) > -3 and (row['numu_NuBkgLL']) > 0:
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

        self.rd.Muon_id[0]          = int(row["numu_Muon_id"])
        self.rd.Muon_PhiReco[0]     = float(row["numu_Muon_PhiReco"])
        self.rd.Muon_ThetaReco[0]   = float(row["numu_Muon_ThetaReco"])
        self.rd.Muon_TrackLength[0] = float(row["numu_Muon_TrackLength"])
        self.rd.Muon_dQdx[0]        = float(row["numu_Muon_dQdx"])
        self.rd.Muon_Edep[0]        = float(row["numu_Muon_Edep"])

        self.rd.Proton_id[0]          = int(row["numu_Proton_id"])
        self.rd.Proton_PhiReco[0]     = float(row["numu_Proton_PhiReco"])
        self.rd.Proton_ThetaReco[0]   = float(row["numu_Proton_ThetaReco"])
        self.rd.Proton_TrackLength[0] = float(row["numu_Proton_TrackLength"])
        self.rd.Proton_dQdx[0]        = float(row["numu_Proton_dQdx"])
        self.rd.Proton_Edep[0]        = float(row["numu_Proton_Edep"])

        self.tree.Fill()
        return True



