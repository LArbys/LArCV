import os,sys,gc
import numpy as np
import pandas as pd

import root_numpy as rn

from common import *
from handler import Handler
from rootdata import ROOTData

class SegmentHandler(Handler):
    
    def __init__(self,tree):
        super(SegmentHandler,self).__init__("SegmentHandler")
        self.df = pd.DataFrame()
        self.tree = tree
        self.rd = ROOTData()
        self.rd.init_segment_tree(self.tree)
        self.ismc = True
        return

    def reshape(self,inputfile) :
        print
        print "--> @segment_handler"
        if inputfile is None:
            self.ismc = False
            return

        self.df = pd.DataFrame(rn.root2array(inputfile,treename="EventMCINFO_DL"))

        print "--> done"
        print
        return
        
    def fill(self,run,subrun,event,ismc):
        self.rd.reset()

        if self.ismc == False:
            self.rd.run[0]    = int(run)
            self.rd.subrun[0] = int(subrun)
            self.rd.event[0]  = int(event)
            self.tree.Fill()
            self.rd.reset()
            return True

        row = self.df.query("run==@run&subrun==@subrun&event==@event")
        if row.index.size == 0: 
            return False

        row = row.iloc[0]

        self.rd.run[0]    = int(row['run'])
        self.rd.subrun[0] = int(row['subrun'])
        self.rd.event[0]  = int(row['event'])
        
        print "@(r,s,e)=(%d,%d,%d)"%(row['run'],row['subrun'],row['event'])

        self.rd.nu_pdg[0] = int(row['parentPDG'])
        self.rd.inter_type[0] = int(row['ineractionMode'])
        self.rd.inter_mode[0] = int(row['interactionType'])
        self.rd.selected1L1P[0] = int(row['selected1L1P'])

        self.rd.true_nu_E[0] = float(row['energyInit'])

        self.rd.true_vertex[0] = float(row['parentX'])
        self.rd.true_vertex[1] = float(row['parentY'])
        self.rd.true_vertex[2] = float(row['parentZ'])

        self.rd.true_vertex_sce[0] = float(row['parentSCEX'])
        self.rd.true_vertex_sce[1] = float(row['parentSCEY'])
        self.rd.true_vertex_sce[2] = float(row['parentSCEZ'])

        self.rd.true_proton_E[0] = float(row['dep_sum_proton'])
        self.rd.true_lepton_E[0] = float(row['dep_sum_lepton'])

        self.rd.true_proton_P[0] = float(row['proton_Px'])
        self.rd.true_proton_P[1] = float(row['proton_Py'])
        self.rd.true_proton_P[2] = float(row['proton_Pz'])

        self.rd.true_lepton_P[0] = float(row['lepton_Px'])
        self.rd.true_lepton_P[1] = float(row['lepton_Py'])
        self.rd.true_lepton_P[2] = float(row['lepton_Pz'])

        self.tree.Fill()
        self.rd.reset()

        return True

