import os,sys,gc
import numpy as np
import pandas as pd

import root_numpy as rn

from common import *
from handler import Handler
from rootdata import ROOTData

class AndyHandler(Handler):
    
    def __init__(self,tree):
        super(AndyHandler,self).__init__("AndyHandler")
        self.df = pd.DataFrame()
        self.tree = tree
        self.rd = ROOTData()
        self.rd.init_andy_tree(self.tree)

    def reshape(self,inputfile) :
        
        if inputfile == "": 
            return

        self.df = pd.DataFrame(rn.root2array(inputfile,treename="EventAndyTree"))
        
    def fill(self,run,subrun,event,ismc):
        self.rd.reset()

        row = self.df.query("run==@run&subrun==@subrun&event==@event")
        if row.index.size == 0: return False
        row = row.iloc[0]

        self.rd.run[0]    = int(row['run'])
        self.rd.subrun[0] = int(row['subrun'])
        self.rd.event[0]  = int(row['event'])
        
        print "@(r,s,e)=(%d,%d,%d)"%(row['run'],row['subrun'],row['event'])
        
        if ismc == False: 
            self.tree.Fill()
            self.rd.reset()
            return True
        
        self.rd.MCFlux_NuPosX[0]   = float(row['MCFlux_NuPosX'])
        self.rd.MCFlux_NuPosY[0]   = float(row['MCFlux_NuPosY'])
        self.rd.MCFlux_NuPosZ[0]   = float(row['MCFlux_NuPosZ'])
        self.rd.MCFlux_NuMomX[0]   = float(row['MCFlux_NuMomX'])
        self.rd.MCFlux_NuMomY[0]   = float(row['MCFlux_NuMomY'])
        self.rd.MCFlux_NuMomZ[0]   = float(row['MCFlux_NuMomZ'])
        self.rd.MCFlux_NuMomE[0]   = float(row['MCFlux_NuMomE'])
        self.rd.MCFlux_ntype[0]    = int(row['MCFlux_ntype'])
        self.rd.MCFlux_ptype[0]    = int(row['MCFlux_ptype'])
        self.rd.MCFlux_nimpwt[0]   = float(row['MCFlux_nimpwt'])
        self.rd.MCFlux_dk2gen[0]   = float(row['MCFlux_dk2gen'])
        self.rd.MCFlux_nenergyn[0] = float(row['MCFlux_nenergyn'])
        self.rd.MCFlux_tpx[0]      = float(row['MCFlux_tpx'])
        self.rd.MCFlux_tpy[0]      = float(row['MCFlux_tpy'])
        self.rd.MCFlux_tpz[0]      = float(row['MCFlux_tpz'])
        self.rd.MCFlux_tptype[0]   = int(row['MCFlux_tptype'])
        self.rd.MCFlux_vx[0]       = float(row['MCFlux_vx'])
        self.rd.MCFlux_vy[0]       = float(row['MCFlux_vy'])
        self.rd.MCFlux_vz[0]       = float(row['MCFlux_vz'])

        self.rd.MCTruth_NParticles[0] = int(row['MCTruth_NParticles'])

        range1_ = self.rd.MCTruth_NParticles[0];
        if range1_ > self.rd.MaxParticles:
            print "WARNING: truncated nparticles from",range1_
            range1_ = self.rd.MaxParticles
            print "to",range1_

            # Also update the NParticles variable
            self.rd.MCTruth_NParticles[0] = range1_

        for pnum1 in xrange(int(range1_)):
            self.rd.MCTruth_particles_TrackId[pnum1]    = int(row['MCTruth_particles_TrackId'][pnum1])
            self.rd.MCTruth_particles_PdgCode[pnum1]    = int(row['MCTruth_particles_PdgCode'][pnum1])
            self.rd.MCTruth_particles_Mother[pnum1]     = int(row['MCTruth_particles_Mother'][pnum1])
            self.rd.MCTruth_particles_StatusCode[pnum1] = int(row['MCTruth_particles_StatusCode'][pnum1])
            self.rd.MCTruth_particles_NDaughters[pnum1] = int(row['MCTruth_particles_NDaughters'][pnum1])
            self.rd.MCTruth_particles_Gvx[pnum1]        = float(row['MCTruth_particles_Gvx'][pnum1])
            self.rd.MCTruth_particles_Gvy[pnum1]        = float(row['MCTruth_particles_Gvy'][pnum1])
            self.rd.MCTruth_particles_Gvz[pnum1]        = float(row['MCTruth_particles_Gvz'][pnum1])
            self.rd.MCTruth_particles_Gvt[pnum1]        = float(row['MCTruth_particles_Gvt'][pnum1])
            self.rd.MCTruth_particles_px0[pnum1]        = float(row['MCTruth_particles_px0'][pnum1])
            self.rd.MCTruth_particles_py0[pnum1]        = float(row['MCTruth_particles_py0'][pnum1])
            self.rd.MCTruth_particles_pz0[pnum1]        = float(row['MCTruth_particles_pz0'][pnum1])
            self.rd.MCTruth_particles_e0[pnum1]         = float(row['MCTruth_particles_e0'][pnum1])
            self.rd.MCTruth_particles_Rescatter[pnum1]  = int(row['MCTruth_particles_Rescatter'][pnum1])
            self.rd.MCTruth_particles_polx[pnum1]       = float(row['MCTruth_particles_polx'][pnum1])
            self.rd.MCTruth_particles_poly[pnum1]       = float(row['MCTruth_particles_poly'][pnum1])
            self.rd.MCTruth_particles_polz[pnum1]       = float(row['MCTruth_particles_polz'][pnum1])
            
            range2_ = self.rd.MCTruth_particles_NDaughters[pnum1]
            if range2_ > self.rd.MaxDaughters:
                print "WARNING: truncated ndaughters from",range2_
                range2_ = self.rd.MaxDaughters
                print "to",range2_

                # Also update the NDaughters variable for this particle
                self.rd.MCtruth_particles_NDaughters[pnum1] = range2_


            for pnum2 in xrange(int(range2_)):
                self.rd.MCTruth_particles_Daughters[pnum1*self.rd.MaxDaughters + pnum2] = int(row['MCTruth_particles_NDaughters'][pnum1][pnum2])
        
        self.rd.MCTruth_neutrino_CCNC[0] = int(row['MCTruth_neutrino_CCNC'])
        self.rd.MCTruth_neutrino_mode[0] = int(row['MCTruth_neutrino_mode'])
        self.rd.MCTruth_neutrino_interactionType[0] = int(row['MCTruth_neutrino_interactionType'])
        self.rd.MCTruth_neutrino_target[0]  = int(row['MCTruth_neutrino_target'])
        self.rd.MCTruth_neutrino_nucleon[0] = int(row['MCTruth_neutrino_nucleon'])
        self.rd.MCTruth_neutrino_quark[0]   = int(row['MCTruth_neutrino_quark'])
        self.rd.MCTruth_neutrino_W[0]  =  float(row['MCTruth_neutrino_W'])
        self.rd.MCTruth_neutrino_X[0]  =  float(row['MCTruth_neutrino_X'])
        self.rd.MCTruth_neutrino_Y[0]  =  float(row['MCTruth_neutrino_Y'])
        self.rd.MCTruth_neutrino_Q2[0] =  float(row['MCTruth_neutrino_Q2'])

        self.rd.GTruth_ProbePDG[0]   = int(row['GTruth_ProbePDG'])
        self.rd.GTruth_IsSeaQuark[0] = int(row['GTruth_IsSeaQuark'])
        self.rd.GTruth_tgtPDG[0]     = int(row['GTruth_tgtPDG'])

        self.rd.GTruth_weight[0]      = float(row['GTruth_weight'])
        self.rd.GTruth_probability[0] = float(row['GTruth_probability'])
        self.rd.GTruth_Xsec[0]        = float(row['GTruth_Xsec'])
        self.rd.GTruth_fDiffXsec[0]   = float(row['GTruth_fDiffXsec'])
        self.rd.GTruth_vertexX[0]     = float(row['GTruth_vertexX'])
        self.rd.GTruth_vertexY[0]     = float(row['GTruth_vertexY'])
        self.rd.GTruth_vertexZ[0]     = float(row['GTruth_vertexZ'])
        self.rd.GTruth_vertexT[0]     = float(row['GTruth_vertexT'])
        self.rd.GTruth_Gscatter[0]   = int(row['GTruth_Gscatter'])
        self.rd.GTruth_Gint[0]       = int(row['GTruth_Gint'])
        self.rd.GTruth_ResNum[0]     = int(row['GTruth_ResNum'])
        self.rd.GTruth_NumPiPlus[0]  = int(row['GTruth_NumPiPlus'])
        self.rd.GTruth_NumPi0[0]     = int(row['GTruth_NumPi0'])
        self.rd.GTruth_NumPiMinus[0] = int(row['GTruth_NumPiMinus'])
        self.rd.GTruth_NumProton[0]  = int(row['GTruth_NumProton'])
        self.rd.GTruth_NumNeutron[0] = int(row['GTruth_NumNeutron'])
        self.rd.GTruth_IsCharm[0]    = int(row['GTruth_IsCharm'])

        self.rd.GTruth_gX[0]  = float(row['GTruth_gX'])
        self.rd.GTruth_gY[0]  = float(row['GTruth_gY'])
        self.rd.GTruth_gZ[0]  = float(row['GTruth_gZ'])
        self.rd.GTruth_gT[0]  = float(row['GTruth_gT'])
        self.rd.GTruth_gW[0]  = float(row['GTruth_gW'])
        self.rd.GTruth_gQ2[0] = float(row['GTruth_gQ2'])
        self.rd.GTruth_gq2[0] = float(row['GTruth_gq2'])
        self.rd.GTruth_ProbeP4x[0]  = float(row['GTruth_ProbeP4x'])
        self.rd.GTruth_ProbeP4y[0]  = float(row['GTruth_ProbeP4y'])
        self.rd.GTruth_ProbeP4z[0]  = float(row['GTruth_ProbeP4z'])
        self.rd.GTruth_ProbeP4E[0]  = float(row['GTruth_ProbeP4E'])
        self.rd.GTruth_HitNucP4x[0] = float(row['GTruth_HitNucP4x'])
        self.rd.GTruth_HitNucP4y[0] = float(row['GTruth_HitNucP4y'])
        self.rd.GTruth_HitNucP4z[0] = float(row['GTruth_HitNucP4z'])
        self.rd.GTruth_HitNucP4E[0] = float(row['GTruth_HitNucP4E'])
        self.rd.GTruth_FShadSystP4x[0] = float(row['GTruth_FShadSystP4x'])
        self.rd.GTruth_FShadSystP4y[0] = float(row['GTruth_FShadSystP4y'])
        self.rd.GTruth_FShadSystP4z[0] = float(row['GTruth_FShadSystP4z'])
        self.rd.GTruth_FShadSystP4E[0] = float(row['GTruth_FShadSystP4E'])

        self.tree.Fill()
        self.rd.reset()

        return True

