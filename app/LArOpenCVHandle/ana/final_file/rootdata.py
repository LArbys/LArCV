import ROOT
from array import array

kINVALID_INT    = ROOT.std.numeric_limits("int")().lowest()
kINVALID_FLOAT  = ROOT.std.numeric_limits("float")().lowest()
kINVALID_DOUBLE = ROOT.std.numeric_limits("double")().lowest()

class ROOTData:
    def __init__(self):

        #
        # book keeping
        # 
        self.run    = array( 'i', [ kINVALID_INT ] )
        self.subrun = array( 'i', [ kINVALID_INT ] )
        self.event  = array( 'i', [ kINVALID_INT ] )

        self.num_croi   = array( 'i', [ kINVALID_INT ] )
        self.num_vertex = array( 'i', [ kINVALID_INT ] )

        #
        # is it selected
        #
        self.reco_selected = array( 'i', [ 0 ] )
        
        #
        # the LL
        #
        self.LL_dist = array( 'f', [ kINVALID_FLOAT ] )
        self.LLc_e   = array( 'f', [ kINVALID_FLOAT ] )
        self.LLc_p   = array( 'f', [ kINVALID_FLOAT ] )
        self.LLe_e   = array( 'f', [ kINVALID_FLOAT ] )
        self.LLe_p   = array( 'f', [ kINVALID_FLOAT ] )

        #
        # MC truth
        #
        self.true_X = array( 'f', [ kINVALID_FLOAT ] )
        self.true_Y = array( 'f', [ kINVALID_FLOAT ] )
        self.true_Z = array( 'f', [ kINVALID_FLOAT ] )
        self.selected1L1P  = array( 'i', [ kINVALID_INT   ] )
        self.scedr         = array( 'f', [ kINVALID_FLOAT ] )
        self.nu_pdg        = array( 'i', [ kINVALID_INT   ] )
        self.true_track_E  = array( 'f', [ kINVALID_FLOAT ] )
        self.true_shower_E = array( 'f', [ kINVALID_FLOAT ] )
        self.true_nu_E     = array( 'f', [ kINVALID_FLOAT ] )

        #
        # reco parameters
        #
        
        # track 
        self.reco_track_E_p  = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_track_E_m  = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_track_len  = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_track_ion  = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_track_dX   = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_track_dY   = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_track_dZ   = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_track_good = array( 'i', [ kINVALID_INT ] )

        # shower
        self.reco_shower_E     = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_shower_dEdx  = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_shower_dX    = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_shower_dY    = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_shower_dZ    = array( 'f', [ kINVALID_FLOAT ] )

        # combined
        self.reco_energy = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_X      = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_Y      = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_Z      = array( 'f', [ kINVALID_FLOAT ] )

    def reset(self):

        self.run[0]        = kINVALID_INT
        self.subrun[0]     = kINVALID_INT
        self.event[0]      = kINVALID_INT
        self.num_croi[0]   = kINVALID_INT
        self.num_vertex[0] = kINVALID_INT

        self.reco_selected[0] = 0

        self.LL_dist[0] = kINVALID_FLOAT
        self.LLc_e[0] = kINVALID_FLOAT
        self.LLc_p[0] = kINVALID_FLOAT
        self.LLe_e[0] = kINVALID_FLOAT
        self.LLe_p[0] = kINVALID_FLOAT

        self.true_X[0]        = kINVALID_FLOAT
        self.true_Y[0]        = kINVALID_FLOAT
        self.true_Z[0]        = kINVALID_FLOAT
        self.selected1L1P[0]  = kINVALID_INT
        self.scedr[0]         = kINVALID_FLOAT
        self.num_croi[0]      = kINVALID_INT
        self.nu_pdg[0]        = kINVALID_INT
        self.true_nu_E[0]     = kINVALID_FLOAT
        self.true_track_E[0]  = kINVALID_FLOAT
        self.true_shower_E[0] = kINVALID_FLOAT

        self.reco_track_E_p[0]  = kINVALID_FLOAT
        self.reco_track_E_m[0]  = kINVALID_FLOAT
        self.reco_track_len[0]  = kINVALID_FLOAT
        self.reco_track_ion[0]  = kINVALID_FLOAT
        self.reco_track_dX[0]   = kINVALID_FLOAT
        self.reco_track_dY[0]   = kINVALID_FLOAT
        self.reco_track_dZ[0]   = kINVALID_FLOAT
        self.reco_track_good[0] = kINVALID_INT
        
        self.reco_shower_E[0]     = kINVALID_FLOAT
        self.reco_shower_dEdx[0]  = kINVALID_FLOAT
        self.reco_shower_dX[0]    = kINVALID_FLOAT
        self.reco_shower_dY[0]    = kINVALID_FLOAT
        self.reco_shower_dZ[0]    = kINVALID_FLOAT

        self.reco_energy[0] = kINVALID_FLOAT
        self.reco_X[0]      = kINVALID_FLOAT
        self.reco_Y[0]      = kINVALID_FLOAT
        self.reco_Z[0]      = kINVALID_FLOAT
    
    def init_tree(self,tree):
        
        tree.Branch("run"   , self.run   , "run/I")
        tree.Branch("subrun", self.subrun, "subrun/I")
        tree.Branch("event" , self.event , "event/I")

        tree.Branch("reco_selected", self.reco_selected, "reco_selected/I")
        
        #LL
        tree.Branch("LL_dist", self.LL_dist, "LL_dist/F")
        tree.Branch("LLc_e"  , self.LLc_e  , "LLc_e/F")
        tree.Branch("LLc_p"  , self.LLc_p  , "LLc_p/F")
        tree.Branch("LLe_e"  , self.LLe_e  , "LLe_e/F")
        tree.Branch("LLe_p"  , self.LLe_p  , "LLe_p/F")

        #track
        tree.Branch("reco_track_E_p" , self.reco_track_E_p  , "reco_track_E_p/F")
        tree.Branch("reco_track_E_m" , self.reco_track_E_m  , "reco_track_E_m/F")
        tree.Branch("reco_track_len" , self.reco_track_len  , "reco_track_len/F")
        tree.Branch("reco_track_ion" , self.reco_track_ion  , "reco_track_ion/F")
        tree.Branch("reco_track_good", self.reco_track_good , "reco_track_good/I")

        #shower
        tree.Branch("reco_shower_E"   , self.reco_shower_E   , "reco_shower_E/F")
        tree.Branch("reco_shower_dEdx", self.reco_shower_dEdx, "reco_shower_dEdx/F")
        tree.Branch("reco_shower_dX"  , self.reco_shower_dX  , "reco_shower_dX/F")
        tree.Branch("reco_shower_dY"  , self.reco_shower_dY  , "reco_shower_dY/F")
        tree.Branch("reco_shower_dZ"  , self.reco_shower_dZ  , "reco_shower_dZ/F")

        #combined
        tree.Branch("reco_energy", self.reco_energy, "reco_energy/F")
        tree.Branch("reco_X"     , self.reco_X     , "reco_X/F")
        tree.Branch("reco_Y"     , self.reco_Y     , "reco_Y/F")
        tree.Branch("reco_Z"     , self.reco_Z     , "reco_Z/F")
