import ROOT
from array import array

kINVALID_INT    = ROOT.std.numeric_limits("int")().min()
kINVALID_FLOAT  = ROOT.std.numeric_limits("float")().min()
kINVALID_DOUBLE = ROOT.std.numeric_limits("double")().min()

class ROOTData:
    def __init__(self):

        #
        # book keeping
        # 
        self.run    = array( 'i', [ kINVALID_INT ] )
        self.subrun = array( 'i', [ kINVALID_INT ] )
        self.event  = array( 'i', [ kINVALID_INT ] )

        self.num_croi   = array( 'i', [ kINVALID_INT   ] )
        self.num_vertex = array( 'i', [ kINVALID_INT   ] )
        
        #
        # the LL
        #
        self.LL = array( 'f', [ kINVALID_FLOAT ] )

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
        self.reco_track_E  = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_shower_E = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_shower_dEdx = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_X = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_Y = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_Z = array( 'f', [ kINVALID_FLOAT ] )

    def reset(self):

        self.run[0]        = kINVALID_INT
        self.subrun[0]     = kINVALID_INT
        self.event[0]      = kINVALID_INT
        self.num_croi[0]   = kINVALID_INT
        self.num_vertex[0] = kINVALID_INT

        self.LL[0] = kINVALID_FLOAT

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

        self.reco_track_E[0]  = kINVALID_FLOAT
        self.reco_shower_E[0] = kINVALID_FLOAT
        self.reco_shower_dEdx[0] = kINVALID_FLOAT
        self.reco_X[0] = kINVALID_FLOAT
        self.reco_Y[0] = kINVALID_FLOAT
        self.reco_Z[0] = kINVALID_FLOAT
    
    def init_tree(self,tree):
        
        tree.Branch("run"   , self.run   , "run/I")
        tree.Branch("subrun", self.subrun, "subrun/I")
        tree.Branch("event" , self.event , "event/I")
        tree.Branch("LL"    , self.LL    , "LL/F")

        tree.Branch("reco_shower_E", self.reco_shower_E, "reco_shower_E/F")
