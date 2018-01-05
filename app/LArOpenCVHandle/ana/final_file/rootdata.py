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
        self.vertex_id  = array( 'i', [ kINVALID_INT ] )

        #
        # is it selected
        #
        self.reco_selected = array( 'i', [ kINVALID_INT ] )
        
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
        self.selected1L1P   = array( 'i', [ kINVALID_INT   ] )
        self.scedr          = array( 'f', [ kINVALID_FLOAT ] )
        self.nu_pdg         = array( 'i', [ kINVALID_INT   ] )
        self.inter_type     = array( 'i', [ kINVALID_INT   ] ) 
        self.inter_mode     = array( 'i', [ kINVALID_INT   ] )
        self.true_vertex    = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ] )

        self.true_proton_E    = array( 'f', [ kINVALID_FLOAT ] )
        self.true_electron_E  = array( 'f', [ kINVALID_FLOAT ] )
        self.true_total_E     = array( 'f', [ kINVALID_FLOAT ] )

        self.true_proton_dR      = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ] )
        self.true_electron_dR    = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ] )

        self.true_proton_theta   = array( 'f', [ kINVALID_FLOAT ] )
        self.true_electron_theta = array( 'f', [ kINVALID_FLOAT ] )

        self.true_proton_phi     = array( 'f', [ kINVALID_FLOAT ] )
        self.true_electron_phi   = array( 'f', [ kINVALID_FLOAT ] )

        self.true_proton_ylen   = array( 'f', [ kINVALID_FLOAT ] )
        self.true_opening_angle = array( 'f', [ kINVALID_FLOAT ] )

        self.true_nu_E = array( 'f', [ kINVALID_FLOAT ] )

        #
        # reco parameters
        #

        # truth
        self.reco_mc_proton_E    = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_mc_electron_E  = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_mc_total_E     = array( 'f', [ kINVALID_FLOAT ] )

        # track 
        self.reco_proton_E    = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_proton_len  = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_proton_ion  = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_proton_dR   = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ] )
        self.reco_proton_good = array( 'i', [ kINVALID_INT ] )

        # shower
        self.reco_electron_E     = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_electron_dEdx  = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_electron_dR    = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ] )

        # combined
        self.reco_total_E = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_vertex  = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ] )

    def reset(self):

        self.run[0]        = kINVALID_INT
        self.subrun[0]     = kINVALID_INT
        self.event[0]      = kINVALID_INT
   
        self.num_croi[0]   = kINVALID_INT
        self.num_vertex[0] = kINVALID_INT
        self.vertex_id[0]  = kINVALID_INT
        
        self.reco_selected[0] = kINVALID_INT

        # LL
        self.LL_dist[0] = kINVALID_FLOAT
        self.LLc_e[0]   = kINVALID_FLOAT
        self.LLc_p[0]   = kINVALID_FLOAT
        self.LLe_e[0]   = kINVALID_FLOAT
        self.LLe_p[0]   = kINVALID_FLOAT

        # mc
        self.true_vertex[0]   = kINVALID_FLOAT
        self.true_vertex[1]   = kINVALID_FLOAT
        self.true_vertex[2]   = kINVALID_FLOAT
        self.selected1L1P[0]  = kINVALID_INT
        self.scedr[0]         = kINVALID_FLOAT
        self.nu_pdg[0]        = kINVALID_INT
        self.true_nu_E[0]     = kINVALID_FLOAT

        self.inter_type[0]     = kINVALID_INT
        self.inter_mode[0]     = kINVALID_INT

        self.true_proton_E[0]   = kINVALID_FLOAT
        self.true_electron_E[0] = kINVALID_FLOAT

        self.true_proton_dR[0]    = kINVALID_FLOAT
        self.true_proton_dR[1]    = kINVALID_FLOAT
        self.true_proton_dR[2]    = kINVALID_FLOAT
        self.true_electron_dR[0]  = kINVALID_FLOAT
        self.true_electron_dR[1]  = kINVALID_FLOAT
        self.true_electron_dR[2]  = kINVALID_FLOAT

        self.true_proton_theta[0]   = kINVALID_FLOAT
        self.true_electron_theta[0] = kINVALID_FLOAT
        self.true_electron_phi[0]   = kINVALID_FLOAT
        self.true_proton_phi[0]     = kINVALID_FLOAT

        self.true_opening_angle[0] = kINVALID_FLOAT 
        self.true_proton_ylen[0]   = kINVALID_FLOAT

        # reco

        self.reco_mc_proton_E[0]   = kINVALID_FLOAT
        self.reco_mc_electron_E[0] = kINVALID_FLOAT
        self.reco_mc_total_E[0]    = kINVALID_FLOAT

        self.reco_proton_E[0]    = kINVALID_FLOAT
        self.reco_proton_len[0]  = kINVALID_FLOAT
        self.reco_proton_ion[0]  = kINVALID_FLOAT
        self.reco_proton_dR[0]   = kINVALID_FLOAT
        self.reco_proton_dR[1]   = kINVALID_FLOAT
        self.reco_proton_dR[2]   = kINVALID_FLOAT
        self.reco_proton_good[0] = kINVALID_INT
        
        self.reco_electron_E[0]     = kINVALID_FLOAT
        self.reco_electron_dEdx[0]  = kINVALID_FLOAT
        self.reco_electron_dR[0]    = kINVALID_FLOAT
        self.reco_electron_dR[1]    = kINVALID_FLOAT
        self.reco_electron_dR[2]    = kINVALID_FLOAT

        self.reco_total_E[0] = kINVALID_FLOAT
        self.reco_vertex[0]  = kINVALID_FLOAT
        self.reco_vertex[1]  = kINVALID_FLOAT
        self.reco_vertex[2]  = kINVALID_FLOAT
    
    def init_tree(self,tree):
        
        tree.Branch("run"   , self.run   , "run/I")
        tree.Branch("subrun", self.subrun, "subrun/I")
        tree.Branch("event" , self.event , "event/I")

        tree.Branch("num_croi"  , self.num_croi,   "num_croi/I")
        tree.Branch("num_vertex", self.num_vertex, "num_vertex/I")
        tree.Branch("vertex_id" , self.vertex_id,  "vertex_id/I")

        # truth 
        tree.Branch("nu_pdg"      , self.nu_pdg      , "nu_pdg/I")
        tree.Branch("selected1L1P", self.selected1L1P, "selected1L1P/I")
        tree.Branch("true_nu_E"   , self.true_nu_E   , "true_nu_E/F")
        tree.Branch("true_vertex" , self.true_vertex , "true_vertex[3]/F")
        tree.Branch("scedr"       , self.scedr       , "scedr/F")

        tree.Branch("inter_type", self.inter_type, "inter_type/I")
        tree.Branch("inter_mode", self.inter_mode, "inter_mode/I")

        tree.Branch("true_proton_E"  , self.true_proton_E  , "true_proton_E/F")
        tree.Branch("true_electron_E", self.true_electron_E, "true_electron_E/F")

        tree.Branch("true_proton_dR"  , self.true_proton_dR  , "true_proton_dR[3]/F")
        tree.Branch("true_electron_dR", self.true_electron_dR, "true_electron_dR[3]/F")

        tree.Branch("true_proton_theta"  , self.true_proton_theta  , "true_proton_theta/F")
        tree.Branch("true_electron_theta", self.true_electron_theta, "true_electron_theta/F")

        tree.Branch("true_proton_phi"    , self.true_proton_phi    , "true_proton_phi/F")
        tree.Branch("true_electron_phi"  , self.true_electron_phi  , "true_electron_phi/F")

        tree.Branch("true_opening_angle", self.true_opening_angle, "true_opening_angle/F")
        tree.Branch("true_proton_ylen"  , self.true_proton_ylen  , "true_proton_ylen/F")

        # reco mc
        tree.Branch("reco_mc_proton_E"  , self.reco_mc_proton_E  , "reco_mc_proton_E/F")
        tree.Branch("reco_mc_electron_E", self.reco_mc_electron_E, "reco_mc_electron_E/F")
        tree.Branch("reco_mc_total_E"   , self.reco_mc_total_E   , "reco_mc_total_E/F")

        tree.Branch("reco_selected", self.reco_selected, "reco_selected/I")

        # LL
        tree.Branch("LL_dist", self.LL_dist, "LL_dist/F")
        tree.Branch("LLc_e"  , self.LLc_e  , "LLc_e/F")
        tree.Branch("LLc_p"  , self.LLc_p  , "LLc_p/F")
        tree.Branch("LLe_e"  , self.LLe_e  , "LLe_e/F")
        tree.Branch("LLe_p"  , self.LLe_p  , "LLe_p/F")

        # reco track
        tree.Branch("reco_proton_E"   , self.reco_proton_E    , "reco_proton_E/F")
        tree.Branch("reco_proton_len" , self.reco_proton_len  , "reco_proton_len/F")
        tree.Branch("reco_proton_ion" , self.reco_proton_ion  , "reco_proton_ion/F")
        tree.Branch("reco_proton_good", self.reco_proton_good , "reco_proton_good/I")

        # reco shower
        tree.Branch("reco_electron_E"   , self.reco_electron_E   , "reco_electron_E/F")
        tree.Branch("reco_electron_dEdx", self.reco_electron_dEdx, "reco_electron_dEdx/F")
        tree.Branch("reco_electron_dR"  , self.reco_electron_dR  , "reco_electron_dR[3]/F")

        # reco combined
        tree.Branch("reco_total_E", self.reco_total_E, "reco_total_E/F")
        tree.Branch("reco_vertex" , self.reco_vertex , "reco_vertex[3]/F")
        
