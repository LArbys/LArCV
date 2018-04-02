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
        self.reco_close    = array( 'i', [ kINVALID_INT ] )
        self.reco_on_nu    = array( 'i', [ kINVALID_INT ] )

        #
        # the LL
        #
        self.LL_dist = array( 'f', [ kINVALID_FLOAT ] )
        self.LLc_e   = array( 'f', [ kINVALID_FLOAT ] )
        self.LLc_p   = array( 'f', [ kINVALID_FLOAT ] )
        self.LLe_e   = array( 'f', [ kINVALID_FLOAT ] )
        self.LLe_p   = array( 'f', [ kINVALID_FLOAT ] )


        #
        # the flash
        #
        self.chi2 = array( 'f', [ kINVALID_FLOAT] )

        #
        # MC truth
        #
        self.selected1L1P    = array( 'i', [ kINVALID_INT   ] )
        self.scedr           = array( 'f', [ -1.0*kINVALID_FLOAT ] )
        self.nu_pdg          = array( 'i', [ kINVALID_INT   ] )
        self.inter_type      = array( 'i', [ kINVALID_INT   ] ) 
        self.inter_mode      = array( 'i', [ kINVALID_INT   ] )
        self.true_vertex     = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ] )
        self.true_vertex_sce = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ] )

        self.true_proton_E    = array( 'f', [ kINVALID_FLOAT ] )
        self.true_lepton_E    = array( 'f', [ kINVALID_FLOAT ] )
        self.true_electron_E  = array( 'f', [ kINVALID_FLOAT ] )
        self.true_total_E     = array( 'f', [ kINVALID_FLOAT ] )

        self.true_proton_P  = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ] )
        self.true_lepton_P  = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ] )
    
        self.true_proton_ylen   = array( 'f', [ kINVALID_FLOAT ] )
        self.true_opening_angle = array( 'f', [ kINVALID_FLOAT ] )

        self.true_nu_E = array( 'f', [ kINVALID_FLOAT ] )

        #
        # reco parameters
        #

        # truth
        self.reco_mc_proton_E  = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_mc_lepton_E  = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_mc_total_E   = array( 'f', [ kINVALID_FLOAT ] )

        # proton and electron
        self.reco_proton_id     = array( 'i', [ kINVALID_INT   ] )
        self.reco_proton_E      = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_proton_theta  = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_proton_phi    = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_proton_dEdx   = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_proton_length = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_proton_mean_pixel_dist = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_proton_width = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_proton_area = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_proton_qsum = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_proton_shower_frac = array( 'f', [ kINVALID_FLOAT ] )

        self.reco_electron_id    = array( 'i', [ kINVALID_INT]    )
        self.reco_electron_E     = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_electron_theta = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_electron_phi   = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_electron_dEdx  = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_electron_length   = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_electron_mean_pixel_dist = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_electron_width = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_electron_area = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_electron_qsum = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_electron_shower_frac = array( 'f', [ kINVALID_FLOAT ] )

        # combined
        self.reco_total_E = array( 'f', [ kINVALID_FLOAT ] )
        self.reco_vertex  = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ] )


        #
        # numu related
        #
        self.CosmicLL      = array( 'f', [ kINVALID_FLOAT ])
        self.NuBkgLL       = array( 'f', [ kINVALID_FLOAT ])
        self.CCpi0LL       = array( 'f', [ kINVALID_FLOAT ])
        self.PassCuts      = array( 'i', [ kINVALID_INT   ])
        self.VtxAlgo       = array( 'i', [ kINVALID_INT   ])
        self.NTracks       = array( 'i', [ kINVALID_INT   ])
        self.N5cmTracks    = array( 'i', [ kINVALID_INT   ])
        self.InFiducial    = array( 'i', [ kINVALID_INT   ])
        self.Good3DReco    = array( 'i', [ kINVALID_INT   ])
        self.AnythingRecod = array( 'i', [ kINVALID_INT   ])

        self.Muon_id          = array( 'i', [ kINVALID_INT   ])
        self.Muon_PhiReco     = array( 'f', [ kINVALID_FLOAT ])
        self.Muon_ThetaReco   = array( 'f', [ kINVALID_FLOAT ])
        self.Muon_TrackLength = array( 'f', [ kINVALID_FLOAT ])
        self.Muon_dQdx        = array( 'f', [ kINVALID_FLOAT ])
        self.Muon_Trunc_dQdx1 = array( 'f', [ kINVALID_FLOAT ])
        self.Muon_Trunc_dQdx3 = array( 'f', [ kINVALID_FLOAT ])
        self.Muon_IonPerLen   = array( 'f', [ kINVALID_FLOAT ])
        self.Muon_Edep        = array( 'f', [ kINVALID_FLOAT ])

        self.Proton_id          = array( 'i', [ kINVALID_INT   ])
        self.Proton_PhiReco     = array( 'f', [ kINVALID_FLOAT ])
        self.Proton_ThetaReco   = array( 'f', [ kINVALID_FLOAT ])
        self.Proton_TrackLength = array( 'f', [ kINVALID_FLOAT ])
        self.Proton_dQdx        = array( 'f', [ kINVALID_FLOAT ])
        self.Proton_Trunc_dQdx1 = array( 'f', [ kINVALID_FLOAT ])
        self.Proton_Trunc_dQdx3 = array( 'f', [ kINVALID_FLOAT ])
        self.Proton_IonPerLen   = array( 'f', [ kINVALID_FLOAT ])
        self.Proton_Edep        = array( 'f', [ kINVALID_FLOAT ])

        #
        # pid related
        # 
        self.inferred     = array( 'i', [ kINVALID_INT   ])
        self.eminus_score = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ])
        self.gamma_score  = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ])
        self.muon_score   = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ])
        self.pion_score   = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ])
        self.proton_score = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ])

        self.eminus_score_vtx = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ])
        self.gamma_score_vtx  = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ])
        self.muon_score_vtx   = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ])
        self.pion_score_vtx   = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ])
        self.proton_score_vtx = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ])
        
        self.nue_perceptron = array( 'f', [ kINVALID_FLOAT, kINVALID_FLOAT, kINVALID_FLOAT ])

        #
        # andy related
        #

        self.MaxParticles = int(50)
        self.MaxDaughters = int(100)

        self.MCFlux_NuPosX = array( 'd', [ kINVALID_DOUBLE ])
        self.MCFlux_NuPosY = array( 'd', [ kINVALID_DOUBLE ])
        self.MCFlux_NuPosZ = array( 'd', [ kINVALID_DOUBLE ])
        self.MCFlux_NuMomX = array( 'd', [ kINVALID_DOUBLE ])
        self.MCFlux_NuMomY = array( 'd', [ kINVALID_DOUBLE ])
        self.MCFlux_NuMomZ = array( 'd', [ kINVALID_DOUBLE ])
        self.MCFlux_NuMomE = array( 'd', [ kINVALID_DOUBLE ])
        self.MCFlux_ntype = array( 'i', [ kINVALID_INT ])
        self.MCFlux_ptype = array( 'i', [ kINVALID_INT ])
        self.MCFlux_nimpwt = array( 'd', [ kINVALID_DOUBLE ])
        self.MCFlux_dk2gen = array( 'd', [ kINVALID_DOUBLE ])
        self.MCFlux_nenergyn = array( 'd', [ kINVALID_DOUBLE ])
        self.MCFlux_tpx = array( 'd', [ kINVALID_DOUBLE ])
        self.MCFlux_tpy = array( 'd', [ kINVALID_DOUBLE ])
        self.MCFlux_tpz = array( 'd', [ kINVALID_DOUBLE ])
        self.MCFlux_tptype = array( 'i', [ kINVALID_INT ])
        self.MCFlux_vx = array( 'd', [ kINVALID_DOUBLE ])
        self.MCFlux_vy = array( 'd', [ kINVALID_DOUBLE ])
        self.MCFlux_vz = array( 'd', [ kINVALID_DOUBLE ])

        self.MCTruth_NParticles = array( 'i', [ kINVALID_INT ])

        self.MCTruth_particles_TrackId    = array( 'i', [ kINVALID_INT ]*self.MaxParticles)
        self.MCTruth_particles_PdgCode    = array( 'i', [ kINVALID_INT ]*self.MaxParticles)
        self.MCTruth_particles_Mother     = array( 'i', [ kINVALID_INT ]*self.MaxParticles)
        self.MCTruth_particles_StatusCode = array( 'i', [ kINVALID_INT ]*self.MaxParticles)
        self.MCTruth_particles_NDaughters = array( 'i', [ kINVALID_INT ]*self.MaxParticles)
        self.MCTruth_particles_Daughters  = array( 'i', [ kINVALID_INT ]*self.MaxParticles*self.MaxDaughters)
        self.MCTruth_particles_Gvx        = array( 'd', [ kINVALID_DOUBLE ]*self.MaxParticles)
        self.MCTruth_particles_Gvy        = array( 'd', [ kINVALID_DOUBLE ]*self.MaxParticles)
        self.MCTruth_particles_Gvz        = array( 'd', [ kINVALID_DOUBLE ]*self.MaxParticles)
        self.MCTruth_particles_Gvt        = array( 'd', [ kINVALID_DOUBLE ]*self.MaxParticles)
        self.MCTruth_particles_px0        = array( 'd', [ kINVALID_DOUBLE ]*self.MaxParticles)
        self.MCTruth_particles_py0        = array( 'd', [ kINVALID_DOUBLE ]*self.MaxParticles)
        self.MCTruth_particles_pz0        = array( 'd', [ kINVALID_DOUBLE ]*self.MaxParticles)
        self.MCTruth_particles_e0         = array( 'd', [ kINVALID_DOUBLE ]*self.MaxParticles)
        self.MCTruth_particles_Rescatter  = array( 'i', [ kINVALID_INT ]*self.MaxParticles)
        self.MCTruth_particles_polx       = array( 'd', [ kINVALID_DOUBLE ]*self.MaxParticles)
        self.MCTruth_particles_poly       = array( 'd', [ kINVALID_DOUBLE ]*self.MaxParticles)
        self.MCTruth_particles_polz       = array( 'd', [ kINVALID_DOUBLE ]*self.MaxParticles)

        self.MCTruth_neutrino_CCNC            = array( 'i', [ kINVALID_INT ])
        self.MCTruth_neutrino_mode            = array( 'i', [ kINVALID_INT ])
        self.MCTruth_neutrino_interactionType = array( 'i', [ kINVALID_INT ])
        self.MCTruth_neutrino_target          = array( 'i', [ kINVALID_INT ])
        self.MCTruth_neutrino_nucleon         = array( 'i', [ kINVALID_INT ])
        self.MCTruth_neutrino_quark           = array( 'i', [ kINVALID_INT ])
        self.MCTruth_neutrino_W               = array( 'd', [ kINVALID_DOUBLE ])
        self.MCTruth_neutrino_X               = array( 'd', [ kINVALID_DOUBLE ])
        self.MCTruth_neutrino_Y               = array( 'd', [ kINVALID_DOUBLE ])
        self.MCTruth_neutrino_Q2              = array( 'd', [ kINVALID_DOUBLE ])

        self.GTruth_ProbePDG   = array( 'i', [ kINVALID_INT ])
        self.GTruth_IsSeaQuark = array( 'i', [ kINVALID_INT ])
        self.GTruth_tgtPDG     = array( 'i', [ kINVALID_INT ])

        self.GTruth_weight      = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_probability = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_Xsec        = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_fDiffXsec   = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_vertexX     = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_vertexY     = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_vertexZ     = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_vertexT     = array( 'd', [ kINVALID_DOUBLE ])

        self.GTruth_Gscatter   = array( 'i', [ kINVALID_INT ])
        self.GTruth_Gint       = array( 'i', [ kINVALID_INT ])
        self.GTruth_ResNum     = array( 'i', [ kINVALID_INT ])
        self.GTruth_NumPiPlus  = array( 'i', [ kINVALID_INT ])
        self.GTruth_NumPi0     = array( 'i', [ kINVALID_INT ])
        self.GTruth_NumPiMinus = array( 'i', [ kINVALID_INT ])
        self.GTruth_NumProton  = array( 'i', [ kINVALID_INT ])
        self.GTruth_NumNeutron = array( 'i', [ kINVALID_INT ])
        self.GTruth_IsCharm    = array( 'i', [ kINVALID_INT ])

        self.GTruth_gX  = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_gY  = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_gZ  = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_gT  = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_gW  = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_gQ2 = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_gq2 = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_ProbeP4x  = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_ProbeP4y  = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_ProbeP4z  = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_ProbeP4E  = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_HitNucP4x = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_HitNucP4y = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_HitNucP4z = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_HitNucP4E = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_FShadSystP4x = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_FShadSystP4y = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_FShadSystP4z = array( 'd', [ kINVALID_DOUBLE ])
        self.GTruth_FShadSystP4E = array( 'd', [ kINVALID_DOUBLE ])

    def reset(self):

        self.run[0]        = kINVALID_INT
        self.subrun[0]     = kINVALID_INT
        self.event[0]      = kINVALID_INT
   
        self.num_croi[0]   = kINVALID_INT
        self.num_vertex[0] = kINVALID_INT
        self.vertex_id[0]  = kINVALID_INT
        
        self.reco_selected[0] = kINVALID_INT
        self.reco_close[0] = kINVALID_INT
        self.reco_on_nu[0] = kINVALID_INT

        # LL
        self.LL_dist[0] = kINVALID_FLOAT
        self.LLc_e[0]   = kINVALID_FLOAT
        self.LLc_p[0]   = kINVALID_FLOAT
        self.LLe_e[0]   = kINVALID_FLOAT
        self.LLe_p[0]   = kINVALID_FLOAT

        # chi2
        self.chi2[0] = kINVALID_FLOAT

        # mc
        self.true_vertex[0]   = kINVALID_FLOAT
        self.true_vertex[1]   = kINVALID_FLOAT
        self.true_vertex[2]   = kINVALID_FLOAT

        self.true_vertex_sce[0]   = kINVALID_FLOAT
        self.true_vertex_sce[1]   = kINVALID_FLOAT
        self.true_vertex_sce[2]   = kINVALID_FLOAT

        self.selected1L1P[0]  = kINVALID_INT
        self.scedr[0]         = -1.0*kINVALID_FLOAT
        self.nu_pdg[0]        = kINVALID_INT
        self.true_nu_E[0]     = kINVALID_FLOAT

        self.inter_type[0]     = kINVALID_INT
        self.inter_mode[0]     = kINVALID_INT

        self.true_proton_E[0]   = kINVALID_FLOAT
        self.true_lepton_E[0]   = kINVALID_FLOAT

        self.true_proton_P[0]    = kINVALID_FLOAT
        self.true_proton_P[1]    = kINVALID_FLOAT
        self.true_proton_P[2]    = kINVALID_FLOAT

        self.true_lepton_P[0]  = kINVALID_FLOAT
        self.true_lepton_P[1]  = kINVALID_FLOAT
        self.true_lepton_P[2]  = kINVALID_FLOAT

        self.true_opening_angle[0] = kINVALID_FLOAT 
        self.true_proton_ylen[0]   = kINVALID_FLOAT

        # reco

        self.reco_mc_proton_E[0]   = kINVALID_FLOAT
        self.reco_mc_lepton_E[0] = kINVALID_FLOAT
        self.reco_mc_total_E[0]    = kINVALID_FLOAT
        
        self.reco_proton_id[0]    = kINVALID_INT
        self.reco_proton_E[0]     = kINVALID_FLOAT
        self.reco_proton_theta[0] = kINVALID_FLOAT
        self.reco_proton_phi[0]   = kINVALID_FLOAT
        self.reco_proton_dEdx[0]  = kINVALID_FLOAT
        self.reco_proton_length[0]   = kINVALID_FLOAT
        self.reco_proton_mean_pixel_dist[0] = kINVALID_FLOAT
        self.reco_proton_width[0] = kINVALID_FLOAT
        self.reco_proton_area[0] = kINVALID_FLOAT
        self.reco_proton_qsum[0] = kINVALID_FLOAT
        self.reco_proton_shower_frac[0] = kINVALID_FLOAT

        self.reco_electron_id[0]    = kINVALID_INT
        self.reco_electron_E[0]     = kINVALID_FLOAT
        self.reco_electron_theta[0] = kINVALID_FLOAT
        self.reco_electron_phi[0]   = kINVALID_FLOAT
        self.reco_electron_dEdx[0]  = kINVALID_FLOAT
        self.reco_electron_length[0]   = kINVALID_FLOAT
        self.reco_electron_mean_pixel_dist[0] = kINVALID_FLOAT
        self.reco_electron_width[0] = kINVALID_FLOAT
        self.reco_electron_area[0] = kINVALID_FLOAT
        self.reco_electron_qsum[0] = kINVALID_FLOAT
        self.reco_electron_shower_frac[0] = kINVALID_FLOAT

        self.reco_total_E[0] = kINVALID_FLOAT
        self.reco_vertex[0]  = kINVALID_FLOAT
        self.reco_vertex[1]  = kINVALID_FLOAT
        self.reco_vertex[2]  = kINVALID_FLOAT

        #
        # numu related
        #
    
        self.CosmicLL[0]      = kINVALID_FLOAT
        self.NuBkgLL[0]       = kINVALID_FLOAT
        self.CCpi0LL[0]       = kINVALID_FLOAT
        self.PassCuts[0]      = kINVALID_INT
        self.VtxAlgo[0]       = kINVALID_INT
        self.NTracks[0]       = kINVALID_INT
        self.N5cmTracks[0]    = kINVALID_INT
        self.InFiducial[0]    = kINVALID_INT
        self.Good3DReco[0]    = kINVALID_INT
        self.AnythingRecod[0] = kINVALID_INT
        
        self.Muon_id[0]          = kINVALID_INT;
        self.Muon_PhiReco[0]     = kINVALID_FLOAT;
        self.Muon_ThetaReco[0]   = kINVALID_FLOAT;
        self.Muon_TrackLength[0] = kINVALID_FLOAT;
        self.Muon_dQdx[0]        = kINVALID_FLOAT;
        self.Muon_Trunc_dQdx1[0] = kINVALID_FLOAT;
        self.Muon_Trunc_dQdx3[0] = kINVALID_FLOAT;
        self.Muon_IonPerLen[0]   = kINVALID_FLOAT;
        self.Muon_Edep[0]        = kINVALID_FLOAT;        

        self.Proton_id[0]          = kINVALID_INT;
        self.Proton_PhiReco[0]     = kINVALID_FLOAT;
        self.Proton_ThetaReco[0]   = kINVALID_FLOAT;
        self.Proton_TrackLength[0] = kINVALID_FLOAT;
        self.Proton_dQdx[0]        = kINVALID_FLOAT;
        self.Proton_Trunc_dQdx1[0] = kINVALID_FLOAT;
        self.Proton_Trunc_dQdx3[0] = kINVALID_FLOAT;
        self.Proton_IonPerLen[0]   = kINVALID_FLOAT;
        self.Proton_Edep[0]        = kINVALID_FLOAT;
        
        
        #
        # PID related
        #
        self.inferred[0]     = kINVALID_INT;
        for pl in xrange(3):
            self.eminus_score[pl] = kINVALID_FLOAT;
            self.gamma_score[pl]  = kINVALID_FLOAT;
            self.muon_score[pl]   = kINVALID_FLOAT;
            self.pion_score[pl]   = kINVALID_FLOAT;
            self.proton_score[pl] = kINVALID_FLOAT;

            self.eminus_score_vtx[pl] = kINVALID_FLOAT;
            self.gamma_score_vtx[pl]  = kINVALID_FLOAT;
            self.muon_score_vtx[pl]   = kINVALID_FLOAT;
            self.pion_score_vtx[pl]   = kINVALID_FLOAT;
            self.proton_score_vtx[pl] = kINVALID_FLOAT;

            self.nue_perceptron[pl] = kINVALID_FLOAT


        #
        # andy related
        #
        self.MCFlux_NuPosX[0]   = kINVALID_DOUBLE
        self.MCFlux_NuPosY[0]   = kINVALID_DOUBLE
        self.MCFlux_NuPosZ[0]   = kINVALID_DOUBLE
        self.MCFlux_NuMomX[0]   = kINVALID_DOUBLE
        self.MCFlux_NuMomY[0]   = kINVALID_DOUBLE
        self.MCFlux_NuMomZ[0]   = kINVALID_DOUBLE
        self.MCFlux_NuMomE[0]   = kINVALID_DOUBLE
        self.MCFlux_ntype[0]    = kINVALID_INT
        self.MCFlux_ptype[0]    = kINVALID_INT
        self.MCFlux_nimpwt[0]   = kINVALID_DOUBLE
        self.MCFlux_dk2gen[0]   = kINVALID_DOUBLE
        self.MCFlux_nenergyn[0] = kINVALID_DOUBLE
        self.MCFlux_tpx[0]      = kINVALID_DOUBLE
        self.MCFlux_tpy[0]      = kINVALID_DOUBLE
        self.MCFlux_tpz[0]      = kINVALID_DOUBLE
        self.MCFlux_tptype[0]   = kINVALID_INT
        self.MCFlux_vx[0]       = kINVALID_DOUBLE
        self.MCFlux_vy[0]       = kINVALID_DOUBLE
        self.MCFlux_vz[0]       = kINVALID_DOUBLE

        self.MCTruth_NParticles[0] = kINVALID_INT

        for pnum in xrange(self.MaxParticles):
            self.MCTruth_particles_TrackId[pnum] = kINVALID_INT
            self.MCTruth_particles_PdgCode[pnum] = kINVALID_INT
            self.MCTruth_particles_Mother[pnum] = kINVALID_INT
            self.MCTruth_particles_StatusCode[pnum] = kINVALID_INT
            self.MCTruth_particles_NDaughters[pnum] = kINVALID_INT
            self.MCTruth_particles_Gvx[pnum] = kINVALID_DOUBLE
            self.MCTruth_particles_Gvy[pnum] = kINVALID_DOUBLE
            self.MCTruth_particles_Gvz[pnum] = kINVALID_DOUBLE
            self.MCTruth_particles_Gvt[pnum] = kINVALID_DOUBLE
            self.MCTruth_particles_px0[pnum] = kINVALID_DOUBLE
            self.MCTruth_particles_py0[pnum] = kINVALID_DOUBLE
            self.MCTruth_particles_pz0[pnum] = kINVALID_DOUBLE
            self.MCTruth_particles_e0[pnum] = kINVALID_DOUBLE
            self.MCTruth_particles_Rescatter[pnum] = kINVALID_INT
            self.MCTruth_particles_polx[pnum] = kINVALID_DOUBLE
            self.MCTruth_particles_poly[pnum] = kINVALID_DOUBLE
            self.MCTruth_particles_polz[pnum] = kINVALID_DOUBLE

        for pnum in xrange(self.MaxParticles*self.MaxDaughters):
            self.MCTruth_particles_Daughters[pnum] = kINVALID_INT

        self.MCTruth_neutrino_CCNC[0] = kINVALID_INT
        self.MCTruth_neutrino_mode[0] = kINVALID_INT
        self.MCTruth_neutrino_interactionType[0] = kINVALID_INT
        self.MCTruth_neutrino_target[0] = kINVALID_INT
        self.MCTruth_neutrino_nucleon[0] = kINVALID_INT
        self.MCTruth_neutrino_quark[0] = kINVALID_INT
        self.MCTruth_neutrino_W[0] =  kINVALID_DOUBLE
        self.MCTruth_neutrino_X[0] =  kINVALID_DOUBLE
        self.MCTruth_neutrino_Y[0] =  kINVALID_DOUBLE
        self.MCTruth_neutrino_Q2[0] = kINVALID_DOUBLE

        self.GTruth_ProbePDG[0]  = kINVALID_INT
        self.GTruth_IsSeaQuark[0]= kINVALID_INT
        self.GTruth_tgtPDG[0]    = kINVALID_INT

        self.GTruth_weight[0]      = kINVALID_DOUBLE
        self.GTruth_probability[0] = kINVALID_DOUBLE
        self.GTruth_Xsec[0]        = kINVALID_DOUBLE
        self.GTruth_fDiffXsec[0]   = kINVALID_DOUBLE
        self.GTruth_vertexX[0]     = kINVALID_DOUBLE
        self.GTruth_vertexY[0]     = kINVALID_DOUBLE
        self.GTruth_vertexZ[0]     = kINVALID_DOUBLE
        self.GTruth_vertexT[0]     = kINVALID_DOUBLE

        self.GTruth_Gscatter[0]   = kINVALID_INT
        self.GTruth_Gint[0]       = kINVALID_INT
        self.GTruth_ResNum[0]     = kINVALID_INT
        self.GTruth_NumPiPlus[0]  = kINVALID_INT
        self.GTruth_NumPi0[0]     = kINVALID_INT
        self.GTruth_NumPiMinus[0] = kINVALID_INT
        self.GTruth_NumProton[0]  = kINVALID_INT
        self.GTruth_NumNeutron[0] = kINVALID_INT
        self.GTruth_IsCharm[0]    = kINVALID_INT

        self.GTruth_gX[0]  = kINVALID_DOUBLE
        self.GTruth_gY[0]  = kINVALID_DOUBLE
        self.GTruth_gZ[0]  = kINVALID_DOUBLE
        self.GTruth_gT[0]  = kINVALID_DOUBLE
        self.GTruth_gW[0]  = kINVALID_DOUBLE
        self.GTruth_gQ2[0] = kINVALID_DOUBLE
        self.GTruth_gq2[0] = kINVALID_DOUBLE
        self.GTruth_ProbeP4x[0]  = kINVALID_DOUBLE
        self.GTruth_ProbeP4y[0]  = kINVALID_DOUBLE
        self.GTruth_ProbeP4z[0]  = kINVALID_DOUBLE
        self.GTruth_ProbeP4E[0]  = kINVALID_DOUBLE
        self.GTruth_HitNucP4x[0] = kINVALID_DOUBLE
        self.GTruth_HitNucP4y[0] = kINVALID_DOUBLE
        self.GTruth_HitNucP4z[0] = kINVALID_DOUBLE
        self.GTruth_HitNucP4E[0] = kINVALID_DOUBLE
        self.GTruth_FShadSystP4x[0] = kINVALID_DOUBLE
        self.GTruth_FShadSystP4y[0] = kINVALID_DOUBLE
        self.GTruth_FShadSystP4z[0] = kINVALID_DOUBLE
        self.GTruth_FShadSystP4E[0] = kINVALID_DOUBLE
        

    def init_nue_precut_tree(self,tree):

        tree.Branch("run"   , self.run   , "run/I")
        tree.Branch("subrun", self.subrun, "subrun/I")
        tree.Branch("event" , self.event , "event/I")

        tree.Branch("num_croi"  , self.num_croi,   "num_croi/I")
        tree.Branch("num_vertex", self.num_vertex, "num_vertex/I")
        tree.Branch("vertex_id" , self.vertex_id,  "vertex_id/I")

        # truth 
        tree.Branch("nu_pdg"    , self.nu_pdg    , "nu_pdg/I")
        tree.Branch("inter_type", self.inter_type, "inter_type/I")
        tree.Branch("inter_mode", self.inter_mode, "inter_mode/I")

        tree.Branch("true_nu_E"      , self.true_nu_E      , "true_nu_E/F")
        tree.Branch("true_vertex"    , self.true_vertex    , "true_vertex[3]/F")
        tree.Branch("true_proton_E"  , self.true_proton_E  , "true_proton_E/F")
        tree.Branch("true_lepton_E"  , self.true_lepton_E  , "true_lepton_E/F")

        tree.Branch("true_proton_P" , self.true_proton_P , "true_proton_P[3]/F")
        tree.Branch("true_lepton_P" , self.true_lepton_P , "true_lepton_P[3]/F")

        tree.Branch("true_opening_angle", self.true_opening_angle, "true_opening_angle/F")
        tree.Branch("true_proton_ylen"  , self.true_proton_ylen  , "true_proton_ylen/F")

        tree.Branch("selected1L1P", self.selected1L1P, "selected1L1P/I")

        # reco mc
        tree.Branch("reco_mc_proton_E"  , self.reco_mc_proton_E  , "reco_mc_proton_E/F")
        tree.Branch("reco_mc_lepton_E"  , self.reco_mc_lepton_E  , "reco_mc_lepton_E/F")
        tree.Branch("reco_mc_total_E"   , self.reco_mc_total_E   , "reco_mc_total_E/F")

        tree.Branch("reco_selected", self.reco_selected, "reco_selected/I")
        tree.Branch("reco_close"   , self.reco_close   , "reco_close/I")
        tree.Branch("reco_on_nu"   , self.reco_on_nu   , "reco_on_nu/I")
        tree.Branch("scedr"        , self.scedr        , "scedr/F")

        tree.Branch("reco_vertex" , self.reco_vertex , "reco_vertex[3]/F")

        # chi2
        tree.Branch("flash_chi2", self.chi2, "flash_chi2/F")

        # reco track
        tree.Branch("reco_proton_id"      , self.reco_proton_id   , "reco_proton_id/I")
        tree.Branch("reco_proton_E"       , self.reco_proton_E    , "reco_proton_E/F")
        tree.Branch("reco_proton_theta"   , self.reco_proton_theta, "reco_proton_theta/F")
        tree.Branch("reco_proton_phi"     , self.reco_proton_phi  , "reco_proton_phi/F")
        tree.Branch("reco_proton_dEdx"    , self.reco_proton_dEdx , "reco_proton_dEdx/F")
        tree.Branch("reco_proton_length"  , self.reco_proton_length  , "reco_proton_length/F")
        tree.Branch("reco_proton_mean_pixel_dist",self.reco_proton_mean_pixel_dist, "reco_proton_mean_pixel_dist/F")
        tree.Branch("reco_proton_width"      , self.reco_proton_width, "reco_proton_width/F")
        tree.Branch("reco_proton_area"       , self.reco_proton_area, "reco_proton_area/F")
        tree.Branch("reco_proton_qsum"       , self.reco_proton_qsum, "reco_proton_qsum/F")
        tree.Branch("reco_proton_shower_frac", self.reco_proton_shower_frac, "reco_proton_shower_frac/F")

        tree.Branch("reco_electron_id"     , self.reco_electron_id   , "reco_electron_id/I")
        tree.Branch("reco_electron_E"      , self.reco_electron_E    , "reco_electron_E/F")
        tree.Branch("reco_electron_theta"  , self.reco_electron_theta, "reco_electron_theta/F")
        tree.Branch("reco_electron_phi"    , self.reco_electron_phi  , "reco_electron_phi/F")
        tree.Branch("reco_electron_dEdx"   , self.reco_electron_dEdx , "reco_electron_dEdx/F")
        tree.Branch("reco_electron_length" , self.reco_electron_length  , "reco_electron_length/F")
        tree.Branch("reco_electron_mean_pixel_dist" , self.reco_electron_mean_pixel_dist, "reco_electron_mean_pixel_dist/F")
        tree.Branch("reco_electron_width"      , self.reco_electron_width, "reco_electron_width/F")
        tree.Branch("reco_electron_area"       , self.reco_electron_area, "reco_electron_area/F")
        tree.Branch("reco_electron_qsum"       , self.reco_electron_qsum, "reco_electron_qsum/F")
        tree.Branch("reco_electron_shower_frac", self.reco_electron_shower_frac, "reco_electron_shower_frac/F")

        # reco combined
        tree.Branch("reco_total_E", self.reco_total_E, "reco_total_E/F")
        
        # reco PID
        tree.Branch("pid_inferred", self.inferred    , "pid_inferred/I")
        tree.Branch("eminus_score", self.eminus_score, "eminus_score[3]/F")
        tree.Branch("gamma_score" , self.gamma_score , "gamma_score[3]/F")
        tree.Branch("muon_score"  , self.muon_score  , "muon_score[3]/F")
        tree.Branch("pion_score"  , self.pion_score  , "pion_score[3]/F")
        tree.Branch("proton_score", self.proton_score, "proton_score[3]/F")

        tree.Branch("eminus_score_vtx", self.eminus_score_vtx, "eminus_score_vtx[3]/F")
        tree.Branch("gamma_score_vtx" , self.gamma_score_vtx , "gamma_score_vtx[3]/F")
        tree.Branch("muon_score_vtx"  , self.muon_score_vtx  , "muon_score_vtx[3]/F")
        tree.Branch("pion_score_vtx"  , self.pion_score_vtx  , "pion_score_vtx[3]/F")
        tree.Branch("proton_score_vtx", self.proton_score_vtx, "proton_score_vtx[3]/F")

        tree.Branch("nue_perceptron", self.nue_perceptron, "nue_perceptron[3]/F")

    def init_nue_tree(self,tree):
        
        tree.Branch("run"   , self.run   , "run/I")
        tree.Branch("subrun", self.subrun, "subrun/I")
        tree.Branch("event" , self.event , "event/I")

        tree.Branch("num_croi"  , self.num_croi,   "num_croi/I")
        tree.Branch("num_vertex", self.num_vertex, "num_vertex/I")
        tree.Branch("vertex_id" , self.vertex_id,  "vertex_id/I")

        # truth 
        tree.Branch("nu_pdg"    , self.nu_pdg    , "nu_pdg/I")
        tree.Branch("inter_type", self.inter_type, "inter_type/I")
        tree.Branch("inter_mode", self.inter_mode, "inter_mode/I")

        tree.Branch("true_nu_E"      , self.true_nu_E      , "true_nu_E/F")
        tree.Branch("true_vertex"    , self.true_vertex    , "true_vertex[3]/F")
        tree.Branch("true_proton_E"  , self.true_proton_E  , "true_proton_E/F")
        tree.Branch("true_lepton_E"  , self.true_lepton_E  , "true_lepton_E/F")

        tree.Branch("true_proton_P" , self.true_proton_P , "true_proton_P[3]/F")
        tree.Branch("true_lepton_P" , self.true_lepton_P , "true_lepton_P[3]/F")

        tree.Branch("true_opening_angle", self.true_opening_angle, "true_opening_angle/F")
        tree.Branch("true_proton_ylen"  , self.true_proton_ylen  , "true_proton_ylen/F")

        tree.Branch("selected1L1P", self.selected1L1P, "selected1L1P/I")

        # reco mc
        tree.Branch("reco_mc_proton_E"  , self.reco_mc_proton_E  , "reco_mc_proton_E/F")
        tree.Branch("reco_mc_lepton_E"  , self.reco_mc_lepton_E  , "reco_mc_lepton_E/F")
        tree.Branch("reco_mc_total_E"   , self.reco_mc_total_E   , "reco_mc_total_E/F")

        tree.Branch("reco_selected", self.reco_selected, "reco_selected/I")
        tree.Branch("reco_close"   , self.reco_close   , "reco_close/I")
        tree.Branch("reco_on_nu"   , self.reco_on_nu   , "reco_on_nu/I")
        tree.Branch("scedr"        , self.scedr        , "scedr/F")

        tree.Branch("reco_vertex" , self.reco_vertex , "reco_vertex[3]/F")

        # LL
        tree.Branch("LL_dist", self.LL_dist, "LL_dist/F")
        tree.Branch("LLc_e"  , self.LLc_e  , "LLc_e/F")
        tree.Branch("LLc_p"  , self.LLc_p  , "LLc_p/F")
        tree.Branch("LLe_e"  , self.LLe_e  , "LLe_e/F")
        tree.Branch("LLe_p"  , self.LLe_p  , "LLe_p/F")


        # reco track
        tree.Branch("reco_proton_id"      , self.reco_proton_id   , "reco_proton_id/I")
        tree.Branch("reco_proton_E"       , self.reco_proton_E    , "reco_proton_E/F")
        tree.Branch("reco_proton_theta"   , self.reco_proton_theta, "reco_proton_theta/F")
        tree.Branch("reco_proton_phi"     , self.reco_proton_phi  , "reco_proton_phi/F")
        tree.Branch("reco_proton_dEdx"    , self.reco_proton_dEdx , "reco_proton_dEdx/F")
        tree.Branch("reco_proton_length"  , self.reco_proton_length  , "reco_proton_length/F")
        tree.Branch("reco_proton_mean_pixel_dist",self.reco_proton_mean_pixel_dist, "reco_proton_mean_pixel_dist/F")
        tree.Branch("reco_proton_width"      , self.reco_proton_width, "reco_proton_width/F")
        tree.Branch("reco_proton_area"       , self.reco_proton_area, "reco_proton_area/F")
        tree.Branch("reco_proton_qsum"       , self.reco_proton_qsum, "reco_proton_qsum/F")
        tree.Branch("reco_proton_shower_frac", self.reco_proton_shower_frac, "reco_proton_shower_frac/F")

        tree.Branch("reco_electron_id"     , self.reco_electron_id   , "reco_electron_id/I")
        tree.Branch("reco_electron_E"      , self.reco_electron_E    , "reco_electron_E/F")
        tree.Branch("reco_electron_theta"  , self.reco_electron_theta, "reco_electron_theta/F")
        tree.Branch("reco_electron_phi"    , self.reco_electron_phi  , "reco_electron_phi/F")
        tree.Branch("reco_electron_dEdx"   , self.reco_electron_dEdx , "reco_electron_dEdx/F")
        tree.Branch("reco_electron_length" , self.reco_electron_length  , "reco_electron_length/F")
        tree.Branch("reco_electron_mean_pixel_dist" , self.reco_electron_mean_pixel_dist, "reco_electron_mean_pixel_dist/F")
        tree.Branch("reco_electron_width"      , self.reco_electron_width, "reco_electron_width/F")
        tree.Branch("reco_electron_area"       , self.reco_electron_area, "reco_electron_area/F")
        tree.Branch("reco_electron_qsum"       , self.reco_electron_qsum, "reco_electron_qsum/F")
        tree.Branch("reco_electron_shower_frac", self.reco_electron_shower_frac, "reco_electron_shower_frac/F")

        # reco combined
        tree.Branch("reco_total_E", self.reco_total_E, "reco_total_E/F")
        
        # reco PID
        tree.Branch("pid_inferred", self.inferred    , "pid_inferred/I")
        tree.Branch("eminus_score", self.eminus_score, "eminus_score[3]/F")
        tree.Branch("gamma_score" , self.gamma_score , "gamma_score[3]/F")
        tree.Branch("muon_score"  , self.muon_score  , "muon_score[3]/F")
        tree.Branch("pion_score"  , self.pion_score  , "pion_score[3]/F")
        tree.Branch("proton_score", self.proton_score, "proton_score[3]/F")

        tree.Branch("eminus_score_vtx", self.eminus_score_vtx, "eminus_score_vtx[3]/F")
        tree.Branch("gamma_score_vtx" , self.gamma_score_vtx , "gamma_score_vtx[3]/F")
        tree.Branch("muon_score_vtx"  , self.muon_score_vtx  , "muon_score_vtx[3]/F")
        tree.Branch("pion_score_vtx"  , self.pion_score_vtx  , "pion_score_vtx[3]/F")
        tree.Branch("proton_score_vtx", self.proton_score_vtx, "proton_score_vtx[3]/F")

        tree.Branch("nue_perceptron", self.nue_perceptron, "nue_perceptron[3]/F")



    def init_numu_tree(self,tree):
        
        tree.Branch("run"   , self.run   , "run/I")
        tree.Branch("subrun", self.subrun, "subrun/I")
        tree.Branch("event" , self.event , "event/I")

        tree.Branch("num_croi"  , self.num_croi,   "num_croi/I")
        tree.Branch("num_vertex", self.num_vertex, "num_vertex/I")
        tree.Branch("vertex_id" , self.vertex_id,  "vertex_id/I")

        tree.Branch("nu_pdg"      , self.nu_pdg      , "nu_pdg/I")
        tree.Branch("selected1L1P", self.selected1L1P, "selected1L1P/I")
        tree.Branch("true_nu_E"   , self.true_nu_E   , "true_nu_E/F")
        tree.Branch("true_vertex" , self.true_vertex , "true_vertex[3]/F")

        tree.Branch("inter_type", self.inter_type, "inter_type/I")
        tree.Branch("inter_mode", self.inter_mode, "inter_mode/I")

        tree.Branch("true_proton_E" , self.true_proton_E , "true_proton_E/F")
        tree.Branch("true_lepton_E" , self.true_lepton_E , "true_lepton_E/F")

        tree.Branch("true_proton_P" , self.true_proton_P , "true_proton_P[3]/F")
        tree.Branch("true_lepton_P" , self.true_lepton_P , "true_lepton_P[3]/F")

        tree.Branch("reco_selected", self.reco_selected, "reco_selected/I")
        tree.Branch("reco_close"   , self.reco_close   , "reco_close/I")
        tree.Branch("reco_on_nu"   , self.reco_on_nu   , "reco_on_nu/I")
        tree.Branch("scedr"        , self.scedr        , "scedr/F")

        tree.Branch("reco_vertex" , self.reco_vertex , "reco_vertex[3]/F")

        tree.Branch("CosmicLL"     , self.CosmicLL     , "CosmicLL/F")
        tree.Branch("NuBkgLL"      , self.NuBkgLL      , "NuBkgLL/F")
        tree.Branch("CCpi0LL"      , self.CCpi0LL      , "CCpi0LL/F")
        tree.Branch("PassCuts"     , self.PassCuts     , "PassCuts/I")
        tree.Branch("VtxAlgo"      , self.VtxAlgo      , "VtxAlgo/I")
        tree.Branch("NTracks"      , self.NTracks      , "NTracks/I")
        tree.Branch("N5cmTracks"   , self.N5cmTracks   , "N5cmTracks/I")
        tree.Branch("InFiducial"   , self.InFiducial   , "InFiducial/I")
        tree.Branch("Good3DReco"   , self.Good3DReco   , "Good3DReco/I")
        tree.Branch("AnythingRecod", self.AnythingRecod, "AnythingRecod/I")

        tree.Branch("Muon_id"          , self.Muon_id          , "Muon_id/I")
        tree.Branch("Muon_PhiReco"     , self.Muon_PhiReco     , "Muon_PhiReco/F")
        tree.Branch("Muon_ThetaReco"   , self.Muon_ThetaReco   , "Muon_ThetaReco/F")
        tree.Branch("Muon_TrackLength" , self.Muon_TrackLength , "Muon_TrackLength/F")
        tree.Branch("Muon_dQdx"        , self.Muon_dQdx        , "Muon_dQdx/F")
        tree.Branch("Muon_Trunc_dQdx1" , self.Muon_Trunc_dQdx1 , "Muon_Trunc_dQdx1/F")
        tree.Branch("Muon_Trunc_dQdx3" , self.Muon_Trunc_dQdx3 , "Muon_Trunc_dQdx3/F")
        tree.Branch("Muon_IonPerLen"   , self.Muon_IonPerLen   , "Muon_IonPerLen/F")
        tree.Branch("Muon_E"           , self.Muon_Edep        , "Muon_E/F")

        tree.Branch("Proton_id"          , self.Proton_id          , "Proton_id/I")
        tree.Branch("Proton_PhiReco"     , self.Proton_PhiReco     , "Proton_PhiReco/F")
        tree.Branch("Proton_ThetaReco"   , self.Proton_ThetaReco   , "Proton_ThetaReco/F")
        tree.Branch("Proton_TrackLength" , self.Proton_TrackLength , "Proton_TrackLength/F")
        tree.Branch("Proton_dQdx"        , self.Proton_dQdx        , "Proton_dQdx/F")
        tree.Branch("Proton_Trunc_dQdx1" , self.Proton_Trunc_dQdx1 , "Proton_Trunc_dQdx1/F")
        tree.Branch("Proton_Trunc_dQdx3" , self.Proton_Trunc_dQdx3 , "Proton_Trunc_dQdx3/F")
        tree.Branch("Proton_IonPerLen"   , self.Proton_IonPerLen   , "Proton_IonPerLen/F")
        tree.Branch("Proton_E"           , self.Proton_Edep        , "Proton_E/F")

        tree.Branch("pid_inferred", self.inferred    , "pid_inferred/I")
        tree.Branch("eminus_score", self.eminus_score, "eminus_score[3]/F")
        tree.Branch("gamma_score" , self.gamma_score , "gamma_score[3]/F")
        tree.Branch("muon_score"  , self.muon_score  , "muon_score[3]/F")
        tree.Branch("pion_score"  , self.pion_score  , "pion_score[3]/F")
        tree.Branch("proton_score", self.proton_score, "proton_score[3]/F")

        tree.Branch("eminus_score_vtx", self.eminus_score_vtx, "eminus_score_vtx[3]/F")
        tree.Branch("gamma_score_vtx" , self.gamma_score_vtx , "gamma_score_vtx[3]/F")
        tree.Branch("muon_score_vtx"  , self.muon_score_vtx  , "muon_score_vtx[3]/F")
        tree.Branch("pion_score_vtx"  , self.pion_score_vtx  , "pion_score_vtx[3]/F")
        tree.Branch("proton_score_vtx", self.proton_score_vtx, "proton_score_vtx[3]/F")

        tree.Branch("nue_perceptron", self.nue_perceptron, "nue_perceptron[3]/F")


    def init_andy_tree(self,tree):

        tree.Branch("run"   , self.run   , "run/I")
        tree.Branch("subrun", self.subrun, "subrun/I")
        tree.Branch("event" , self.event , "event/I")
        
        tree.Branch("MCFlux_NuPosX",   self.MCFlux_NuPosX    , "MCFlux_NuPosX/D");
        tree.Branch("MCFlux_NuPosY",   self.MCFlux_NuPosY    , "MCFlux_NuPosY/D");
        tree.Branch("MCFlux_NuPosZ",   self.MCFlux_NuPosZ    , "MCFlux_NuPosZ/D");
        tree.Branch("MCFlux_NuMomX",   self.MCFlux_NuMomX    , "MCFlux_NuMomX/D");
        tree.Branch("MCFlux_NuMomY",   self.MCFlux_NuMomY    , "MCFlux_NuMomY/D");
        tree.Branch("MCFlux_NuMomZ",   self.MCFlux_NuMomZ    , "MCFlux_NuMomZ/D");
        tree.Branch("MCFlux_NuMomE",   self.MCFlux_NuMomE    , "MCFlux_NuMomE/D");
        tree.Branch("MCFlux_ntype",    self.MCFlux_ntype     , "MCFlux_ntype/I");
        tree.Branch("MCFlux_ptype",    self.MCFlux_ptype     , "MCFlux_ptype/I");
        tree.Branch("MCFlux_nimpwt",   self.MCFlux_nimpwt    , "MCFlux_nimpwt/D");
        tree.Branch("MCFlux_dk2gen",   self.MCFlux_dk2gen    , "MCFlux_dk2gen/D");
        tree.Branch("MCFlux_nenergyn", self.MCFlux_nenergyn  , "MCFlux_nenergyn/D");
        tree.Branch("MCFlux_tpx",      self.MCFlux_tpx       , "MCFlux_tpx/D");
        tree.Branch("MCFlux_tpy",      self.MCFlux_tpy       , "MCFlux_tpy/D");
        tree.Branch("MCFlux_tpz",      self.MCFlux_tpz       , "MCFlux_tpz/D");
        tree.Branch("MCFlux_tptype",   self.MCFlux_tptype    , "MCFlux_tptype/I");
        tree.Branch("MCFlux_vx",       self.MCFlux_vx        , "MCFlux_vx/D");
        tree.Branch("MCFlux_vy",       self.MCFlux_vy        , "MCFlux_vy/D");
        tree.Branch("MCFlux_vz",       self.MCFlux_vz        , "MCFlux_vz/D");
        
        tree.Branch("MCTruth_NParticles",           self.MCTruth_NParticles,"MCTruth_NParticles/I");
        tree.Branch("MCTruth_particles_TrackId",    self.MCTruth_particles_TrackId, "MCTruth_particles_TrackId[50]/I");
        tree.Branch("MCTruth_particles_PdgCode",    self.MCTruth_particles_PdgCode, "MCTruth_particles_PdgCode[50]/I");
        tree.Branch("MCTruth_particles_Mother",     self.MCTruth_particles_Mother, "MCTruth_particles_Mother[50]/I");
        tree.Branch("MCTruth_particles_StatusCode", self.MCTruth_particles_StatusCode, "MCTruth_particles_StatusCode[50]/I");
        tree.Branch("MCTruth_particles_NDaughters", self.MCTruth_particles_NDaughters, "MCTruth_particles_NDaughters[50]/I");
        tree.Branch("MCTruth_particles_Daughters",  self.MCTruth_particles_Daughters, "MCTruth_particles_Daughters[50][100]/I");
        tree.Branch("MCTruth_particles_Gvx",        self.MCTruth_particles_Gvx, "MCTruth_particles_Gvx[50]/D");
        tree.Branch("MCTruth_particles_Gvy",        self.MCTruth_particles_Gvy, "MCTruth_particles_Gvy[50]/D");
        tree.Branch("MCTruth_particles_Gvz",        self.MCTruth_particles_Gvz, "MCTruth_particles_Gvz[50]/D");
        tree.Branch("MCTruth_particles_Gvt",        self.MCTruth_particles_Gvt, "MCTruth_particles_Gvt[50]/D");
        tree.Branch("MCTruth_particles_px0",        self.MCTruth_particles_px0, "MCTruth_particles_px0[50]/D");
        tree.Branch("MCTruth_particles_py0",        self.MCTruth_particles_py0, "MCTruth_particles_py0[50]/D");
        tree.Branch("MCTruth_particles_pz0",        self.MCTruth_particles_pz0, "MCTruth_particles_pz0[50]/D");
        tree.Branch("MCTruth_particles_e0",         self.MCTruth_particles_e0, "MCTruth_particles_e0[50]/D");
        tree.Branch("MCTruth_particles_Rescatter",  self.MCTruth_particles_Rescatter, "MCTruth_particles_Rescatter[50]/I");
        tree.Branch("MCTruth_particles_polx",       self.MCTruth_particles_polx, "MCTruth_particles_polx[50]/D");
        tree.Branch("MCTruth_particles_poly",       self.MCTruth_particles_poly, "MCTruth_particles_poly[50]/D");
        tree.Branch("MCTruth_particles_polz",       self.MCTruth_particles_polz, "MCTruth_particles_polz[50]/D");
    
        tree.Branch("MCTruth_neutrino_CCNC",            self.MCTruth_neutrino_CCNC, "MCTruth_neutrino_CCNC/I");
        tree.Branch("MCTruth_neutrino_mode",            self.MCTruth_neutrino_mode, "MCTruth_neutrino_mode/I");
        tree.Branch("MCTruth_neutrino_interactionType", self.MCTruth_neutrino_interactionType, "MCTruth_neutrino_interactionType/I");
        tree.Branch("MCTruth_neutrino_target",          self.MCTruth_neutrino_target, "MCTruth_neutrino_target/I");         
        tree.Branch("MCTruth_neutrino_nucleon",         self.MCTruth_neutrino_nucleon, "MCTruth_neutrino_nucleon/I");        
        tree.Branch("MCTruth_neutrino_quark",           self.MCTruth_neutrino_quark, "MCTruth_neutrino_quark/I");          
        tree.Branch("MCTruth_neutrino_W",               self.MCTruth_neutrino_W, "MCTruth_neutrino_W/D");              
        tree.Branch("MCTruth_neutrino_X",               self.MCTruth_neutrino_X, "MCTruth_neutrino_X/D");              
        tree.Branch("MCTruth_neutrino_Y",               self.MCTruth_neutrino_Y, "MCTruth_neutrino_Y/D");              
        tree.Branch("MCTruth_neutrino_Q2",              self.MCTruth_neutrino_Q2, "MCTruth_neutrino_Q2/D");             
        
        tree.Branch("GTruth_ProbePDG"    , self.GTruth_ProbePDG, "GTruth_ProbePDG/I");                  
        tree.Branch("GTruth_IsSeaQuark"  , self.GTruth_IsSeaQuark, "GTruth_IsSeaQuark/I");                
        tree.Branch("GTruth_tgtPDG"      , self.GTruth_tgtPDG, "GTruth_tgtPDG/I");                    
        tree.Branch("GTruth_weight"      , self.GTruth_weight, "GTruth_weight/D");                    
        tree.Branch("GTruth_probability" , self.GTruth_probability, "GTruth_probability/D");               
        tree.Branch("GTruth_Xsec"        , self.GTruth_Xsec, "GTruth_Xsec/D");                      
        tree.Branch("GTruth_fDiffXsec"   , self.GTruth_fDiffXsec, "GTruth_fDiffXsec/D");                 
        tree.Branch("GTruth_vertexX"     , self.GTruth_vertexX, "GTruth_vertexX/D");                   
        tree.Branch("GTruth_vertexY"     , self.GTruth_vertexY, "GTruth_vertexY/D");                   
        tree.Branch("GTruth_vertexZ"     , self.GTruth_vertexZ, "GTruth_vertexZ/D");                   
        tree.Branch("GTruth_vertexT"     , self.GTruth_vertexT, "GTruth_vertexT/D");                   
        tree.Branch("GTruth_Gscatter"    , self.GTruth_Gscatter, "GTruth_Gscatter/I");                  
        tree.Branch("GTruth_Gint"        , self.GTruth_Gint, "GTruth_Gint/I");                      
        tree.Branch("GTruth_ResNum"      , self.GTruth_ResNum, "GTruth_ResNum/I");                    
        tree.Branch("GTruth_NumPiPlus"   , self.GTruth_NumPiPlus, "GTruth_NumPiPlus/I");                 
        tree.Branch("GTruth_NumPi0"      , self.GTruth_NumPi0, "GTruth_NumPi0/I");                    
        tree.Branch("GTruth_NumPiMinus"  , self.GTruth_NumPiMinus, "GTruth_NumPiMinus/I");                
        tree.Branch("GTruth_NumProton"   , self.GTruth_NumProton, "GTruth_NumProton/I");                 
        tree.Branch("GTruth_NumNeutron"  , self.GTruth_NumNeutron, "GTruth_NumNeutron/I");                
        tree.Branch("GTruth_IsCharm"     , self.GTruth_IsCharm, "GTruth_IsCharm/I");                   
        tree.Branch("GTruth_gX" , self.GTruth_gX  , "GTruth_gX/D");                        
        tree.Branch("GTruth_gY" , self.GTruth_gY  , "GTruth_gY/D");                        
        tree.Branch("GTruth_gZ" , self.GTruth_gZ  , "GTruth_gZ/D");                        
        tree.Branch("GTruth_gT" , self.GTruth_gT  , "GTruth_gT/D");                        
        tree.Branch("GTruth_gW" , self.GTruth_gW  , "GTruth_gW/D");                        
        tree.Branch("GTruth_gQ2", self.GTruth_gQ2 , "GTruth_gQ2/D");                       
        tree.Branch("GTruth_gq2", self.GTruth_gq2 , "GTruth_gq2/D");                       
        tree.Branch("GTruth_ProbePDG" , self.GTruth_ProbePDG  , "GTruth_ProbePDG/I");                  
        tree.Branch("GTruth_ProbeP4x" , self.GTruth_ProbeP4x  , "GTruth_ProbeP4x/D");                  
        tree.Branch("GTruth_ProbeP4y" , self.GTruth_ProbeP4y  , "GTruth_ProbeP4y/D");                  
        tree.Branch("GTruth_ProbeP4z" , self.GTruth_ProbeP4z  , "GTruth_ProbeP4z/D");                  
        tree.Branch("GTruth_ProbeP4E" , self.GTruth_ProbeP4E  , "GTruth_ProbeP4E/D");                  
        tree.Branch("GTruth_HitNucP4x", self.GTruth_HitNucP4x , "GTruth_HitNucP4x/D");                 
        tree.Branch("GTruth_HitNucP4y", self.GTruth_HitNucP4y , "GTruth_HitNucP4y/D");                 
        tree.Branch("GTruth_HitNucP4z", self.GTruth_HitNucP4z , "GTruth_HitNucP4z/D");                 
        tree.Branch("GTruth_HitNucP4E", self.GTruth_HitNucP4E , "GTruth_HitNucP4E/D");                 
        tree.Branch("GTruth_FShadSystP4x", self.GTruth_FShadSystP4x, "GTruth_FShadSystP4x/D");              
        tree.Branch("GTruth_FShadSystP4y", self.GTruth_FShadSystP4y, "GTruth_FShadSystP4y/D");              
        tree.Branch("GTruth_FShadSystP4z", self.GTruth_FShadSystP4z, "GTruth_FShadSystP4z/D");              
        tree.Branch("GTruth_FShadSystP4E", self.GTruth_FShadSystP4E, "GTruth_FShadSystP4E/D");

    def init_segment_tree(self,tree):

        tree.Branch("run"   , self.run   , "run/I")
        tree.Branch("subrun", self.subrun, "subrun/I")
        tree.Branch("event" , self.event , "event/I")

        tree.Branch("nu_pdg"    , self.nu_pdg    , "nu_pdg/I")
        tree.Branch("inter_type", self.inter_type, "inter_type/I")
        tree.Branch("inter_mode", self.inter_mode, "inter_mode/I")

        tree.Branch("true_nu_E"      , self.true_nu_E      , "true_nu_E/F")
        tree.Branch("true_vertex"    , self.true_vertex    , "true_vertex[3]/F")
        tree.Branch("true_vertex_sce", self.true_vertex_sce, "true_vertex_sce[3]/F")
        tree.Branch("true_proton_E"  , self.true_proton_E  , "true_proton_E/F")
        tree.Branch("true_lepton_E"  , self.true_lepton_E  , "true_lepton_E/F")

        tree.Branch("true_proton_P" , self.true_proton_P , "true_proton_P[3]/F")
        tree.Branch("true_lepton_P" , self.true_lepton_P , "true_lepton_P[3]/F")

        tree.Branch("selected1L1P", self.selected1L1P, "selected1L1P/I")

