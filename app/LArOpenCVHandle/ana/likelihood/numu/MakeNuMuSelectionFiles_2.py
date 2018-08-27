import ROOT
from ROOT import TFile,TTree
import matplotlib.pyplot as plt
import pickle
import numpy as np
from numpy import mean
from math import sqrt,acos,cos,sin,pi,exp,log
from sys import argv
from array import array
import os,sys

## When you call this program it must have the following format... ##
## python MakeNuMuSelectionFiles.py INPUT1 INPUT2....              ##
## with ...                                                        ##
##    INPUT1 = 3D Track Reco ROOT file                             ##
##    INPUT2 = Vertex reconstruction ana ROOT file                 ##
##    INPUT3 = pickle containing numu PDFs (prob functions)        ##
##    INPUT4 = Output directory                                    ##
## --------------------------------------------------------------- ##

with open(argv[3],'rb') as handle: pdfs = pickle.load(handle)          # Load LL histograms for 1mu1p cosmic differentiation

# --- Some functions for internal analysis use -------------------- #
def Compute3DAngle(theta0,theta1,phi0,phi1):

    return acos(cos(theta0)*cos(theta1)+sin(theta0)*sin(theta1)*cos(phi0-phi1))
    
def ComputeVarProb(LLpdf,value):

    foundBin = False
    spacing  = LLpdf[0][1]-LLpdf[0][0]
    for i,x in enumerate(LLpdf[0]):
        if value >= x and (value-x) < spacing:
            valBin   = i
            foundBin = True
            break
        
    if foundBin == True:
        if LLpdf[1][valBin] !=0:
            return log(LLpdf[1][valBin])
        else:
            return -1
    else:
        return 0

    
def GetBin(bins,value):

    step = bins[1]-bins[0]
    
    if value > max(bins) or value < min(bins):
        return -1

    for i,x in enumerate(bins):
        if value >= x and value < x + step:
            return i

        
def BeyesFactor(inputDict,model='cosmic'):

    LL = 0
    variables = inputDict.keys()

    cosVars = ['etaHist','openAngHist','thetaHist','phiTHist','pTHist','xHist','phiDiffHist','sphHist','alphaHist']#,'dEHist','phiDiffHist','pPhiHist']
    nuVars  = ['etaHist','openAngHist','thetaHist','phiTHist','pTHist','xHist','phiDiffHist','sphHist','alphaHist']#,'dEHist','phiDiffHist','pPhiHist']

    for var in variables:

        if model=='cosmic' and var not in cosVars: continue
        if model=='nuBack' and var not in nuVars: continue
        
        bins = pdfs[var][1]

        sigNorm  = sum(x for x in pdfs[var][0][0])
        cosNorm  = sum(x for x in pdfs[var][0][1])-sigNorm
        nuNorm   = sum(x for x in pdfs[var][0][2])-cosNorm-sigNorm
        
        whichBin = GetBin(bins,inputDict[var]) 

        if whichBin == -1: continue
        
        PSgE  = max([pdfs[var][0][0][whichBin] / sigNorm,0.001])
        PCgE  = max([(pdfs[var][0][1][whichBin] - pdfs[var][0][0][whichBin]) / cosNorm,0.001])
        PNgE  = max([(pdfs[var][0][2][whichBin] - pdfs[var][0][1][whichBin]) / nuNorm,0.001])
        
        if model=='cosmic':
            LL+= log(PSgE/PCgE)
        elif model=='nuBack':
            LL+= log(PSgE/PNgE)

    return LL

def VtxInFid(vtxX,vtxY,vtxZ,edgeCut=10):

    xmin =  0      + edgeCut
    xmax =  256.25 - edgeCut
    ymin = -116.5  + edgeCut
    ymax =  116.5  - edgeCut
    zmin =  0      + edgeCut
    zmax =  1036.8 - edgeCut
    
    if vtxX > xmax or vtxX < xmin or vtxY > ymax or vtxY < ymin or vtxZ > zmax or vtxZ < zmin:
        return False
    else:
        return True

def GetPhiT(El,Ep,Thl,Thp,Phl,Php):

    MuMass = 105.658
    PMass  = 938.673

    Pl  = sqrt((El+MuMass)**2 - MuMass**2)
    Pp  = sqrt((Ep+PMass)**2 - PMass**2)
    
    Plt = [Pl*sin(Thl)*cos(Phl),Pl*sin(Thl)*sin(Phl),0]
    Ppt = [Pp*sin(Thp)*cos(Php),Pp*sin(Thp)*sin(Php),0]
    
    PltM = sqrt(Plt[0]**2+Plt[1]**2)
    PptM = sqrt(Ppt[0]**2+Ppt[1]**2)

    if PltM == 0 or PptM == 0:
        return 9999
    
    phit = acos(-1.0*(Plt[0]*Ppt[0]+Plt[1]*Ppt[1])/(PltM*PptM))
    
    return phit

def pTrans(El,Ep,Thl,Thp,Phl,Php):

    MuMass = 105.658
    PMass  = 938.673

    Pl  = sqrt((El+MuMass)**2 - MuMass**2)
    Pp  = sqrt((Ep+PMass )**2 - PMass**2 )
    
    Plt = [Pl*sin(Thl)*cos(Phl),Pl*sin(Thl)*sin(Phl),0]
    Ppt = [Pp*sin(Thp)*cos(Php),Pp*sin(Thp)*sin(Php),0]
    
    Pt  = [Ppt[0]+Plt[0],Ppt[1]+Plt[1],0]
                                
    PtMag = sqrt(Pt[0]**2 + Pt[1]**2)

    return PtMag

def alphaT(El,Ep,Thl,Thp,Phl,Php):

    MuMass = 105.658
    PMass  = 938.673

    Pl  = sqrt((El+MuMass)**2 - MuMass**2)
    Pp  = sqrt((Ep+PMass)**2 - PMass**2)

    Plt = [Pl*sin(Thl)*cos(Phl),Pl*sin(Thl)*sin(Phl),0]
    Ppt = [Pp*sin(Thp)*cos(Php),Pp*sin(Thp)*sin(Php),0]

    PltM = sqrt(Plt[0]**2+Plt[1]**2)
    PptM = sqrt(Ppt[0]**2+Ppt[1]**2)

    Pt  = [Ppt[0]+Plt[0],Ppt[1]+Plt[1],0]
    PtMag = sqrt(Pt[0]**2 + Pt[1]**2)

    if PltM == 0 or PptM == 0:
        return 9999
    
    alphat = acos(-1.0*(Plt[0]*Pt[0]+Plt[1]*Pt[1])/(PtMag*PltM))

    return alphat

def ECCQE(KE,theta,pid="muon"):
    
    Mn  = 939.5654
    Mp  = 938.2721
    Mmu = 105.6584
    B   = 40.0
    
    if pid == "muon":
        Muon_theta = theta
        Muon_KE    = KE
        EnuQE=  0.5*( (2*(Mn-B)*(Muon_KE+Mmu)-((Mn-B)*(Mn-B)+Mmu*Mmu-Mp*Mp  ))/( (Mn-B)-(Muon_KE+Mmu)+sqrt((((Muon_KE+Mmu)*(Muon_KE+Mmu))-(Mmu*Mmu)))*cos(Muon_theta  ) ) );

    elif pid == "proton":
        Proton_theta = theta
        Proton_KE    = KE
        EnuQE = 0.5*( (2*(Mn-B)*(Proton_KE+Mp) -((Mn-B)*(Mn-B)+Mp*Mp  -Mmu*Mmu))/( (Mn-B)-(Proton_KE+Mp) +sqrt((((Proton_KE+Mp) *(Proton_KE+Mp) )-(Mp *Mp )))*cos(Proton_theta) ) );

    return EnuQE

def ECal(KEp,KEmu):

    Mn  = 939.5654
    Mmu = 105.6584
    Mp  = 938.2721
    B   = 40
    
    EnuCal = KEp+KEmu+B+Mmu+(Mn-Mp)

    return EnuCal

def Q2(Enu,Emu,theta):
    # Feed in MeV, which is what is usually in the trees
    
    Enu = Enu/1000.
    Emu = Emu/1000. 
    Mmu = 105.6584/1000.
    Pmu = sqrt((Emu+Mmu)**2 - Mmu**2)

    return -1.0*Mmu**2 + 2*Enu*(Emu + Mmu - Pmu*cos(theta))

def OpenAngle(th1,th2,phi1,phi2):

    return cos(th1)*cos(th2)+sin(th1)*sin(th2)*cos(phi1-phi2)

def PhiDiff(phi1,phi2):

    bigPhi = max([phi1,phi2])
    lilPhi = min([phi1,phi2])

    return bigPhi - lilPhi

def edgeCut(wallDistVec):

    vtxEdge = 5
    trkEdge = 15
    
    if abs(wallDistVec[0]-wallDistVec[1]) < 5:
        if min(wallDistVec) < vtxEdge:
            return True
        else:
            return False
    else:
        if min(wallDistVec) < trkEdge:
            return True
        else:
            return False

# ----------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------- #

# --- Open 2 Ana Files, one from Vertex Reco and one from Track Reco --- #
TrkRecoFile = TFile(argv[1])
VtxRecoFile = TFile(argv[2])
# ---------------------------------------------------------------------- #

# --- Load relevant analysis trees from track and vertex ana files ----- #
TrkTree  = TrkRecoFile.Get("_recoTree")
VtxTree  = VtxRecoFile.Get("VertexTree")
shpTree  = VtxRecoFile.Get("ShapeAnalysis")
shrTree  = VtxRecoFile.Get("SecondShowerAnalysis")

VtxTree.AddFriend(shpTree)
VtxTree.AddFriend(shrTree)
# ---------------------------------------------------------------------- #

# --- Perform some alignment checks to be sure the ana files match up -- #

fileTag1 = int(os.path.split(argv[1])[1].lstrip('tracker_anaout_').rstrip('.root'))
fileTag2 = int(os.path.split(argv[2])[1].lstrip('vertexana_').rstrip('.root'))

if fileTag1 != fileTag2:
    print "file tags don't match up... ( %i vs %i )this isn't critical but you may have mismatched files. Proceeding to explicit alignment checks..." %(fileTag1,fileTag2)
    
LenList = [TrkTree.GetEntries(),VtxTree.GetEntries(),shpTree.GetEntries(),shrTree.GetEntries()]

if LenList.count(LenList[0]) != len(LenList):
    print "Mismatch in # entries found in ana trees!!!"
    print "Vertices found per tree... \n",LenList
    quit(1)

RSE1   = []
RSE2   = []
vtxID2 = []
for ev in TrkTree:
    y = tuple((ev.run,ev.subrun,ev.event))
    RSE1.append(y)

for ev in VtxTree:
    y = tuple((ev.run,ev.subrun,ev.event))
    RSE2.append(y)
    vtxID2.append(ev.vtxid)
    
RSEV1 = []
for ev in TrkTree:
    y = tuple((ev.run,ev.subrun,ev.event,ev.vtx_id))
    RSEV1.append(y)

RSEV2 = []
for x in range(LenList[1]):
    if RSE2[x] == RSE2[x-1] and x != 0:
        rescaled+=1
        y = tuple((RSE2[x][0],RSE2[x][1],RSE2[x][2],rescaled))
        RSEV2.append(y)
    else:
        rescaled =0
        y = tuple((RSE2[x][0],RSE2[x][1],RSE2[x][2],rescaled))
        RSEV2.append(y)

in1not2 = sum(1 for ev in RSEV1 if ev not in RSEV2)
in2not1 = sum(1 for ev in RSEV2 if ev not in RSEV1)

if in1not2 != 0:
    print "Found %i (Run,Subrun,Event,VtxID) in track reco trees that don't exist in vertex reco trees"%(in1not2)
    quit(1)
    
if in2not1 != 0:
    print "Found %i (Run,Subrun,Event,VtxID) in vertex reco trees that don't exist in track reco trees"%(in2not1)
    quit(1)

for x in range(len(RSEV1)):

    if RSEV1[x] != RSEV2[x]:
        print "Trees are not aligned"
        quit(1)

print "Files seem to be 100% kosher, proceeding with analysis"
print ""
print "...Vic approves of kosher files, be like Vic! :D"
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
    
# --- Create output ROOT file and initialize variables ----------------- #
outFileName = 'FinalVertexVariables_%i.root'%(fileTag1)
outFileName = os.path.join(sys.argv[4],outFileName)
outFile = TFile(outFileName,'RECREATE')
outTree = TTree('NuMuVertexVariables','Final Vertex Variable Tree')

_run          = array('i',[0])
_subrun       = array('i',[0])
_event        = array('i',[0])
_vtxid        = array('i',[0])
_x            = array('f',[0])
_y            = array('f',[0])
_z            = array('f',[0])
_vtxalgo      = array('i',[0])
_infiducial   = array('i',[0])
_anyReco      = array('i',[0])
_ntracks      = array('i',[0])
_n5tracks     = array('i',[0])
_passCuts     = array('i',[0])
_cosmicLL     = array('f',[0])
_nubkgLL      = array('f',[0])
_good3DReco   = array('f',[0])
_eta          = array('f',[0])
_openAng      = array('f',[0])
_thetas       = array('f',[0])
_phis         = array('f',[0])
_phiT         = array('f',[0])
_alphaT       = array('f',[0])
_pT           = array('f',[0])
_bjX          = array('f',[0])
_sph          = array('f',[0])
_phi_v        = ROOT.vector('double')()
_theta_v      = ROOT.vector('double')()
_length_v     = ROOT.vector('double')()
_dqdx_v       = ROOT.vector('double')()
_EifP_v       = ROOT.vector('double')()
_EifMu_v      = ROOT.vector('double')()

_muon_id      = array('i',[-1])
_muon_phi     = array('f',[-1])
_muon_theta   = array('f',[-1])
_muon_length  = array('f',[-1])
_muon_dqdx    = array('f',[-1])
_muon_E       = array('f',[-1])

_proton_id    = array('i',[-1])
_proton_phi   = array('f',[-1])
_proton_theta = array('f',[-1])
_proton_length= array('f',[-1])
_proton_dqdx  = array('f',[-1])
_proton_E     = array('f',[-1])

def clear_vertex():
    _muon_id[0]          = int(-1)
    _muon_phi[0]         = float(-1)
    _muon_theta[0]       = float(-1)
    _muon_length[0]      = float(-1)
    _muon_dqdx[0]        = float(-1)
    _muon_E[0]           = float(-1)
    
    _proton_id[0]        = int(-1)
    _proton_phi[0]       = float(-1)
    _proton_theta[0]     = float(-1)
    _proton_length[0]    = float(-1)
    _proton_dqdx[0]      = float(-1)
    _proton_E[0]         = float(-1)
    

outTree.Branch('run'           , _run         , '_run/I'         ) 
outTree.Branch('subrun'        , _subrun      , '_subrun/I'      )
outTree.Branch('event'         , _event       , '_event/I'       )
outTree.Branch('vtxid'         , _vtxid       , '_vtxid/I'       )
outTree.Branch('Xreco'         , _x           , '_x/F'           )
outTree.Branch('Yreco'         , _y           , '_y/F'           )
outTree.Branch('Zreco'         , _z           , '_z/F'           )
outTree.Branch('VtxAlgo'       , _vtxalgo     , '_vtxalgo/I'     )
outTree.Branch('Good3DReco'    , _good3DReco  , '_good3DReco/I'  )
outTree.Branch('InFiducial'    , _infiducial  , '_infiducial/I'  )
outTree.Branch('AnythingRecod' , _anyReco     , '_anyReco/I'     )
outTree.Branch('NTracks'       , _ntracks     , '_ntracks/I'     )
outTree.Branch('N5cmTracks'    , _n5tracks    , '_n5tracks/I'    )
outTree.Branch('PassCuts'      , _passCuts    , '_passCuts/I'    )
outTree.Branch('CosmicLL'      , _cosmicLL    , '_cosmiLL/F'     )
outTree.Branch('NuBkgLL'       , _nubkgLL     , '_nubkgLL/F'     )
outTree.Branch('eta'           , _eta         , '_eta/F'         )
outTree.Branch('openAng'       , _openAng     , '_openAng/F'     )
outTree.Branch('thetas'        , _thetas      , '_thetas/F'      )
outTree.Branch('phis'          , _phis        , '_phis/F'        )
outTree.Branch('phiT'          , _phiT        , '_phiT/F'        )
outTree.Branch('alphaT'        , _alphaT      , '_alphaT/F'      )
outTree.Branch('pT'            , _pT          , '_pT/F'          )
outTree.Branch('bjX'           , _bjX         , '_bjX/F'         )
outTree.Branch('sph'           , _sph         , '_sph/F'         )
outTree.Branch('PhiReco_v'     , _phi_v         )
outTree.Branch('ThetaReco_v'   , _theta_v       )
outTree.Branch('TrackLength_v' , _length_v      )
outTree.Branch('dQdx_v'        , _dqdx_v        )
outTree.Branch('Edep_ifP_v'    , _EifP_v        )
outTree.Branch('Edep_ifMu_v'   , _EifMu_v       )

outTree.Branch('Muon_id'          , _muon_id         , 'Muon_id/I'          )
outTree.Branch('Muon_PhiReco'     , _muon_phi        , 'Muon_PhiReco/F'     )
outTree.Branch('Muon_ThetaReco'   , _muon_theta      , 'Muon_ThetaReco/F'   )
outTree.Branch('Muon_TrackLength' , _muon_length     , 'Muon_TrackLength/F' )
outTree.Branch('Muon_dQdx'        , _muon_dqdx       , 'Muon_dQdx/F'        )
outTree.Branch('Muon_Edep'        , _muon_E          , 'Muon_Edep/F'        )

outTree.Branch('Proton_id'          , _proton_id         , 'Proton_id/I'          )
outTree.Branch('Proton_PhiReco'     , _proton_phi        , 'Proton_PhiReco/F'     )
outTree.Branch('Proton_ThetaReco'   , _proton_theta      , 'Proton_ThetaReco/F'   )
outTree.Branch('Proton_TrackLength' , _proton_length     , 'Proton_TrackLength/F' )
outTree.Branch('Proton_dQdx'        , _proton_dqdx       , 'Proton_dQdx/F'        )
outTree.Branch('Proton_Edep'        , _proton_E          , 'Proton_Edep/F'        )

Vtx2DInfo = {}
for i,ev in enumerate(VtxTree):
    idx = RSEV2[i]

    foundClus = [i for i in ev.shower_frac_Y_v if i >=0]

    if len(foundClus) > 0:
        shrFrac = mean(foundClus)
    else:
        shrFrac = -1
        
    Vtx2DInfo[idx] = [ev.vertex_type,shrFrac,ev.scedr,ev.secondshower]

    
for ev in TrkTree:

    run            = ev.run
    subrun         = ev.subrun
    event          = ev.event
    vtxid          = ev.vtx_id
    IDvtx          = tuple((run,subrun,event,vtxid))
    vtxX           = ev.vtx_x
    vtxY           = ev.vtx_y
    vtxZ           = ev.vtx_z
    vtxPhi_v       = ev.vertexPhi
    vtxTheta_v     = ev.vertexTheta
    length_v       = ev.Length_v
    dqdx_v         = ev.Avg_Ion_v
    iondlen_v      = ev.IondivLength_v
    VertexType     = Vtx2DInfo[IDvtx][0] 
    NothingRecod   = ev.nothingReconstructed
    InFiducial     = VtxInFid(vtxX,vtxY,vtxZ)
    NumTracks      = len(length_v)
    Num5cmTracks   = sum(1 for x in length_v if x > 5)
    EifP_v         = ev.E_proton_v
    EifMu_v        = ev.E_muon_v
    Good3DReco     = ev.GoodVertex

    passCuts = True
    if NumTracks ==2 and Num5cmTracks ==2:
        mid            = np.argmin(ev.Avg_Ion_v)
        pid            = np.argmax(ev.Avg_Ion_v)
        mTh            = ev.vertexTheta[mid]
        pTh            = ev.vertexTheta[pid]
        mPh            = ev.vertexPhi[mid]
        pPh            = ev.vertexPhi[pid]
        mE             = ev.E_muon_v[mid]
        pE             = ev.E_proton_v[pid]
        EpCCQE         = ECCQE(pE,pTh,pid="proton")
        EmCCQE         = ECCQE(mE,mTh,pid="muon")
        Ecal           = ECal(mE,pE)
        dEp            = EpCCQE - Ecal
        dEm            = EmCCQE - Ecal
        dEmp           = EpCCQE-EmCCQE
        Q2CCQE         = Q2(EpCCQE,mE,mTh)
        Q2cal          = Q2(Ecal,mE,mTh)
        eta            = abs(ev.Avg_Ion_v[pid]-ev.Avg_Ion_v[mid])/(ev.Avg_Ion_v[pid]+ev.Avg_Ion_v[mid])
        openAng        = OpenAngle(pTh,mTh,pPh,mPh)
        thetas         = mTh+pTh
        phis           = PhiDiff(mPh,pPh)
        y              = pE/Ecal
        x              = Q2cal/(2*0.93956*(Ecal/1000. - mE/1000. - 0.105))
        sph            = sqrt(dEp**2+dEm**2+dEmp**2)
        shFrac         = Vtx2DInfo[IDvtx][1]
        secSh          = Vtx2DInfo[IDvtx][3]
        phiT           = GetPhiT(mE,pE,mTh,pTh,mPh,pPh)
        pT             = pTrans(mE,pE,mTh,pTh,mPh,pPh)
        alphT          = alphaT(mE,pE,mTh,pTh,mPh,pPh)
    
        if NothingRecod == 1:
            passCuts = False
        if InFiducial == False:
            passCuts = False
        if edgeCut(ev.closestWall):
            passCuts = False
        if sph > 1000:
            passCuts = False
        if pT > 500:
            passCuts = False
        if phiT > 3*pi/8:
            passCuts = False
        if y > 1:
            passCuts = False
        if abs(phis-pi) > 1.1:
            passCuts = False
        if secSh == 1:
            passCuts = False
        if shFrac > 0.5:
            passCuts = False
        if (not Good3DReco) and (abs(EpCCQE - EmCCQE)/EpCCQE > 0.3 or abs(EpCCQE-Ecal)/Ecal > 0.3):
            passCuts = False

    else:
        passCuts = False
    
        
    if passCuts == True:

        _muon_id[0]          = int(mid)
        _muon_phi[0]         = float(vtxPhi_v[mid])
        _muon_theta[0]       = float(vtxTheta_v[mid])
        _muon_length[0]      = float(length_v[mid])
        _muon_dqdx[0]        = float(dqdx_v[mid])
        _muon_E[0]           = float(EifMu_v.at(mid))
        
        _proton_id[0]          = int(pid)
        _proton_phi[0]         = float(vtxPhi_v[pid])
        _proton_theta[0]       = float(vtxTheta_v[pid])
        _proton_length[0]      = float(length_v[pid])
        _proton_dqdx[0]        = float(dqdx_v[pid])
        _proton_E[0]           = float(EifP_v.at(pid))

        feedValues = {"etaHist":eta,"openAngHist":openAng,"thetaHist":thetas,"phiTHist":phiT,"pTHist":pT,"xHist":x,"phiDiffHist":phis,"sphHist":sph,"alphaHist":alphT}
        
        cosmicLL = BeyesFactor(feedValues)
        nusepLL  = BeyesFactor(feedValues,'nuBack')
            
    _run[0]          = run
    _subrun[0]       = subrun
    _event[0]        = event
    _vtxid[0]        = vtxid
    _x[0]            = vtxX
    _y[0]            = vtxY
    _z[0]            = vtxZ
    _vtxalgo[0]      = VertexType
    _anyReco[0]      = not(NothingRecod)
    _infiducial[0]   = InFiducial
    _ntracks[0]      = NumTracks
    _n5tracks[0]     = Num5cmTracks
    _passCuts[0]     = passCuts
    _good3DReco[0]   = Good3DReco
    _cosmicLL[0]     = cosmicLL if passCuts else -9999
    _nubkgLL[0]      = nusepLL if passCuts else -9999
    _eta[0]          = eta if passCuts else -9999
    _openAng[0]      = openAng if passCuts else -9999
    _thetas[0]       = thetas if passCuts else -9999
    _phis[0]         = phis if passCuts else -9999
    _phiT[0]         = phiT if passCuts else -9999
    _alphaT[0]       = alphT if passCuts else -9999
    _pT[0]           = pT if passCuts else -9999
    _bjX[0]          = x if passCuts else -9999
    _sph[0]          = sph if passCuts else -9999

    
    _phi_v.clear()
    _theta_v.clear()
    _length_v.clear()
    _dqdx_v.clear()
    _EifP_v.clear()
    _EifMu_v.clear()
    for i in vtxPhi_v:      _phi_v.push_back(i)
    for i in vtxTheta_v:    _theta_v.push_back(i)
    for i in length_v:      _length_v.push_back(i) 
    for i in dqdx_v:        _dqdx_v.push_back(i) 
    for i in EifP_v:        _EifP_v.push_back(i)
    for i in EifMu_v:       _EifMu_v.push_back(i)
    
    outTree.Fill()
    clear_vertex()

outTree.Write()
outFile.Close()
sys.exit(0)       


