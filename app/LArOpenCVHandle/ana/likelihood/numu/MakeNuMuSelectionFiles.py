import ROOT
from ROOT import TFile,TTree
from larlite import larutil
import numpy as np
import matplotlib.pyplot as plt
import pickle
from math import sin,cos,sqrt,log,pi,acos
from sys import argv
from array import array
import os,sys

## When you call this program it must have the following format... ##
## python MakeNuMuSelectionFiles.py INPUT1 INPUT2....              ##
## with ...                                                        ##
##    INPUT1 = 3D Track Reco ROOT file                             ##
##    INPUT2 = Vertex reconstruction ana ROOT file                 ##
##    INPUT3 = pickle containing numu cosmic LL histograms         ##
##    INPUT4 = pickle containing nu background LL histograms       ##
##    INPUT5 = Output directory                                    ##
## --------------------------------------------------------------- ##

#sce = larutil.SpaceChargeMicroBooNE()

# --- Some functions for internal analysis use -------------------- #
def Compute3DAngle(theta0,theta1,phi0,phi1):

    return acos(cos(theta0)*cos(theta1)+sin(theta0)*sin(theta1)*cos(phi0-phi1))

def ComputeEta(dqdx0,dqdx1):

    return abs(dqdx0-dqdx1)/(dqdx0+dqdx1)
    
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

def ComputeVarProb2D(LLpdf,value1,value2):

    foundBin = False
    spacing1 = LLpdf[0][1] - LLpdf[0][0]
    spacing2 = LLpdf[1][1] - LLpdf[1][0]
    
    for i,x in enumerate(LLpdf[0]):
        for j,y in enumerate(LLpdf[1]):

            if value1 >= x and (value1-x) < spacing1 and value2 >= y and (value2-y) < spacing2:
                valBins  = [i,j]
                foundBin = True
                break

    if foundBin == True:
        if LLpdf[2][valBins[0]][valBins[1]] != 0:
            return log(LLpdf[2][valBins[0]][valBins[1]])
        else:
            return -1
    else:
        return 0
                
def VtxInFid(vtxX,vtxY,vtxZ,edgeCut=10):

    #sceOffsets = sce.GetPosOffsets(vtxX,vtxY,vtxZ)
    sceOffsets = [0,0,0]
    
    xmin =  0      + edgeCut 
    xmax =  256.25 - edgeCut
    ymin = -116.5  + edgeCut
    ymax =  116.5  - edgeCut
    zmin =  0      + edgeCut
    zmax =  1036.8 - edgeCut

#    sceX = vtxX + sceOffsets[0] - 0.7
    sceX = vtxX + sceOffsets[0]
    sceY = vtxY - sceOffsets[1]
    sceZ = vtxZ - sceOffsets[2]
    
    if sceX > xmax or sceX < xmin or sceY > ymax or sceY < ymin or sceZ > zmax or sceZ < zmin:
        return False
    else:
        return True

    
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
LEETree  = VtxRecoFile.Get("LEE1e1pTree")
angTree  = VtxRecoFile.Get("AngleAnalysis")
gapTree  = VtxRecoFile.Get("GapAnalysis")
shpTree  = VtxRecoFile.Get("ShapeAnalysis")
matTree  = VtxRecoFile.Get("MatchAnalysis")
dqdxTree = VtxRecoFile.Get("dQdSAnalysis")

VtxTree.AddFriend(LEETree)
VtxTree.AddFriend(angTree)
VtxTree.AddFriend(gapTree)
VtxTree.AddFriend(shpTree)
VtxTree.AddFriend(matTree)
VtxTree.AddFriend(dqdxTree)
# ---------------------------------------------------------------------- #

# --- Perform some alignment checks to be sure the ana files match up -- #

fileTag1 = int(os.path.split(argv[1])[1].lstrip('tracker_anaout_').rstrip('.root'))
fileTag2 = int(os.path.split(argv[2])[1].lstrip('vertexana_').rstrip('.root'))

if fileTag1 != fileTag2:
    print "file tags don't match up... ( %i vs %i )this isn't critical but you may have mismatched files. Proceeding to explicit alignment checks..." %(fileTag1,fileTag2)
    
LenList = [TrkTree.GetEntries(),VtxTree.GetEntries(),LEETree.GetEntries(),angTree.GetEntries(),shpTree.GetEntries(),matTree.GetEntries(),dqdxTree.GetEntries()]

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
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

with open(argv[3],'rb') as handle: LLPdfs = pickle.load(handle)          # Load LL histograms for 1mu1p cosmic differentiation
with open(argv[4],'rb') as handle: LLPdfs_nusep = pickle.load(handle)    # Load LL histograms for 1mu1p nu background differentiation
with open(argv[5],'rb') as handle: LLPdfs_findpi = pickle.load(handle)   # Load LL histograms for 1mu1p pi0 background differentiation
    
# --- Create output ROOT file and initialize variables ----------------- #
outFileName = 'FinalVertexVariables_%i.root'%(fileTag1)
outFileName = os.path.join(sys.argv[6],outFileName)
outFile = TFile(outFileName,'RECREATE')
outTree = TTree('NuMuVertexVariables','Final Vertex Variable Tree')

_run        = array('i',[0])
_subrun     = array('i',[0])
_event      = array('i',[0])
_vtxid      = array('i',[0])
_x          = array('f',[0])
_y          = array('f',[0])
_z          = array('f',[0])
_vtxalgo    = array('i',[0])
_infiducial = array('i',[0])
_anyReco    = array('i',[0])
_ntracks    = array('i',[0])
_n5tracks   = array('i',[0])
_passCuts   = array('i',[0])
_cosmicLL   = array('f',[0])
_nubkgLL    = array('f',[0])
_ccpi0LL    = array('f',[0])
_good3DReco = array('i',[0])
_eta        = array('f',[0])
_wallDist   = array('f',[0])
_openAng    = array('f',[0])
_phi_v         = ROOT.vector('double')()
_theta_v       = ROOT.vector('double')()
_length_v      = ROOT.vector('double')()
_dqdx_v        = ROOT.vector('double')()
_trunc_dqdx1_v = ROOT.vector('double')()
_trunc_dqdx3_v = ROOT.vector('double')()
_iondlen_v     = ROOT.vector('double')()
_EifP_v        = ROOT.vector('double')()
_EifMu_v       = ROOT.vector('double')()

_muon_id          = array('i',[-1])
_muon_phi         = array('f',[-1])
_muon_theta       = array('f',[-1])
_muon_length      = array('f',[-1])
_muon_dqdx        = array('f',[-1])
_muon_trunc_dqdx1 = array('f',[-1])
_muon_trunc_dqdx3 = array('f',[-1])
_muon_iondlen     = array('f',[-1])
_muon_E           = array('f',[-1])

_proton_id          = array('i',[-1])
_proton_phi         = array('f',[-1])
_proton_theta       = array('f',[-1])
_proton_length      = array('f',[-1])
_proton_dqdx        = array('f',[-1])
_proton_trunc_dqdx1 = array('f',[-1])
_proton_trunc_dqdx3 = array('f',[-1])
_proton_iondlen     = array('f',[-1])
_proton_E           = array('f',[-1])

def clear_vertex():
    _muon_id[0]          = int(-1)
    _muon_phi[0]         = float(-1)
    _muon_theta[0]       = float(-1)
    _muon_length[0]      = float(-1)
    _muon_dqdx[0]        = float(-1)
    _muon_trunc_dqdx1[0] = float(-1)
    _muon_trunc_dqdx3[0] = float(-1)
    _muon_iondlen[0]     = float(-1)
    _muon_E[0]           = float(-1)
    
    _proton_id[0]          = int(-1)
    _proton_phi[0]         = float(-1)
    _proton_theta[0]       = float(-1)
    _proton_length[0]      = float(-1)
    _proton_dqdx[0]        = float(-1)
    _proton_trunc_dqdx1[0] = float(-1)
    _proton_trunc_dqdx3[0] = float(-1)
    _proton_iondlen[0]     = float(-1)
    _proton_E[0]           = float(-1)
    

outTree.Branch('run'           , _run         , '_run/I'        ) 
outTree.Branch('subrun'        , _subrun      , '_subrun/I'     )
outTree.Branch('event'         , _event       , '_event/I'      )
outTree.Branch('vtxid'         , _vtxid       , '_vtxid/I'      )
outTree.Branch('Xreco'         , _x           , '_x/F'          )
outTree.Branch('Yreco'         , _y           , '_y/F'          )
outTree.Branch('Zreco'         , _z           , '_z/F'          )
outTree.Branch('VtxAlgo'       , _vtxalgo     , '_vtxalgo/I'    )
outTree.Branch('Good3DReco'    , _good3DReco  , '_good3DReco/I' )
outTree.Branch('InFiducial'    , _infiducial  , '_infiducial/I' )
outTree.Branch('AnythingRecod' , _anyReco     , '_anyReco/I'    )
outTree.Branch('NTracks'       , _ntracks     , '_ntracks/I'    )
outTree.Branch('N5cmTracks'    , _n5tracks    , '_n5tracks/I'   )
outTree.Branch('PassCuts'      , _passCuts    , '_passCuts/I'   )
outTree.Branch('CosmicLL'      , _cosmicLL    , '_cosmiLL/F'    )
outTree.Branch('NuBkgLL'       , _nubkgLL     , '_nubkgLL/F'    )
outTree.Branch('CCpi0LL'       , _ccpi0LL     , '_ccpi0LL/F'    )
outTree.Branch('WallDist'      , _wallDist    , '_wallDist/F'   )
outTree.Branch('eta'           , _eta         , '_eta/F'        )
outTree.Branch('openAng'       , _openAng     , '_openAng/F'    )
outTree.Branch('PhiReco_v'     , _phi_v         )
outTree.Branch('ThetaReco_v'   , _theta_v       )
outTree.Branch('TrackLength_v' , _length_v      )
outTree.Branch('dQdx_v'        , _dqdx_v        )
outTree.Branch('Trunc_dQdx1_v' , _trunc_dqdx1_v )
outTree.Branch('Trunc_dQdx3_v' , _trunc_dqdx3_v ) 
outTree.Branch('IonPerLen_v'   , _iondlen_v     )
outTree.Branch('Edep_ifP_v'    , _EifP_v        )
outTree.Branch('Edep_ifMu_v'   , _EifMu_v       )

outTree.Branch('Muon_id'          , _muon_id         , 'Muon_id/I'          )
outTree.Branch('Muon_PhiReco'     , _muon_phi        , 'Muon_PhiReco/F'     )
outTree.Branch('Muon_ThetaReco'   , _muon_theta      , 'Muon_ThetaReco/F'   )
outTree.Branch('Muon_TrackLength' , _muon_length     , 'Muon_TrackLength/F' )
outTree.Branch('Muon_dQdx'        , _muon_dqdx       , 'Muon_dQdx/F'        )
outTree.Branch('Muon_Trunc_dQdx1' , _muon_trunc_dqdx1, 'Muon_Trunc_dQdx1/F' )
outTree.Branch('Muon_Trunc_dQdx3' , _muon_trunc_dqdx3, 'Muon_Trunc_dQdx3/F' )
outTree.Branch('Muon_IonPerLen'   , _muon_iondlen    , 'Muon_IonPerLen/F'   )
outTree.Branch('Muon_Edep'        , _muon_E          , 'Muon_Edep/F'        )

outTree.Branch('Proton_id'          , _proton_id         , 'Proton_id/I'          )
outTree.Branch('Proton_PhiReco'     , _proton_phi        , 'Proton_PhiReco/F'     )
outTree.Branch('Proton_ThetaReco'   , _proton_theta      , 'Proton_ThetaReco/F'   )
outTree.Branch('Proton_TrackLength' , _proton_length     , 'Proton_TrackLength/F' )
outTree.Branch('Proton_dQdx'        , _proton_dqdx       , 'Proton_dQdx/F'        )
outTree.Branch('Proton_Trunc_dQdx1' , _proton_trunc_dqdx1, 'Proton_Trunc_dQdx1/F' )
outTree.Branch('Proton_Trunc_dQdx3' , _proton_trunc_dqdx3, 'Proton_Trunc_dQdx3/F' )
outTree.Branch('Proton_IonPerLen'   , _proton_iondlen    , 'Proton_IonPerLen/F'   )
outTree.Branch('Proton_Edep'        , _proton_E          , 'Proton_Edep/F'        )

Vtx2DInfo = {}
for i,ev in enumerate(VtxTree):
    idx = RSEV2[i]
    if ev.npar ==2:
        Vtx2DInfo[idx] = [ev.vertex_type,[ev.shower_frac_v[0],ev.shower_frac_v[1]],ev.scedr]
    else:
        Vtx2DInfo[idx] = [ev.vertex_type,-1,ev.scedr]
        
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
    trunc_dqdx1_v  = ev.Truncated_dQdX1_v
    trunc_dqdx3_v  = ev.Truncated_dQdX3_v
    iondlen_v      = ev.IondivLength_v
    VertexType     = Vtx2DInfo[IDvtx][0] 
    NothingRecod   = ev.nothingReconstructed
    InFiducial     = VtxInFid(vtxX,vtxY,vtxZ)
    NumTracks      = len(length_v)
    #Num5cmTracks   = ev.NtracksReco
    Num5cmTracks   = sum(1 for x in length_v if x > 5)
    EifP_v         = ev.E_proton_v
    EifMu_v        = ev.E_muon_v
    PassAllChecks  = ev.GoodVertex
    
    passCuts = True
    if NothingRecod == 1:
        passCuts = False 
    if InFiducial == False:
        passCuts = False
    if NumTracks !=2 or Num5cmTracks !=2:
        passCuts = False 

    if passCuts == True:
        theta0    = vtxTheta_v[0]
        theta1    = vtxTheta_v[1]
        phi0      = vtxPhi_v[0]
        phi1      = vtxPhi_v[1]
        ion0      = dqdx_v[0]
        ion1      = dqdx_v[1]
        iondlen0  = iondlen_v[0]
        iondlen1  = iondlen_v[1]    
        openAng   = Compute3DAngle(theta0,theta1,phi0,phi1)
        dist0     = ev.closestWall[0]
        dist1     = ev.closestWall[1] 
        shfrac    = (Vtx2DInfo[IDvtx][1][0]+Vtx2DInfo[IDvtx][1][1]) if isinstance(Vtx2DInfo[IDvtx][1],list) else -1
        wallDist = min([dist0,dist1])
        eta      = ComputeEta(ion0,ion1)
        ionplen  = ComputeEta(iondlen0,iondlen1)
        
        #
        # muon and proton selection
        #
        mid = np.argmax(dqdx_v)
        pid = np.argmin(dqdx_v)
        
        _muon_id[0]          = int(mid)
        _muon_phi[0]         = float(vtxPhi_v[mid])
        _muon_theta[0]       = float(vtxTheta_v[mid])
        _muon_length[0]      = float(length_v[mid])
        _muon_dqdx[0]        = float(dqdx_v[mid])
        _muon_trunc_dqdx1[0] = float(trunc_dqdx1_v.at(mid))
        _muon_trunc_dqdx3[0] = float(trunc_dqdx3_v.at(mid))
        _muon_iondlen[0]     = float(iondlen_v[mid])
        _muon_E[0]           = float(EifMu_v.at(mid))
        
        _proton_id[0]          = int(pid)
        _proton_phi[0]         = float(vtxPhi_v[pid])
        _proton_theta[0]       = float(vtxTheta_v[pid])
        _proton_length[0]      = float(length_v[pid])
        _proton_dqdx[0]        = float(dqdx_v[pid])
        _proton_trunc_dqdx1[0] = float(trunc_dqdx1_v.at(pid))
        _proton_trunc_dqdx3[0] = float(trunc_dqdx3_v.at(pid))
        _proton_iondlen[0]     = float(iondlen_v[pid])
        _proton_E[0]           = float(EifP_v.at(pid))
        #

        processDict = {'openAng':openAng,'wallDist':wallDist,'eta':eta,'ionplen':ionplen,'theta':[theta0,theta1],'phi':[phi0,phi1]}
        skipVars    = ['ionplen'] #Will skip these variables when calculating LL, ignored due to data/MC diffs                                                                
        cosmicLL = 0
        nusepLL  = 0
        ccpi0LL  = 0
        for y in processDict.keys():
           if y in skipVars: continue
           processVal = processDict[y]
           if isinstance(processVal,list):
               cosmicLL += ComputeVarProb2D(LLPdfs[y],processVal[0],processVal[1])
               nusepLL  += ComputeVarProb2D(LLPdfs_nusep[y],processVal[0],processVal[1])
               ccpi0LL  += ComputeVarProb2D(LLPdfs_findpi[y],processVal[0],processVal[1])
           else:
               cosmicLL += ComputeVarProb(LLPdfs[y],processVal)
               nusepLL  += ComputeVarProb(LLPdfs_nusep[y],processVal) 
               ccpi0LL  += ComputeVarProb(LLPdfs_findpi[y],processVal)
            
    _run[0]        = run
    _subrun[0]     = subrun
    _event[0]      = event
    _vtxid[0]      = vtxid
    _x[0]          = vtxX
    _y[0]          = vtxY
    _z[0]          = vtxZ
    _vtxalgo[0]    = VertexType
    _anyReco[0]    = not(NothingRecod)
    _infiducial[0] = InFiducial
    _ntracks[0]    = NumTracks
    _n5tracks[0]   = Num5cmTracks
    _passCuts[0]   = passCuts
    _cosmicLL[0]   = cosmicLL if passCuts == True else -9999
    _nubkgLL[0]    = nusepLL if passCuts == True else -9999
    _ccpi0LL[0]    = ccpi0LL if passCuts == True else -9999
    _eta[0]        = eta if passCuts == True else -9999
    _wallDist[0]   = wallDist if passCuts == True else -9999
    _openAng[0]    = openAng if passCuts == True else -9999
    _good3DReco[0] = PassAllChecks
    
    _phi_v.clear()
    _theta_v.clear()
    _length_v.clear()
    _dqdx_v.clear()
    _trunc_dqdx1_v.clear()
    _trunc_dqdx3_v.clear()
    _EifP_v.clear()
    _EifMu_v.clear()
    for i in vtxPhi_v:      _phi_v.push_back(i)
    for i in vtxTheta_v:    _theta_v.push_back(i)
    for i in length_v:      _length_v.push_back(i) 
    for i in dqdx_v:        _dqdx_v.push_back(i) 
    for i in trunc_dqdx1_v: _trunc_dqdx1_v.push_back(i)
    for i in trunc_dqdx3_v: _trunc_dqdx3_v.push_back(i)
    for i in iondlen_v:     _iondlen_v.push_back(i)
    for i in EifP_v:        _EifP_v.push_back(i)
    for i in EifMu_v:       _EifMu_v.push_back(i)
    
    outTree.Fill()
    clear_vertex()

outTree.Write()
outFile.Close()
    
