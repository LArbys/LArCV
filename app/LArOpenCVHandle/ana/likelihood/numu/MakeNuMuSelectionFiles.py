import ROOT
from ROOT import TFile,TTree
from larlite import larutil
import matplotlib.pyplot as plt
import pickle
from math import sin,cos,sqrt,log,pi,acos
from sys import argv
from array import array
import os

## When you call this program it must have the following format... ##
## python MakeNuMuSelectionFiles.py INPUT1 INPUT2 INPUT3 INPUT4    ##
## with ...                                                        ##
##    INPUT1 = 3D Track Reco ROOT file                             ##
##    INPUT2 = Vertex reconstruction ana ROOT file                 ##
##    INPUT3 = pickle containing numu cosmic LL histograms         ##
##    INPUT4 = pickle containing nu background LL histograms       ##
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

#    sceOffsets = sce.GetPosOffsets(vtxX,vtxY,vtxZ)
    sceOffsets = [0,0,0]
    
    xmin =  0      + edgeCut 
    xmax =  256.25 - edgeCut
    ymin = -116.5  + edgeCut
    ymax =  116.5  - edgeCut
    zmin =  0      + edgeCut
    zmax =  1036.8 - edgeCut

    sceX = vtxX + sceOffsets[0] - 0.7
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

fileTag1 = int(os.path.split(argv[1])[1].lstrip('reco3d_ana_').rstrip('.root'))
fileTag2 = int(os.path.split(argv[2])[1].lstrip('vertexana_larcv_').rstrip('.root'))

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
    if RSE2[x] == RSE2[x-1]:
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

# --- Create output ROOT file and initialize variables ----------------- #
outFileName = 'FinalVertexVariables_%i.root'%(fileTag1)
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
_good3DReco = array('i',[0])
_phi_v      = ROOT.vector('double')()
_theta_v    = ROOT.vector('double')()
_length_v   = ROOT.vector('double')()
_dqdx_v     = ROOT.vector('double')()
_iondlen_v  = ROOT.vector('double')()
_EifP_v     = ROOT.vector('double')()
_EifMu_v    = ROOT.vector('double')()

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
outTree.Branch('PhiReco_v'     , _phi_v       )
outTree.Branch('ThetaReco_v'   , _theta_v     )
outTree.Branch('TrackLength_v' , _length_v    )
outTree.Branch('dQdx_v'        , _dqdx_v      )
outTree.Branch('IonPerLen_v'   , _iondlen_v   )
outTree.Branch('Edep_ifP_v'    , _EifP_v      )
outTree.Branch('Edep_ifMu_v'   , _EifMu_v     )

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
    vtxX           = ev.RecoVertex.X()
    vtxY           = ev.RecoVertex.Y()
    vtxZ           = ev.RecoVertex.Z()
    vtxPhi_v       = ev.vertexPhi
    vtxTheta_v     = ev.vertexTheta
    length_v       = ev.Length_v
    dqdx_v         = ev.Avg_Ion_v
    iondlen_v      = ev._IondivLength_v
    VertexType     = Vtx2DInfo[IDvtx][0] 
    NothingRecod   = ev.nothingReconstructed
    InFiducial     = VtxInFid(vtxX,vtxY,vtxZ)
    NumTracks      = len(length_v)
    Num5cmTracks   = ev.NtracksReco
    EifP_v         = ev.E_proton_v
    EifMu_v        = ev.E_muon_v
    PassAllChecks  = ev.GoodVertex
    
    passCuts = True
    if VertexType != 3:
        passCuts = False
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

        processVars = [openAng,wallDist,eta,ionplen,[theta0,theta1],[phi0,phi1],shfrac]
        skipVars    = [3,6] #Will skip these variable indices when calculating LL, 3,6 currently ignored due to MC/data diffs
        cosmicLL = 0
        nusepLL  = 0
        for i,y in enumerate(processVars):
            if i in skipVars: continue
            if isinstance(y,list):
                cosmicLL+= ComputeVarProb2D(LLPdfs[i],y[0],y[1])
                nusepLL += ComputeVarProb2D(LLPdfs_nusep[i],y[0],y[1])
            else:
                cosmicLL+= ComputeVarProb(LLPdfs[i],y)
                nusepLL += ComputeVarProb(LLPdfs_nusep[i],y)

    
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
    _good3DReco[0] = PassAllChecks
    
    _phi_v.clear()
    _theta_v.clear()
    _length_v.clear()
    _dqdx_v.clear()
    _EifP_v.clear()
    _EifMu_v.clear()
    for i in vtxPhi_v:    _phi_v.push_back(i)
    for i in vtxTheta_v:  _theta_v.push_back(i)
    for i in length_v:    _length_v.push_back(i) 
    for i in dqdx_v:      _dqdx_v.push_back(i) 
    for i in iondlen_v:   _iondlen_v.push_back(i)
    for i in EifP_v:      _EifP_v.push_back(i)
    for i in EifMu_v:     _EifMu_v.push_back(i)
    
    outTree.Fill()

outTree.Write()
outFile.Close()
    
