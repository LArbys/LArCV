import os,sys

print len(sys.argv)
print
print
if len(sys.argv) < 4:
    print
    print "IMG_FILE        = str(sys.argv[1])"
    print "LARLITE_IN_FILE = str(sys.argv[2])"
    print "OUT_VTX         = str(sys.argv[3])"
    print "OUT_IMAGE2D     = str(sys.argv[4])"
    print
    sys.exit(1)

import ROOT, sys
from ROOT import std
from larcv import larcv

ROOT.gROOT.SetBatch(True)

CONFIG_FILE     = "cfg/chimera.cfg" #str(sys.argv[1])
IMG_FILE        = str(sys.argv[1])
LARLITE_IN_FILE = str(sys.argv[2])
#OUTPUT_DIR      = str(sys.argv[3])
OUT_VTX         = str(sys.argv[3])
OUT_IMAGE2D     = str(sys.argv[4])
OUTPUT_DIR     = str(sys.argv[5])
#MY_EVENT        = int(sys.argv[6])
#MY_VTXID        = int(sys.argv[7])
#MY_TRACK        = int(sys.argv[8])
EVENT_FILE      = ""

print "This is the output dir: ", OUTPUT_DIR

#num = int(os.path.basename(IMG_FILE).split(".")[0].split("_")[-1])

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

proc = larcv.ProcessDriver('ProcessDriver')

proc.configure(CONFIG_FILE)
flist=ROOT.std.vector('std::string')()
flist.push_back(ROOT.std.string(IMG_FILE))
proc.override_input_file(flist)

alg_id = proc.process_id("Chimera")
alg    = proc.process_ptr(alg_id)
print "GOT: ",alg,"@ id=",alg_id

SPLINE_PATH = os.path.join(BASE_PATH,"..","Proton_Muon_Range_dEdx_LAr_TSplines.root")
alg.SetSplineLocation(SPLINE_PATH)

print "PRINDING",LARLITE_IN_FILE

alg.SetOutDir(ROOT.std.string(OUTPUT_DIR))
alg.SetLLInName(ROOT.std.string(LARLITE_IN_FILE))
alg.SetOutVtx(ROOT.std.string(OUT_VTX))
alg.SetOutImage2d(ROOT.std.string(OUT_IMAGE2D))
#alg.SetMySubrun(MY_SUBRUN)
#alg.SetMyEvent(MY_EVENT)
#alg.SetMyVtxid(MY_VTXID)
#alg.SetMyTrack(MY_TRACK)
#alg.SetRootAnaFile(ROOT.std.string(ROOT_ANAFILE))

proc.initialize()
#proc.batch_process(0,1)
proc.batch_process()
proc.finalize()
