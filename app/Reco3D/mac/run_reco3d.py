import os,sys

if len(sys.argv) != 6:
    print 
    print "CONFIG_FILE = str(sys.argv[1])"
    print "IMG_FILE    = str(sys.argv[2])"
    print "TAGGER_FILE = str(sys.argv[3])"
    print "PGRAPH_FILE = str(sys.argv[4])"
    print "OUTPUT_DIR  = str(sys.argv[5])"
    print
    sys.exit(1)

import ROOT, sys
from ROOT import std
from larcv import larcv

ROOT.gROOT.SetBatch(True)

CONFIG_FILE = str(sys.argv[1])
IMG_FILE    = str(sys.argv[2])
TAGGER_FILE = str(sys.argv[3])
PGRAPH_FILE = str(sys.argv[3])
OUTPUT_DIR  = str(sys.argv[4])

num = int(os.path.basename(PGRAPH_FILE).split(".")[0].split("_")[-1])

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

proc = larcv.ProcessDriver('ProcessDriver')

proc.configure(CONFIG_FILE)
flist=ROOT.std.vector('std::string')()
flist.push_back(ROOT.std.string(IMG_FILE))
flist.push_back(ROOT.std.string(TAGGER_FILE))
flist.push_back(ROOT.std.string(PGRAPH_FILE))
proc.override_input_file(flist)

proc.override_ana_file(ROOT.std.string(os.path.join(OUTPUT_DIR,"tracker_anaout_%d.root" % num)))

alg_id = proc.process_id("Run3DTracker")
alg    = proc.process_ptr(alg_id)
print "GOT: ",alg,"@ id=",alg_id

SPLINE_PATH = os.path.join(BASE_PATH,"..","Proton_Muon_Range_dEdx_LAr_TSplines.root")
alg.SetSplineLocation(SPLINE_PATH)
alg.SetOutDir(OUTPUT_DIR)
alg.SetLLOutName(ROOT.std.string(os.path.join(OUTPUT_DIR,"tracker_reco_%d.root" % num)))

proc.initialize()
proc.batch_process()
proc.finalize()
