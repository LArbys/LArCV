import os,sys

if len(sys.argv) != 4:
    print 
    print "IMG_FILE    = str(sys.argv[1])"
    print "PGRAPH_FILE = str(sys.argv[2])"
    print "OUTPUT_DIR  = str(sys.argv[3])"
    print
    sys.exit(1)

import ROOT, sys
from ROOT import std
from larcv import larcv

IMG_FILE    = str(sys.argv[1])
PGRAPH_FILE = str(sys.argv[2])
OUTPUT_DIR  = str(sys.argv[3])

num = int(os.path.basename(PGRAPH_FILE).split(".")[0].split("_")[-1])

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

proc = larcv.ProcessDriver('ProcessDriver')

proc.configure(os.path.join(BASE_PATH,"cfg","read_nue.cfg"))
flist=ROOT.std.vector('std::string')()
flist.push_back(ROOT.std.string(IMG_FILE))
flist.push_back(ROOT.std.string(PGRAPH_FILE))
proc.override_input_file(flist)

alg_id = proc.process_id("ReadNueFile")
alg    = proc.process_ptr(alg_id)
print "GOT: ",alg,"@ id=",alg_id

SPLINE_PATH = os.path.join(BASE_PATH,"Proton_Muon_Range_dEdx_LAr_TSplines.root")
alg.SetSplineLocation(SPLINE_PATH)

proc.initialize()
proc.batch_process()
proc.finalize()
