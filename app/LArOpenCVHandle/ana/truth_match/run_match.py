import os, sys

if len(sys.argv) != 5:
    print 
    print "CFG      = str(sys.argv[1])"
    print "SSFILE   = str(sys.argv[2])"
    print "PGRFILE  = str(sys.argv[3])"
    print "OUTDIR   = str(sys.argv[4])" 
    print 
    sys.exit(1)


BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

CFG     = str(sys.argv[1])
SSFILE  = str(sys.argv[2])
PGRFILE = str(sys.argv[3])
OUTDIR  = str(sys.argv[4])

num = int(os.path.basename(PGRFILE).split(".")[0].split("_")[-1])

from larcv import larcv
import ROOT
from ROOT import std

proc = larcv.ProcessDriver('ProcessDriver')
proc.configure(CFG)
flist = ROOT.std.vector('std::string')()
flist.push_back(ROOT.std.string(SSFILE))
flist.push_back(ROOT.std.string(PGRFILE))
proc.override_input_file(flist)
proc.override_ana_file(os.path.join(OUTDIR,"pgraph_truth_match_%d.root" % num))
proc.initialize()
proc.batch_process()
proc.finalize()
