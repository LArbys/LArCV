import os, sys

if len(sys.argv) != 5:
    print 
    print "CFG    = str(sys.argv[1])"
    print "FILE   = str(sys.argv[2])"
    print "NUM    = str(sys.argv[3])"
    print "OUTDIR = str(sys.argv[4])" 
    print 
    sys.exit(1)


BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

CFG     = str(sys.argv[1])
FILE    = str(sys.argv[2])
NUM     = str(sys.argv[3])
OUTDIR  = str(sys.argv[4])

from larcv import larcv
import ROOT
from ROOT import std

proc = larcv.ProcessDriver('ProcessDriver')
proc.configure(CFG)
flist = ROOT.std.vector('std::string')()
flist.push_back(ROOT.std.string(FILE))
proc.override_input_file(flist)
proc.override_ana_file(os.path.join(OUTDIR,"larbys_image_mc_%s.root" % NUM))
proc.initialize()
proc.batch_process()
proc.finalize()
