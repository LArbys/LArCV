import os, sys

if len(sys.argv) != 4:
    print 
    print "SUPERA = str(sys.argv[1])"
    print "NUM    = str(sys.argv[2])"
    print "OUTDIR = str(sys.argv[3])" 
    print 
    sys.exit(1)

import ROOT

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

SUPERA  = str(sys.argv[1])
NUM     = str(sys.argv[2])
OUTDIR  = str(sys.argv[3])

OUTFILE = "dead_wire_out_%s.root" % NUM

from larcv import larcv
proc = larcv.ProcessDriver(ROOT.std.string("ProcessDriver"))
cfg = ROOT.std.string(os.path.join(BASE_PATH,"determine.cfg"))
proc.configure(cfg)
flist_v = ROOT.std.vector("std::string")()
flist_v.push_back(ROOT.std.string(SUPERA))
proc.override_input_file(flist_v)
proc.override_output_file(ROOT.std.string(os.path.join(OUTDIR,OUTFILE)))
proc.override_ana_file(ROOT.std.string(os.path.join(OUTDIR,OUTFILE.replace("out","ana"))))
proc.initialize()
proc.batch_process()
proc.finalize()
sys.exit(0)
