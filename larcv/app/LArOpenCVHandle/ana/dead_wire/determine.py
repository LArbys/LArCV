import os, sys

if len(sys.argv) != 3:
    print 
    print "SSFILE  = str(sys.argv[1])"
    print "OUTDIR  = str(sys.argv[2])" 
    print 
    sys.exit(1)

import ROOT

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

SSFILE  = str(sys.argv[1])
OUTDIR  = str(sys.argv[2])
num = int(os.path.basename(SSFILE).split(".")[0].split("_")[-1])

OUTFILE = "dead_wire_out_%d.root" % num

from larcv import larcv
proc = larcv.ProcessDriver(ROOT.std.string("ProcessDriver"))
cfg = ROOT.std.string(os.path.join(BASE_PATH,"determine.cfg"))
print cfg
proc.configure(cfg)
flist_v = ROOT.std.vector("std::string")()
flist_v.push_back(ROOT.std.string(SSFILE))
proc.override_input_file(flist_v)
proc.override_output_file(ROOT.std.string(os.path.join(OUTDIR,OUTFILE)))
proc.override_ana_file(ROOT.std.string(os.path.join(OUTDIR,OUTFILE.replace("out","ana"))))
proc.initialize()
proc.batch_process()
proc.finalize()
sys.exit(0)
