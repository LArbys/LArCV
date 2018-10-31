import os, sys

if len(sys.argv) != 7:
    print 
    print "SUPERA = str(sys.argv[1])"
    print "TAGGER = str(sys.argv[2])"
    print "SSNET  = str(sys.argv[3])"
    print "SERVER = str(sys.argv[4])"
    print "NUM    = str(sys.argv[5])"
    print "OUTDIR = str(sys.argv[6])"
    print 
    sys.exit(1)

import ROOT

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

SUPERA  = str(sys.argv[1])
TAGGER  = str(sys.argv[2])
SSNET   = str(sys.argv[3])
SERVER  = int(str(sys.argv[4]))
NUM     = str(sys.argv[5])
OUTDIR  = str(sys.argv[6])
OUTFILE = "check_ssnet_out_%s.root" % NUM

from larcv import larcv
proc = larcv.ProcessDriver(ROOT.std.string("ProcessDriver"))
cfg = ROOT.std.string(os.path.join(BASE_PATH,"check_ssnet_config_server.cfg"))
proc.configure(cfg)
flist_v = ROOT.std.vector("std::string")()
flist_v.push_back(ROOT.std.string(SUPERA))
if SERVER==1:
    flist_v.push_back(ROOT.std.string(TAGGER))
    flist_v.push_back(ROOT.std.string(SSNET))
proc.override_input_file(flist_v)
proc.override_output_file(ROOT.std.string(os.path.join(OUTDIR,OUTFILE)))
proc.override_ana_file(ROOT.std.string(os.path.join(OUTDIR,OUTFILE.replace("out","ana"))))
proc.initialize()

pid = proc.process_id(ROOT.std.string("SSNetChecker"))
ssnet_checker = proc.process_ptr(pid)
print "@pid=",pid,"ptr=",ssnet_checker
ssnet_checker.SetFileName(ROOT.std.string(os.path.basename(SUPERA)))

proc.batch_process(0,20)

proc.finalize()
sys.exit(0)
