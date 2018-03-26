import sys, os

if len(sys.argv) != 3:
    print
    print "INFILE = str(sys.argv[1])"
    print "IN_CSV = str(sys.argv[2])"
    print
    sys.exit(1)

import tempfile

fd, path = tempfile.mkstemp()
tmp = os.fdopen(fd, 'w')

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

INFILE = str(sys.argv[1])
IN_CSV = str(sys.argv[2])

OUTFILE = INFILE.split(".")[0]
OUTFILE += "_filter.root"

BASE_CONFIG = os.path.join(BASE_PATH,"cfg","rse_config_base.cfg")

data = ""
with open(BASE_CONFIG,'r') as f:
    data = f.read()

data = data.replace("AAA",IN_CSV)
tmp.write(data)
tmp.close()

print "INFILE=%s"%INFILE
print "IN_CSV=%s"%IN_CSV
print "OUTFILE=%s"%OUTFILE
print "BASE_CONFIG=%s"%BASE_CONFIG
print "CONFIG=%s"%path

import ROOT
from larcv import larcv
proc = larcv.ProcessDriver('ProcessDriver')

proc.configure(ROOT.std.string(path))
proc.override_input_file(ROOT.std.vector("std::string")(1,ROOT.std.string(INFILE)))
proc.override_output_file(ROOT.std.string(OUTFILE))
proc.initialize()
proc.batch_process()
proc.finalize()

os.remove(path)
sys.exit(0)
