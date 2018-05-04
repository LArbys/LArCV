import os,sys, tempfile

if len(sys.argv) != 10:
    print
    print "PKL        = str(sys.argv[1])"
    print "RUN        = str(sys.argv[2])"
    print "SUBRUN     = str(sys.argv[3])"
    print "EVENT      = str(sys.argv[4])"
    print "STAGE1_DIR = str(sys.argv[5])"
    print "STAGE2_DIR = str(sys.argv[6])"
    print "PREFIX     = str(sys.argv[7])"
    print "STAGE      = str(sys.argv[8])"
    print "OUT_DIR    = str(sys.argv[9])"
    print
    sys.exit(1)
    
    
PKL        = str(sys.argv[1])
RUN        = str(sys.argv[2])
SUBRUN     = str(sys.argv[3])
EVENT      = str(sys.argv[4])
STAGE1_DIR = str(sys.argv[5])
STAGE2_DIR = str(sys.argv[6])
PREFIX     = str(sys.argv[7])
STAGE      = int(sys.argv[8])
OUT_DIR    = str(sys.argv[9])


print "PKL        = %s" % PKL
print "RUN        = %s" % RUN
print "SUBRUN     = %s" % SUBRUN
print "EVENT      = %s" % EVENT
print "STAGE1_DIR = %s" % STAGE1_DIR
print "STAGE2_DIR = %s" % STAGE2_DIR
print "PREFIX     = %s" % PREFIX
print "STAGE      = %d" % STAGE
print "OUT_DIR    = %s" % OUT_DIR

BINARY = "/usr/local/share/dllee_unified/LArCV/app/LArOpenCVHandle/ana/entry_copy/entry_copy"

#
# read RSE to get the entry
#
import pandas as pd

df = pd.read_pickle(PKL);

row = df.query("run==@RUN&subrun==@SUBRUN&event==@EVENT").iloc[0]
ENTRY = int(row['entry'])
FNAME = str(row['fname'])

_RUN    = int(FNAME.split("-")[0][3:])
_SUBRUN = int(FNAME.split("-")[-1][6:])

runmod100    = _RUN%100
rundiv100    = _RUN/100
subrunmod100 = _SUBRUN%100
subrundiv100 = _SUBRUN/100

JOBTAG = 10000*_RUN + _SUBRUN
INDIR  = "%03d/%02d/%03d/%02d/"%(rundiv100,runmod100,subrundiv100,subrunmod100)

#
# get the input files
#
PREFIX_FILE = ""
if STAGE == int(1):
    PREFIX_FILE = os.path.join(STAGE1_DIR,INDIR,"%s-%s.root" % (PREFIX,FNAME))
if STAGE == int(2):
    PREFIX_FILE = os.path.join(STAGE2_DIR,INDIR,"%s_%s.root" % (PREFIX,JOBTAG))
else:
    raise Exception

temp = tempfile.NamedTemporaryFile()
SS = ""
ENTRY_V = [ENTRY]
for entry in ENTRY_V:
    SS += "%d" % int(entry)
    SS += " ";
SS = SS[:-1]
print "WRITE [%s]" % SS
temp.write(SS)
temp.flush()

infile_v  = [PREFIX_FILE]

outfile  = os.path.basename(PREFIX_FILE).split(".")[0]
outfile += "_%s_%s_%s.root"
outfile  = outfile % (RUN,SUBRUN,EVENT)

outfile_v = [os.path.join(OUT_DIR,outfile)]

SS = "%s %s %s %s"
for infile,outfile in zip(infile_v,outfile_v):
    if infile == "": continue
    print
    print "@infile=%s" % infile
    print "@outfile=%s" % outfile
    print
    SS0 = str(SS)
    SS0 = SS0 % (BINARY,infile,outfile,temp.name)
    print SS0
    os.system(SS0)
    print
temp.close()
sys.exit(0)
