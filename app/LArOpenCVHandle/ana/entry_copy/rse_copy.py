import os,sys, tempfile

if len(sys.argv) != 9:
    print
    print "PKL       = str(sys.argv[1])"
    print "RUN       = str(sys.argv[2])"
    print "SUBRUN    = str(sys.argv[3])"
    print "EVENT     = str(sys.argv[4])"
    print "SSNET_DIR = str(sys.argv[5])"
    print "VTX_DIR   = str(sys.argv[6])"
    print "ST_DIR    = str(sys.argv[7])"
    print "OUT_DIR   = str(sys.argv[8])"
    print
    sys.exit(1)
    
    
PKL       = str(sys.argv[1])
RUN       = str(sys.argv[2])
SUBRUN    = str(sys.argv[3])
EVENT     = str(sys.argv[4])
SSNET_DIR = str(sys.argv[5])
VTX_DIR   = str(sys.argv[6])
ST_DIR    = str(sys.argv[7])
OUT_DIR   = str(sys.argv[8])


print "PKL        = %s" % PKL
print "RUN        = %s" % RUN
print "SUBRUN     = %s" % SUBRUN
print "EVENT      = %s" % EVENT
print "SSNET_DIR  = %s" % SSNET_DIR
print "VTX_DIR    = %s" % VTX_DIR
print "ST_DIR     = %s" % ST_DIR
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
SSNET_FILE   = os.path.join(SSNET_DIR,INDIR,"ssnetout-larcv-%s.root" % FNAME)
PGRAPH_FILE  = os.path.join(VTX_DIR,INDIR,"vertexout_%05d.root" % JOBTAG)
SHOWER_FILE  = os.path.join(ST_DIR,INDIR,"shower_reco_out_%05d.root" % JOBTAG)
TRACK_FILE   = os.path.join(ST_DIR,INDIR,"tracker_reco_%05d.root" % JOBTAG)
OPFLASH_FILE = os.path.join(SSNET_DIR,INDIR,"opreco-%s.root" % FNAME)

name_v    = ["ssnet",    "pgraph"   , "shower"   , "track"   , "op"        ]
infile_v  = [SSNET_FILE, PGRAPH_FILE, SHOWER_FILE, TRACK_FILE, OPFLASH_FILE]
outfile_v = []

for infile,name in zip(infile_v,name):

    if infile != "":
        outfile = name + "_Run%06d-SubRun%06d-Event%06d.root"
        outfile = outfile % (int(RUN),int(SUBRUN),int(EVENT))
        outfile = os.path.join(OUT_DIR,outfile)
    else:
        outfile = ""

    outfile_v.append(outfile)


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
