import os,sys, tempfile

if len(sys.argv) < 7:
    print
    print "SSNET_FILE  = str(sys.argv[1])"
    print "PGRAPH_FILE = str(sys.argv[2])"
    print "SHOWER_FILE = str(sys.argv[3])"
    print "TRACK_FILE  = str(sys.argv[4])"
    print "OUTPUT_DIR  = str(sys.argv[5])"
    print "ENTRY_V     = sys.argv[6:]"
    print
    sys.exit(1)

SSNET_FILE  = str(sys.argv[1])
PGRAPH_FILE = str(sys.argv[2])
SHOWER_FILE = str(sys.argv[3])
TRACK_FILE  = str(sys.argv[4])
OUTPUT_DIR  = str(sys.argv[5])
ENTRY_V     = sys.argv[6:]

#BINARY = "/usr/local/share/dllee_unified/LArCV/app/LArOpenCVHandle/ana/entry_copy/entry_copy"
BINARY = "/home/vgenty/sw/larcv/app/LArOpenCVHandle/ana/entry_copy/entry_copy"

if SSNET_FILE != "":
    SSNET_OUT = os.path.basename(SSNET_FILE).split(".")
    SSNET_OUT = SSNET_OUT[0] + "_filter.root"
    SSNET_OUT = os.path.join(OUTPUT_DIR,SSNET_OUT)
else:
    SSNET_OUT = ""

if PGRAPH_FILE != "":
    PGRAPH_OUT = os.path.basename(PGRAPH_FILE).split(".")
    PGRAPH_OUT = PGRAPH_OUT[0] + "_filter.root"
    PGRAPH_OUT = os.path.join(OUTPUT_DIR,PGRAPH_OUT)
else:
    PGRAPH_OUT = ""

if SHOWER_FILE != "":
    SHOWER_OUT = os.path.basename(SHOWER_FILE).split(".")
    SHOWER_OUT = SHOWER_OUT[0] + "_filter.root"
    SHOWER_OUT = os.path.join(OUTPUT_DIR,SHOWER_OUT)
else:
    SHOWER_OUT = ""

if TRACK_FILE != "":
    TRACK_OUT = os.path.basename(TRACK_FILE).split(".")
    TRACK_OUT = TRACK_OUT[0] + "_filter.root"
    TRACK_OUT = os.path.join(OUTPUT_DIR,TRACK_OUT)
else:
    TRACK_OUT = ""

infile_v  = [SSNET_FILE, PGRAPH_FILE, SHOWER_FILE, TRACK_FILE]
outfile_v = [SSNET_OUT,  PGRAPH_OUT,  SHOWER_OUT,  TRACK_OUT]

temp = tempfile.NamedTemporaryFile()
SS = ""
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
