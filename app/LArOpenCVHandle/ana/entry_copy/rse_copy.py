import os,sys, tempfile

if len(sys.argv) < 8:
    print
    print "SSNET_FILE  = str(sys.argv[1])"
    print "PGRAPH_FILE = str(sys.argv[2])"
    print "SHOWER_FILE = str(sys.argv[3])"
    print "TRACK_FILE  = str(sys.argv[4])"
    print "OPFLASH_FILE= str(sys.argv[5])"
    print "OUTPUT_DIR  = str(sys.argv[6])"
    print "ENTRY_V     = sys.argv[7:]"
    print
    sys.exit(1)

SSNET_FILE  = str(sys.argv[1])
PGRAPH_FILE = str(sys.argv[2])
SHOWER_FILE = str(sys.argv[3])
TRACK_FILE  = str(sys.argv[4])
OPFLASH_FILE= str(sys.argv[5])
OUTPUT_DIR  = str(sys.argv[6])
ENTRY_V     = sys.argv[7:]

#BINARY = "/usr/local/share/dllee_unified/LArCV/app/LArOpenCVHandle/ana/entry_copy/entry_copy"
BINARY = "/home/vgenty/sw/larcv/app/LArOpenCVHandle/ana/entry_copy/entry_copy"

infile_v  = [SSNET_FILE, PGRAPH_FILE, SHOWER_FILE, TRACK_FILE, OPFLASH_FILE]
outfile_v = []

for infile in infile_v:

    if infile != "":
        outfile = os.path.basename(infile).split(".")
        outfile = outfile[0] + "_filter.root"
        outfile = os.path.join(OUTPUT_DIR,outfile)
    else:
        outfile = ""

    outfile_v.append(outfile)


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
