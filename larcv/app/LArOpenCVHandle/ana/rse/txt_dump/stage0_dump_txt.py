
print "--------------"
print "|  DUMP RSE  |"
print "--------------"

import sys, os

if len(sys.argv) != 3:
    print
    print "INFILE=str(sys.argv[1])"
    print "OUTFILE=str(sys.argv[2])"
    print
    sys.exit(1)

INFILE=str(sys.argv[1])
OUTFILE=str(sys.argv[2])

print "INFILE=%s"%INFILE
print "OUTFILE=%s"%OUTFILE

from larcv import larcv as larcv
iom = larcv.IOManager(larcv.IOManager.kREAD)
iom.add_in_file(INFILE)
iom.initialize()

nentries = int(iom.get_n_entries())

f = open(OUTFILE,"w+")
f.write("run,subrun,event\n")

for entry in xrange(nentries):
    print "@entry=",entry
    iom.read_entry(entry)
    
    ev_seg = iom.get_data(larcv.kProductImage2D,"wire")
    SS = "%d,%d,%d\n"
    SS = SS % (int(ev_seg.run()),
               int(ev_seg.subrun()),
               int(ev_seg.event()))
    print SS
    f.write(SS)

f.close()
sys.exit(0)
