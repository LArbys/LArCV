print "--------------"
print "|  DUMP HADD |"
print "--------------"

import sys, os

if len(sys.argv) != 3:
    print
    print "INFILE1 = str(sys.argv[1])"
    print "INFILE2 = str(sys.argv[2])"
    print
    sys.exit(1)

INFILE1=str(sys.argv[1])
INFILE2=str(sys.argv[2])

print "INFILE1=%s"%INFILE1
print "INFILE2=%s"%INFILE2

OUTFILE = INFILE2.split(".")[0]+"_hadd.root"

print "OUTFILE=%s" % OUTFILE

SS = "hadd -f %s %s %s"
SS = SS % (OUTFILE,INFILE1,INFILE2)

print SS
os.system(SS)

sys.exit(0)
