import sys,os

if len(sys.argv) != 5:
    print ""
    print "SIG_PKL         = str(sys.argv[1])"
    print "BKG_PKL         = str(sys.argv[2])"
    print "SIG_MCINFO_ROOT = str(sys.argv[3])"
    print "BKG_MCINFO_ROOT = str(sys.argv[4])"
    print ""
    sys.exit(1)

SIG_PKL         = str(sys.argv[1])
BKG_PKL         = str(sys.argv[2])
SIG_MCINFO_ROOT = str(sys.argv[3])
BKG_MCINFO_ROOT = str(sys.argv[4])

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

from lib.ll_functions import generate_llem_pdfs

generate_llem_pdfs(SIG_PKL,
                   BKG_PKL,
                   SIG_MCINFO_ROOT,
                   BKG_MCINFO_ROOT)
sys.exit(0)
