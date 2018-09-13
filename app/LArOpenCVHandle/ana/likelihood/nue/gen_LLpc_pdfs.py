import sys,os

if len(sys.argv) != 4:
    print ""
    print "SIG_PKL         = str(sys.argv[1])"
    print "BKG_PKL         = str(sys.argv[2])"
    print "SIG_MCINFO_ROOT = str(sys.argv[3])"
    print ""
    sys.exit(1)

SIG_PKL         = str(sys.argv[1])
BKG_PKL         = str(sys.argv[2])
SIG_MCINFO_ROOT = str(sys.argv[3])

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

from lib.ll_functions import generate_llpc_pdfs

generate_llpc_pdfs(SIG_PKL,
                   BKG_PKL,
                   SIG_MCINFO_ROOT)

sys.exit(0)
