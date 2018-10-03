import sys,os

if len(sys.argv) != 3:
    print ""
    print "PDF_ROOT = str(sys.argv[1])"
    print "OUTDIR   = str(sys.argv[2])"
    print ""
    sys.exit(1)

PDF_ROOT = str(sys.argv[1])
OUTDIR   = str(sys.argv[2])

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

from lib.ll_functions import dump_ll_pdfs

dump_ll_pdfs(PDF_ROOT,OUTDIR)

sys.exit(0)
