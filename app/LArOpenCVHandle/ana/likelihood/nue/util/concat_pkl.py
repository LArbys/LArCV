import os,sys,gc

import pandas as pd


BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

from common import concat_pkls

def main(argv):

    if len(argv) < 3:
        print 
        print "......................"
        print "OUT_PREFIX = str(sys.argv[1])"
        print "PKLS_v     = list(sys.argv[2:])"
        print "......................"
        print 
        sys.exit(1)

    OUT_PREFIX = str(sys.argv[1])
    PKLS_v = list(sys.argv[2:])
    PKLS_v = [str(f) for f in PKLS_v]

    concat_pkls(PKLS_v,OUT_PREFIX)

    return

if __name__ == "__main__":
    main(sys.argv)
    sys.exit(1)
