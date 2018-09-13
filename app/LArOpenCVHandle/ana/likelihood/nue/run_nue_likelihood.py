import os, sys, gc

if len(sys.argv) != 6:
    print 
    print "COMB_DF  = str(sys.argv[1])"
    print "LLEP     = str(sys.argv[2])"
    print "LLPC     = str(sys.argv[3])"
    print "NUM      = str(sys.argv[4])"
    print "OUTDIR   = str(sys.argv[5])"
    print 
    sys.exit(1)

import ROOT
import numpy as np
import pandas as pd

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

COMB_DF  = str(sys.argv[1])
LLEP     = str(sys.argv[2])
LLPC     = str(sys.argv[3])
NUM      = str(sys.argv[4])
OUTDIR   = str(sys.argv[5])

from lib.ll_functions import apply_ll

out_df = apply_ll(COMB_DF,LLEP,LLPC)

out_file = os.path.join(OUTDIR,"nue_ll_df_%s.pkl" % NUM)
out_df.to_pickle(out_file)

del out_df
gc.collect()

sys.exit(0)
