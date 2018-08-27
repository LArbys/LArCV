import os, sys, gc

if len(sys.argv) != 4:
    print 
    print "COMB_DF = str(sys.argv[1])"
    print "NUM     = str(sys.argv[2])"
    print "OUTDIR  = str(sys.argv[3])"
    print 
    sys.exit(1)

import ROOT
import numpy as np
import pandas as pd

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

COMB_DF = str(sys.argv[1])
NUM     = str(sys.argv[2])
OUTDIR  = str(sys.argv[3])

from lib.cuts import apply_cuts

cut_df = apply_cuts(COMB_DF)

out_file = os.path.join(OUTDIR,"comb_cut_df_%s.pkl" % NUM)
cut_df.to_pickle(out_file)

del cut_df
gc.collect()

sys.exit(0)
