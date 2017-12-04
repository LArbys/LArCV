import os, sys, gc

if len(sys.argv) != 8:
    print 
    print "SHR_ANA1  = str(sys.argv[1]) -- showerqualsingle"
    print "SHR_TRUTH = str(sys.argv[2]) -- shower_truth_match"
    print "TRK_ANA1  = str(sys.argv[3]) -- trackqualsingle"
    print "TRK_ANA2  = str(sys.argv[4]) -- tracker_anaout"
    print "TRK_TRUTH = str(sys.argv[5]) -- track_truth_match"
    print "TRK_PGRPH = str(sys.argv[6]) -- track_pgraph_match"
    print "OUTDIR    = str(sys.argv[7])"
    print 
    sys.exit(1)

import ROOT
import numpy as np
import pandas as pd
BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

SHR_ANA1  = str(sys.argv[1])
SHR_TRUTH = str(sys.argv[2])
TRK_ANA1  = str(sys.argv[3])
TRK_ANA2  = str(sys.argv[4])
TRK_TRUTH = str(sys.argv[5])
TRK_PGRPH = str(sys.argv[6])
OUTDIR    = str(sys.argv[7])

num = int(os.path.basename(SHR_ANA1).split(".")[0].split("_")[-1])

from util.fill_df import *

print "--> initialize_st(...)"

st_df = initialize_st(SHR_ANA1,
                      SHR_TRUTH,
                      TRK_ANA1,
                      TRK_ANA2,
                      TRK_TRUTH,
                      TRK_PGRPH)

st_df.to_pickle(os.path.join(OUTDIR,"st_comb_df_%d.pkl" % num))
del st_df
gc.collect()

print "---> done"
sys.exit(0)
