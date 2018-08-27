import os, sys, gc

if len(sys.argv) != 5:
    print 
    print "VTX_DF  = str(sys.argv[1]) -- ana_comb_df"
    print "NUE_DF  = str(sys.argv[2]) -- nueid_comb_df"
    print "NUM     = str(sys.argv[3]) -- jobid"
    print "OUTDIR  = str(sys.argv[4])"
    print 
    sys.exit(1)

import ROOT
import numpy as np
import pandas as pd

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

VTX_DF  = str(sys.argv[1])
NUE_DF  = str(sys.argv[2])
NUM     = str(sys.argv[3])
OUTDIR  = str(sys.argv[4])

from util.fill_df import *

print "--> initialize_rst(...)"

rst_df = initialize_rst(VTX_DF,NUE_DF)

rst_df.to_pickle(os.path.join(OUTDIR,"comb_df_%s.pkl" % NUM))

del rst_df
gc.collect()

print "---> done"
sys.exit(0)
