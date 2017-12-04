import os, sys, gc

if len(sys.argv) != 5:
    print 
    print "VTX_DF  = str(sys.argv[1]) -- ana_all_df"
    print "ST_DF   = str(sys.argv[2]) -- st_comb_df"
    print "TRUE_DF = str(sys.argv[3]) -- ana_truth_df"
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
ST_DF   = str(sys.argv[2])
TRUE_DF = str(sys.argv[3])
OUTDIR  = str(sys.argv[4])

num = int(os.path.basename(VTX_DF).split(".")[0].split("_")[-1])

from util.fill_df import *

print "--> initialize_rst(...)"

rst_df = initialize_rst(VTX_DF,ST_DF,TRUE_DF)

rst_df.to_pickle(os.path.join(OUTDIR,"rst_comb_df_%d.pkl" % num))
del rst_df
gc.collect()

print "---> done"
sys.exit(0)
