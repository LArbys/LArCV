import os, sys, gc

if len(sys.argv) != 5:
    print 
    print "VERTEX_PKL   = str(sys.argv[1]) -- ana_comb_df"
    print "NUMU_LL_ROOT = str(sys.argv[2]) -- FinalVertexVariables"
    print "IS_MC        = str(sys.argv[3]) -- IS_MC"
    print "OUTDIR       = str(sys.argv[4])"
    print 
    sys.exit(1)

import ROOT
import numpy as np
import pandas as pd
BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

VERTEX_PKL   = str(sys.argv[1])
NUMU_LL_ROOT = str(sys.argv[2])
IS_MC        = bool(int(sys.argv[3]))
OUTDIR       = str(sys.argv[4])

num = int(os.path.basename(VERTEX_PKL).split(".")[0].split("_")[-1])

from util.fill_df import *
from util.ll_functions import *

print "--> initialize_rt(...)"

rt_df = initialize_rt(VERTEX_PKL,NUMU_LL_ROOT)
rt_df = prep_common_vars(rt_df,IS_MC)                      

rt_df.to_pickle(os.path.join(OUTDIR,"rt_LL_comb_df_%d.pkl" % num))
del rt_df
gc.collect()

print "---> done"
sys.exit(0)
