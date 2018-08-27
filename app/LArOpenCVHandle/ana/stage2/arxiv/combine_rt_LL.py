import os, sys, gc

if len(sys.argv) != 4:
    print 
    print "VERTEX_PKL   = str(sys.argv[1])  -- ana_comb_df"
    print "NUMU_LL_ROOT = str(sys.argv[2])  -- FinalVertexVariables"
    print "OUTDIR       = str(sys.argv[3])"
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
OUTDIR       = str(sys.argv[3])

num = int(os.path.basename(VERTEX_PKL).split(".")[0].split("_")[-1])

from util.fill_df import *

print "--> initialize_rt_LL(...)"

rt_LL_df = initialize_rt_LL(VERTEX_PKL,NUMU_LL_ROOT)

rt_LL_df.to_pickle(os.path.join(OUTDIR,"%s_%d.pkl" % (os.path.basename(VERTEX_PKL).replace(".pkl",""),num)))
del rt_LL_df
gc.collect()

print "---> done"
sys.exit(0)
