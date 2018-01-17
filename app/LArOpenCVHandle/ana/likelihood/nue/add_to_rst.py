import os, sys, gc

if len(sys.argv) != 4:
    print 
    print "RST_DF   = str(sys.argv[1]) -- rst_comb_df"
    print "NUMU_ROOT= str(sys.argv[2]) -- FinalVertexVariables"
    print "OUTDIR   = str(sys.argv[3])"
    print 
    sys.exit(1)

import ROOT
import numpy as np
import pandas as pd
BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

RST_DF    = str(sys.argv[1])
NUMU_ROOT = str(sys.argv[2])
OUTDIR    = str(sys.argv[3])

num = int(os.path.basename(RST_DF).split(".")[0].split("_")[-1])

from util.fill_df import *

print "--> add_to_rst(...)"

new_rst_df = add_to_rst(RST_DF,NUMU_ROOT)

new_rst_df.to_pickle(os.path.join(OUTDIR,"rst_numu_comb_df_%d.pkl" % num))

del new_rst_df
gc.collect()

print "---> done"
sys.exit(0)
