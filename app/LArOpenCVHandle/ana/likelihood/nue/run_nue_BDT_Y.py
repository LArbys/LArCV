import os, sys, gc

if len(sys.argv) != 6:
    print 
    print "COMB_DF  = str(sys.argv[1])"
    print "LLEM_BIN = str(sys.argv[2])"
    print "LLPC_BIN = str(sys.argv[3])"
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
LLEM_BIN = str(sys.argv[2])
LLPC_BIN = str(sys.argv[3])
NUM      = str(sys.argv[4])
OUTDIR   = str(sys.argv[5])

from lib.ll_functions import apply_bdt_y

out_df = apply_bdt_y(COMB_DF,LLEM_BIN,LLPC_BIN)

out_file = os.path.join(OUTDIR,"nue_bdt_df_%s.pkl" % NUM)
out_df.to_pickle(out_file)

del out_df
gc.collect()

sys.exit(0)
