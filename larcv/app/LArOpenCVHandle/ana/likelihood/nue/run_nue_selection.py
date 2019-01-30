import os, sys, gc

if len(sys.argv) != 6:
    print 
    print "COMB_DF  = str(sys.argv[1])"
    print "CUT_CFG  = str(sys.argv[2])"
    print "DROP     = int(sys.argv[3])"
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
CUT_CFG  = str(sys.argv[2])
DROP     = int(str(sys.argv[3]))
NUM      = str(sys.argv[4])
OUTDIR   = str(sys.argv[5])

from lib.cuts import parse_cuts
from lib.cuts import apply_cuts

drop_list = None
if DROP == 1: 
    drop_list = os.path.join(BASE_PATH,"txt","drop_list.txt")

cut_df = parse_cuts(COMB_DF,drop_list=drop_list)
cut_df = apply_cuts(cut_df,CUT_CFG)

out_file = os.path.join(OUTDIR,"nue_selected_df_%s.pkl" % NUM)
cut_df.to_pickle(out_file)

del cut_df
gc.collect()

sys.exit(0)
