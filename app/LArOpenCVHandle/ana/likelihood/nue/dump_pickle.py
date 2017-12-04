import os, sys, gc

if len(sys.argv) != 3:
    print 
    print "ANAFILE = str(sys.argv[1])"
    print "OUTDIR  = str(sys.argv[2])" 
    print 
    sys.exit(1)

import ROOT
import numpy as np
import pandas as pd
BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

ANAFILE = str(sys.argv[1])
OUTDIR  = str(sys.argv[2])

num = int(os.path.basename(ANAFILE).split(".")[0].split("_")[-1])

from util.fill_df import *

print "--> truth_df(...)"
truth_df = initialize_truth(ANAFILE)
truth_df.to_pickle(os.path.join(OUTDIR,"ana_truth_df_%d.pkl" % num))
del truth_df
gc.collect()

print "--> initialize_df(...)"
try:
    all_df = initialize_df(ANAFILE)
    all_df.to_pickle(os.path.join(OUTDIR,"ana_all_df_%d.pkl" % num))
    del all_df
    gc.collect()
except IOError:
    print "...no vertex found in file"


print "---> done"
sys.exit(0)
