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

print "--> initialize_truth(...)"
truth_df = pd.DataFrame()
truth_df = initialize_truth(ANAFILE)

vertex_df = pd.DataFrame()

try:
    print "--> initialize_df(...)"
    vertex_df = initialize_df(ANAFILE)
    vertex_df.to_pickle(os.path.join(OUTDIR,"ana_vertex_df_%d.pkl" % num))

    print "--> initialize_r(...)"    
    comb_df = pd.DataFrame()
    comb_df = initialize_r(truth_df,vertex_df)
    comb_df.to_pickle(os.path.join(OUTDIR,"ana_comb_df_%d.pkl" % num))

    del vertex_df
    gc.collect()

    del comb_df
    gc.collect()

except IOError:
    print "...no vertex found in file"
    truth_df.to_pickle(os.path.join(OUTDIR,"ana_comb_df_%d.pkl" % num))

truth_df.to_pickle(os.path.join(OUTDIR,"ana_truth_df_%d.pkl" % num))

del truth_df
gc.collect()

print "---> done"
sys.exit(0)
