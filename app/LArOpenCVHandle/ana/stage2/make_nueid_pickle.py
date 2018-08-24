import os, sys, gc

if len(sys.argv) != 9:
    print 
    print "SHR_ANA1   = str(sys.argv[1])  -- showerqualsingle"
    print "SHR_TRUTH  = str(sys.argv[2])  -- shower_truth_match"
    print "PID_ANA1   = str(sys.argv[3])  -- multipid_out"
    print "PID_ANA2   = str(sys.argv[4])  -- multiplicity_out"
    print "NUEID_ANA  = str(sys.argv[5])  -- nueid_ana"
    print "FLASH_ANA  = str(sys.argv[6])  -- flash_ana_nue"
    print "NUM        = str(sys.argv[7])"
    print "OUTDIR     = str(sys.argv[8])"
    print 
    sys.exit(1)

import ROOT
import numpy as np
import pandas as pd

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

SHR_ANA1   = str(sys.argv[1])
SHR_TRUTH  = str(sys.argv[2])
PID_ANA1   = str(sys.argv[3])
PID_ANA2   = str(sys.argv[4])
NUEID_ANA  = str(sys.argv[5])
FLASH_ANA  = str(sys.argv[6])
NUM        = str(sys.argv[7])
OUTDIR     = str(sys.argv[8])

from util.fill_df import initialize_nueid

print "--> make_nueid_pickle"

nueid_df = initialize_nueid(SHR_ANA1,
                            SHR_TRUTH,
                            PID_ANA1,
                            PID_ANA2,
                            NUEID_ANA,
                            FLASH_ANA)

nueid_df.to_pickle(os.path.join(OUTDIR,"nueid_comb_df_%s.pkl" % NUM))
del nueid_df
gc.collect()

print "---> done"
sys.exit(0)
