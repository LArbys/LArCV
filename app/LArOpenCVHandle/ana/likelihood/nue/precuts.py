import os, sys, gc

if len(sys.argv) != 9:
    print
    print "INPUT_DF   = str(sys.argv[1])"
    print "COSMIC_ROOT= str(sys.argv[2])"
    print "FLASH_ROOT = str(sys.argv[3])"
    print "DEDX_ROOT  = str(sys.argv[4])"
    print "PRECUT_TXT = str(sys.argv[5])"
    print "OUT_PREFIX = str(sys.argv[6])"
    print "OUT_DIR    = str(sys.argv[7])"
    print "IS_MC      = str(sys.argv[8])"
    print
    sys.exit(1)


BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)
    
INPUT_DF   = str(sys.argv[1])
COSMIC_ROOT= str(sys.argv[2])
FLASH_ROOT = str(sys.argv[3])
DEDX_ROOT  = str(sys.argv[4])
PRECUT_TXT = str(sys.argv[5])
OUT_PREFIX = str(sys.argv[6])
OUT_DIR    = str(sys.argv[7])
IS_MC      = int(str(sys.argv[8]))


import pandas as pd
from util.precut_functions import perform_precuts

print "START"
comb_df = perform_precuts(INPUT_DF,
                          COSMIC_ROOT,
                          FLASH_ROOT,
                          DEDX_ROOT,
                          PRECUT_TXT,
                          IS_MC)
print "END"

print "Pickle output"
comb_df.to_pickle(os.path.join(OUT_DIR,OUT_PREFIX + ".pkl"))

sys.exit(0)
