import os, sys, gc

if len(sys.argv) != 6:
    print
    print "INPUT_DF   = str(sys.argv[1])"
    print "PDF_FILE   = str(sys.argv[2])"
    print "LINE_FILE  = str(sys.argv[3])"
    print "IS_MC      = bool(int(sys.argv[4]))"
    print "OUT_DIR    = str(sys.argv[5])"
    print
    print "...bye"
    print
    sys.exit(1)

INPUT_DF  = str(sys.argv[1])
PDF_FILE  = str(sys.argv[2])
LINE_FILE = str(sys.argv[3])
IS_MC     = bool(int(sys.argv[4]))
OUT_DIR   = str(sys.argv[5])

print "IS_MC=",IS_MC
num = int(os.path.basename(INPUT_DF).split(".")[0].split("_")[-1])

import ROOT
import numpy as np
import pandas as pd

import root_numpy as rn

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

from util.common import *
from util.ll_functions import *

print "Reading PDFs..."
lepton_spec_m, proton_spec_m, cosmic_spec_m = read_nue_pdfs(PDF_FILE)
print "... read"

print "Reading in the line..."
line_param = read_line_file(LINE_FILE)
print "... read"

print "Reading in file... %s" % INPUT_DF
df = pd.read_pickle(INPUT_DF)
print "... read"

print "Prep..."
df_co = prep_common_vars(df,ismc=IS_MC)
df_ll = prep_test_df(df_co,copy=True)
df_ll = prep_two_par_df(df_ll,copy=False)
df_ll = prep_LL_vars(df_ll,ismc=IS_MC)
print "...preped"

print "LL..."
df_ll = LL_reco_particle(df_ll,lepton_spec_m,proton_spec_m)
df_ll = LL_reco_nue(df_ll,lepton_spec_m,proton_spec_m,cosmic_spec_m)
df_ll = LL_reco_line(df_ll,line_param)
df_ll = LL_reco_parameters(df_ll)
print "...LLed"

# write out
new_col_v = [col for col in list(df_ll.columns) if col not in list(df_co.columns)]

for new_col in new_col_v: df_co[new_col] = np.nan

df_co.loc[df_ll.index,new_col_v] = df_ll[new_col_v]

df_ll.to_pickle(os.path.join(OUT_DIR,"LL_comb_df_%d.pkl" % num))
df_co.to_pickle(os.path.join(OUT_DIR,"rst_LL_comb_df_%d.pkl" % num))

sys.exit(0)
