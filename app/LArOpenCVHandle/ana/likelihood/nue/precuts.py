import os, sys, gc

if len(sys.argv) != 7:
    print
    print "INPUT_DF   = str(sys.argv[1])"
    print "COSMIC_ROOT= str(sys.argv[2])"
    print "FLASH_ROOT = str(sys.argv[3])"
    print "PRECUT_TXT = str(sys.argv[4])"
    print "OUT_PREFIX = str(sys.argv[5])"
    print "OUT_DIR    = str(sys.argv[6])"
    print
    sys.exit(1)


BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)
    
INPUT_DF   = str(sys.argv[1])
COSMIC_ROOT= str(sys.argv[2])
FLASH_ROOT = str(sys.argv[3])
PRECUT_TXT = str(sys.argv[4])
OUT_PREFIX = str(sys.argv[5])
OUT_DIR    = str(sys.argv[6])

import pandas as pd
import root_numpy as rn
from util.ll_functions import LL_reco_parameters
from util.precut_functions import *

print "Reading=",INPUT_DF
input_df = pd.read_pickle(INPUT_DF)
vertex_df = input_df.query("locv_num_vertex>0").copy()

cosmic_df = pd.DataFrame()
flash_df  = pd.DataFrame()

name_v = []
comb_v = []

if len(COSMIC_ROOT)>0:
    print "Reading=",COSMIC_ROOT
    cosmic_df= pd.DataFrame(rn.root2array(COSMIC_ROOT))

    name_v.append("Cosmic")
    comb_v.append(cosmic_df)

if len(FLASH_ROOT)>0:
    print "Reading=",FLASH_ROOT
    flash_df = pd.DataFrame(rn.root2array(FLASH_ROOT))

    name_v.append("Flash")
    comb_v.append(flash_df)

print "Combining..."
print "vertex_df sz=",vertex_df.index.size,"RSE=",len(vertex_df.groupby(RSE))

comb_df = vertex_df.copy()

for name,df in zip(name_v,df_v):
    print "@name=",name,"_df sz=",df.index.size,"RSE=",len(df.groupby(RSE))
    print "comb_df sz=",comb_df.index.size

    comb_df.set_index(RSEV,inplace=True)
    df.set_index(RSEV,inplace=True)

    comb_df = comb_df.join(df)
    
    comb_df.reset_index(inplace=True)
    df.reset_index(inplace=True)


print "Preparing precuts"
comb_df = prepare_precuts(comb_df)

print "Reading precuts=",PRECUT_TXT

cuts = ""
with open(PRECUT_TXT,'r') as f:
    cuts = f.read().split("\n")

if cuts[-1]=="": 
    cuts = cuts[:-1]

SS = ""
for ix,cut in enumerate(cuts): 
    SS+= "(" + cut + ")"
    if ix != len(cuts)-1:
        SS+= " and "
print "SS=",SS

print "Precutting"
comb_df.query(SS,inplace=True)

print "Setting particle ID"
comb_df = set_ssnet_particle_reco_id(df)

print "Preparing parameters"
comb_df = LL_reco_parameters(df)

print "Pickle output"
comb_df.to_pickle(os.path.join(OUT_DIR,OUT_PREFIX + ".pkl"))

sys.exit(0)
