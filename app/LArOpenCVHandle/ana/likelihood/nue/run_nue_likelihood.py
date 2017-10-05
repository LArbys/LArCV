import os, sys, gc

import ROOT
import numpy as np
import pandas as pd

import root_numpy as rn

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

from util.fill_df import *

rse    = ['run','subrun','event']
rserv  = ['run','subrun','event','roid','vtxid']

# Vertex data frame
dfs  = {}

# Event data frame
edfs = {}
mdfs = {}

name        = str(sys.argv[1])
INPUT_FILE  = str(sys.argv[2])
PDF_FILE    = str(sys.argv[3])

comb_df = initialize_df(INPUT_FILE)

print "Open storage..."
print
print "@ sample",name
print
print "Storing comb_df"
OUT   = os.path.join(BASE_PATH,"ll_bin","%s_all.pkl" % name)
comb_df.to_pickle(OUT)
print "DONE!"
print "...stored"
print "--> nue assumption"
comb_cut_df = nue_assumption(comb_df)
print "Removing comb_df..."
del comb_df
print "Collect..."
gc.collect()
print "...collected"
print "--> fill parameters"
comb_cut_df = fill_parameters(comb_cut_df)

print "Storing comb_cut_df"
OUT   = os.path.join(BASE_PATH,"ll_bin","%s_post_nue.pkl" % name)
comb_cut_df.to_pickle(OUT)
print "...stored"

#
# Choose the vertex @ event with the highest LL
#
print "Choosing vertex with max LL"
fin = os.path.join(BASE_PATH,"ll_bin",PDF_FILE + ".root")
passed_df = apply_ll(comb_cut_df,fin)

print "Storing passed_df"
OUT   = os.path.join(BASE_PATH,"ll_bin","%s_post_LL.pkl" % name)
passed_df.to_pickle(OUT)
print "...stored"

print "Done"
sys.exit(1)
