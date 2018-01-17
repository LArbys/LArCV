import os,sys
import pandas as pd

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

res0 = pd.read_pickle("ana_comb_df_91031509.pkl")
res1 = pd.read_pickle("ana_truth_df_91031509.pkl")
res2 = pd.read_pickle("ana_vertex_df_91031509.pkl")
res3 = pd.read_pickle("st_comb_df_91031509.pkl")
res4 = pd.read_pickle("rst_comb_df_91031509.pkl")

res_v = [res0,res1,res2,res3,res4]

RSE=['run','subrun','event']

for ix,res in enumerate(res_v):
    print "@ix=",ix
    
    print res.index.size

    if ix in [0,2]:
        print res.query("num_vertex>0").index.size

    if ix in [4]:
        print res.query("locv_num_vertex>0").index.size

    print len(res.groupby(RSE))
    if ix in [4]:
        print_full(res.iloc[0])
    print
    print

sys.exit(0)
