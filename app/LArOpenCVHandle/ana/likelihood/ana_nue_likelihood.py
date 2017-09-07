import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)

rse    = ['run','subrun','event']
rsev   = ['run','subrun','event','vtxid']
rserv  = ['run','subrun','event','roid','vtxid']

name = str(sys.argv[1])
print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "~~~~~~~~~ Raw Output ~~~~~~~~~"
print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
all_df   = pd.read_pickle(os.path.join(BASE_PATH,"ll_bin","{}_all.pkl".format(name)))
event_df = all_df.groupby(rse).nth(0) 

scedr=5
print "Loaded events...........",event_df.index.size
print "...num with cROI........",event_df.query("num_croi>0").index.size
print "...good cROI counter....",event_df.query("good_croi_ctr>0").index.size
print "...reco.................",event_df.query("num_vertex>0").index.size
print
if name in ['nue','ncpizero']:
    print "1L1P....................",event_df.query("selected1L1P==1").index.size
    print "...num with cROI........",event_df.query("num_croi>0 & selected1L1P==1").index.size
    print "...good cROI counter....",event_df.query("good_croi_ctr>0 & selected1L1P==1").index.size
    print "...reco.................",event_df.query("good_croi_ctr>0 & selected1L1P==1 & num_vertex>0").dropna().index.size
    print
    print "1L1P E in [200,800] MeV.",event_df.query("selected1L1P==1 & energyInit>=200 & energyInit<=800").index.size
    print "...num with cROI........",event_df.query("num_croi>0 & selected1L1P==1 & energyInit>=200 & energyInit<=800").index.size
    print "...good cROI counter....",event_df.query("selected1L1P==1 & good_croi_ctr>0 & energyInit>=200 & energyInit<=800").index.size
    print "...reco.................",event_df.query("selected1L1P==1 & good_croi_ctr>0 & energyInit>=200 & energyInit<=800 & num_vertex>0").dropna().index.size
    print

print "===> Total Vertices <===".format(scedr)
print "...total................",all_df.query("num_vertex>0").index.size
print "...events...............",len(all_df.query("num_vertex>0").groupby(rse))
print

if name in ['nue','ncpizero']:
    print "===> GOOD vertices scedr<{} <===".format(scedr)
    SS="scedr<@scedr"
    print "...total................",all_df.query("num_vertex>0").query(SS).index.size
    print "...events...............",len(all_df.query("num_vertex>0").query(SS).groupby(rse))
    print
    print ">>>good cROI<<<"
    SS="scedr<@scedr & good_croi_ctr>0"
    print "...total................",all_df.query("num_vertex>0").query(SS).index.size
    print "...events...............",len(all_df.query("num_vertex>0").query(SS).groupby(rse))
    print 
    print ">>>good cROI + 1L1P<<<"
    SS="scedr<@scedr & good_croi_ctr>0 & selected1L1P==1"
    print "...total................",all_df.query("num_vertex>0").query(SS).index.size
    print "...events...............",len(all_df.query("num_vertex>0").query(SS).groupby(rse))
    print
    print ">>>good cROI + 1L1P + E<<<"
    SS="scedr<@scedr & good_croi_ctr>0 & selected1L1P==1 & energyInit>=200 & energyInit<=800"
    print "...total................",all_df.query("num_vertex>0").query(SS).index.size
    print "...events...............",len(all_df.query("num_vertex>0").query(SS).groupby(rse))
    print




print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "~~~~~~~~~ Nue Assumption Output ~~~~~~~~~"
print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

all_df   = pd.read_pickle(os.path.join(BASE_PATH,"ll_bin","{}_post_nue.pkl".format(name)))
#event_df = all_df.groupby(rse).nth(0) 

scedr=5
print "Loaded events...........",event_df.index.size
print "...num with cROI........",event_df.query("num_croi>0").index.size
print "...good cROI counter....",event_df.query("good_croi_ctr>0").index.size
print "...reco.................",event_df.query("num_vertex>0").index.size
print
if name in ['nue','ncpizero']:
    print "1L1P....................",event_df.query("selected1L1P==1").index.size
    print "...num with cROI........",event_df.query("num_croi>0 & selected1L1P==1").index.size
    print "...good cROI counter....",event_df.query("good_croi_ctr>0 & selected1L1P==1").index.size
    print "...reco.................",event_df.query("good_croi_ctr>0 & selected1L1P==1 & num_vertex>0").dropna().index.size
    print
    print "1L1P E in [200,800] MeV.",event_df.query("selected1L1P==1 & energyInit>=200 & energyInit<=800").index.size
    print "...num with cROI........",event_df.query("num_croi>0 & selected1L1P==1 & energyInit>=200 & energyInit<=800").index.size
    print "...good cROI counter....",event_df.query("selected1L1P==1 & good_croi_ctr>0 & energyInit>=200 & energyInit<=800").index.size
    print "...reco.................",event_df.query("selected1L1P==1 & good_croi_ctr>0 & energyInit>=200 & energyInit<=800 & num_vertex>0").dropna().index.size
    print

print "===> Total Vertices <===".format(scedr)
print "...total................",all_df.query("num_vertex>0").index.size
print "...events...............",len(all_df.query("num_vertex>0").groupby(rse))
print

if name in ['nue','ncpizero']:
    print "=> GOOD vertices scedr<{} <=".format(scedr)
    SS="scedr<@scedr"
    print "...total................",all_df.query("num_vertex>0").query(SS).index.size
    print "...events...............",len(all_df.query("num_vertex>0").query(SS).groupby(rse))
    print
    print ">>>good cROI<<<"
    SS="scedr<@scedr & good_croi_ctr>0"
    print "...total................",all_df.query("num_vertex>0").query(SS).index.size
    print "...events...............",len(all_df.query("num_vertex>0").query(SS).groupby(rse))
    print 
    print ">>>good cROI + 1L1P<<<"
    SS="scedr<@scedr & good_croi_ctr>0 & selected1L1P==1"
    print "...total................",all_df.query("num_vertex>0").query(SS).index.size
    print "...events...............",len(all_df.query("num_vertex>0").query(SS).groupby(rse))
    print
    print ">>>good cROI + 1L1P + E<<<"
    SS="scedr<@scedr & good_croi_ctr>0 & selected1L1P==1 & energyInit>=200 & energyInit<=800"
    print "...total................",all_df.query("num_vertex>0").query(SS).index.size
    print "...events...............",len(all_df.query("num_vertex>0").query(SS).groupby(rse))
    print

print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "~~~~~~~~~ LL Output ~~~~~~~~~"
print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

LLCUT=-14.625
all_df   = pd.read_pickle(os.path.join(BASE_PATH,"ll_bin","{}_post_LL.pkl".format(name)))
#event_df = all_df.groupby(rse).nth(0) 

scedr=5
print "Loaded events...........",event_df.index.size
print "...num with cROI........",event_df.query("num_croi>0").index.size
print "...good cROI counter....",event_df.query("good_croi_ctr>0").index.size
print "...reco.................",event_df.query("num_vertex>0").index.size
print
if name in ['nue','ncpizero']:
    print "1L1P....................",event_df.query("selected1L1P==1").index.size
    print "...num with cROI........",event_df.query("num_croi>0 & selected1L1P==1").index.size
    print "...good cROI counter....",event_df.query("good_croi_ctr>0 & selected1L1P==1").index.size
    print "...reco.................",event_df.query("good_croi_ctr>0 & selected1L1P==1 & num_vertex>0").dropna().index.size
    print
    print "1L1P E in [200,800] MeV.",event_df.query("selected1L1P==1 & energyInit>=200 & energyInit<=800").index.size
    print "...num with cROI........",event_df.query("num_croi>0 & selected1L1P==1 & energyInit>=200 & energyInit<=800").index.size
    print "...good cROI counter....",event_df.query("selected1L1P==1 & good_croi_ctr>0 & energyInit>=200 & energyInit<=800").index.size
    print "...reco.................",event_df.query("selected1L1P==1 & good_croi_ctr>0 & energyInit>=200 & energyInit<=800 & num_vertex>0").dropna().index.size
    print

print "===> Total Vertices <===".format(scedr)
print "...total..LL............",all_df.query("num_vertex>0 & LL>@LLCUT").index.size
print "...events.LL............",len(all_df.query("num_vertex>0 & LL>@LLCUT").groupby(rse))
print

if name in ['nue','ncpizero']:
    print "=> GOOD vertices scedr<{} <=".format(scedr)
    SS="scedr<@scedr"
    print "...total..LL............",all_df.query("num_vertex>0 & LL>@LLCUT ").query(SS).index.size
    print "...events.LL............",len(all_df.query("num_vertex>0 & LL>@LLCUT").query(SS).groupby(rse))
    print
    print ">>>good cROI<<<"
    SS="scedr<@scedr & good_croi_ctr>0"
    print "...total..LL............",all_df.query("num_vertex>0 & LL>@LLCUT ").query(SS).index.size
    print "...events.LL............",len(all_df.query("num_vertex>0 & LL>@LLCUT ").query(SS).groupby(rse))
    print 
    print ">>>good cROI + 1L1P<<<"
    SS="scedr<@scedr & good_croi_ctr>0 & selected1L1P==1"
    print "...total..LL............",all_df.query("num_vertex>0 & LL>@LLCUT").query(SS).index.size
    print "...events.LL............",len(all_df.query("num_vertex>0 & LL>@LLCUT").query(SS).groupby(rse))
    print
    print ">>>good cROI + 1L1P + E<<<"
    SS="scedr<@scedr & good_croi_ctr>0 & selected1L1P==1 & energyInit>=200 & energyInit<=800"
    print "...total..LL............",all_df.query("num_vertex>0 & LL>@LLCUT").query(SS).index.size
    print "...events.LL............",len(all_df.query("num_vertex>0 & LL>@LLCUT").query(SS).groupby(rse))
    print


if name != 'nue':
    sys.exit(1)

#
# Kinematic Plots
#
from util.plot import eff_plot

ll_df = pd.read_pickle(os.path.join(BASE_PATH,"ll_bin","{}_post_LL.pkl".format(name)))

eff_plot(event_df.query("good_croi_ctr>0 & selected1L1P==1"),
         ll_df.query("good_croi_ctr>0 & LL>@LLCUT & scedr<@scedr & selected1L1P==1"),
         200,800,20,
         "energyInit >= @Xmin & energyInit <= @Xmax","energyInit",
         "energyInit [MeV]","Count / 20 [MeV]",
         "energyInit")

for df in [event_df,ll_df]:
    p0_px = df['daughterPx_v'].str[0].values
    p0_py = df['daughterPy_v'].str[0].values
    p0_pz = df['daughterPz_v'].str[0].values
    p0_cosYZ = p0_pz / np.sqrt( p0_pz*p0_pz + p0_py*p0_py )

    p1_px = df['daughterPx_v'].str[1].values
    p1_py = df['daughterPy_v'].str[1].values
    p1_pz = df['daughterPz_v'].str[1].values
    p1_cosYZ = p1_pz / np.sqrt( p1_pz*p1_pz + p1_py*p1_py )

    p01_dot  = p0_px * p1_px + p0_py * p1_py + p0_pz * p1_pz
    p01_dot /= np.sqrt(p0_px*p0_px + p0_py*p0_py + p0_pz*p0_pz)
    p01_dot /= np.sqrt(p1_px*p1_px + p1_py*p1_py + p1_pz*p1_pz)

    df['P0ThetaYZ'] = np.nan_to_num(np.arccos(p0_cosYZ))
    df['P1ThetaYZ'] = np.nan_to_num(np.arccos(p1_cosYZ))
    df['P01Theta']  = np.nan_to_num(np.arccos(p01_dot))

eff_plot(event_df.query("good_croi_ctr>0 & selected1L1P==1"),
         ll_df.query("good_croi_ctr>0 & LL>@LLCUT & scedr<@scedr & selected1L1P==1"),
         0,3.14,3.14/20.,
         "energyInit >= 200 & energyInit <= 800","P0ThetaYZ",
         "P0ThetaYZ [rad]","Count / pi/20 [rad]",
         "P0ThetaYZ")

eff_plot(event_df.query("good_croi_ctr>0 & selected1L1P==1"),
         ll_df.query("good_croi_ctr>0 & LL>@LLCUT & scedr<@scedr & selected1L1P==1"),
         0,3.14,3.14/20.,
         "energyInit >= 200 & energyInit <= 800","P1ThetaYZ",
         "P1ThetaYZ [rad]","Count / pi/20 [rad]",
         "P1ThetaYZ")

eff_plot(event_df.query("good_croi_ctr>0 & selected1L1P==1"),
         ll_df.query("good_croi_ctr>0 & LL>@LLCUT & scedr<@scedr & selected1L1P==1"),
         0,3.14,3.14/20.,
         "energyInit >= 200 & energyInit <= 800","P01Theta",
         "P01Theta [rad]","Count / pi/20 [rad]",
         "P01Theta")

         
#
# Likelihood Plots

from util.plot import histo1, histo2, eff_scan

histo1(ll_df,
       -50,0,1.0,
       None,"LL",
       "LL","Fraction of Events",
       "LL")

histo2(ll_df,
       -50,0,1.0,
       "scedr>5","scedr<=5","LL",
       "LL","Fraction of Events",
       "LL_scedr")

eff_scan(ll_df.query("scedr>@scedr"),
         ll_df.query("scedr<=@scedr"),
         -50,0,0.5,
         "LL",
         "LL","Efficiency",
         "LL",)

