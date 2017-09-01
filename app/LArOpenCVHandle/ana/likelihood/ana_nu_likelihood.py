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

#
# Get the All dataframe
#
all_df = pd.read_pickle(os.path.join(BASE_PATH,"ll_bin","{}_all.pkl".format(name)))
print "\t>>> Loaded",all_df.index.size,"all data <<<"
#
# Get the event dataframe
#
event_df = all_df.groupby(rse).nth(0) 
print "Loaded events.........",event_df.index.size
print "...good cROI counter..",event_df.query("good_croi_ctr>0").index.size
print "...reco...............",event_df.query("num_vertex>0").index.size
print
print "E in [200,800] MeV....",event_df.query("energyInit>=200 & energyInit<=800").index.size
print "...good cROI counter..",event_df.query("good_croi_ctr>0 & energyInit>=200 & energyInit<=800").index.size
print "...reco...............",event_df.query("energyInit>=200 & energyInit<=800 & num_vertex>0").dropna().index.size
print
scedr=5
print "Good vertices scedr<{}".format(scedr)
print "...total..............",all_df.query("num_vertex>0").query("scedr<5").index.size
print "...events.............",len(all_df.query("num_vertex>0").query("scedr<5").groupby(rse))
print "Bad vertices scedr>{} ".format(scedr)
print "...total..............",all_df.query("num_vertex>0").query("scedr>@scedr").index.size
print "...events.............",len(all_df.query("num_vertex>0").query("scedr>@scedr").groupby(rse))
print
print


#
# Get the {} assumption dataframe
#
nu_ass_df = pd.read_pickle(os.path.join(BASE_PATH,"ll_bin","{}_post_nue.pkl".format(name)))
print "\t>>> Loaded",nu_ass_df.index.size,"post nue assumption data <<<"
print "Events................",len(nu_ass_df.groupby(rse))
print "...good cROI counter..",len(nu_ass_df.query("good_croi_ctr>0").groupby(rse))
print "...reco...............",len(nu_ass_df.query("num_vertex>0").groupby(rse))
print
scedr=5
print "Good vertices scedr<{}".format(scedr)
print "...total..............",nu_ass_df.query("num_vertex>0").query("scedr<5").index.size
print "...events.............",len(nu_ass_df.query("num_vertex>0").query("scedr<5").groupby(rse))
print "Bad vertices scedr>{} ".format(scedr)
print "...total..............",nu_ass_df.query("num_vertex>0").query("scedr>@scedr").index.size
print "...events.............",len(nu_ass_df.query("num_vertex>0").query("scedr>@scedr").groupby(rse))
print
print

#
# Get the LL dataframe
# 
ll_df = pd.read_pickle(os.path.join(BASE_PATH,"ll_bin","{}_post_LL.pkl".format(name)))
print "\t>>> Loaded",ll_df.index.size,"LL events <<<"
print "Events................",len(ll_df.groupby(rse))
print "...good cROI counter..",len(ll_df.query("good_croi_ctr>0").groupby(rse))
print "...reco...............",len(ll_df.query("num_vertex>0").groupby(rse))
print
LLCUT=-16.25
print "\t >>> Applying LL cut @ > {} <<<".format(LLCUT)
print "Events................",len(ll_df.query("LL>@LLCUT").groupby(rse))
print "...good cROI counter..",len(ll_df.query("LL>@LLCUT").query("good_croi_ctr>0").groupby(rse))
print "...reco...............",len(ll_df.query("LL>@LLCUT").query("num_vertex>0").groupby(rse))
print "Good vertices scedr<{}".format(scedr)
print "...total..............",ll_df.query("LL>@LLCUT").query("scedr<@scedr").index.size
print "...events.............",len(ll_df.query("LL>@LLCUT").query("scedr<@scedr").groupby(rse))
print "Bad vertices scedr>{} ".format(scedr)
print "...total..............",ll_df.query("LL>@LLCUT").query("scedr>@scedr").index.size
print "...events.............",len(ll_df.query("LL>@LLCUT").query("scedr>@scedr").groupby(rse))
print
print


#
# Kinematic Plots
#
from util.plot import eff_plot

eff_plot(event_df.query("good_croi_ctr>0"),
         ll_df.query("good_croi_ctr>0 & LL>-16.25 & scedr<@scedr"),
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

eff_plot(event_df.query("good_croi_ctr>0"),
         ll_df.query("good_croi_ctr>0 & LL>-16.25 & scedr<@scedr"),
         0,3.14,3.14/20.,
         "energyInit >= 200 & energyInit <= 800","P0ThetaYZ",
         "P0ThetaYZ [rad]","Count / pi/20 [rad]",
         "P0ThetaYZ")

eff_plot(event_df.query("good_croi_ctr>0"),
         ll_df.query("good_croi_ctr>0 & LL>-16.25 & scedr<@scedr"),
         0,3.14,3.14/20.,
         "energyInit >= 200 & energyInit <= 800","P1ThetaYZ",
         "P1ThetaYZ [rad]","Count / pi/20 [rad]",
         "P1ThetaYZ")

eff_plot(event_df.query("good_croi_ctr>0"),
         ll_df.query("good_croi_ctr>0 & LL>-16.25 & scedr<@scedr"),
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

eff_scan(ll_df.query("scedr<=@scedr"),
         ll_df.query("scedr>@scedr"),
         -50,0,0.5,
         "LL",
         "LL","Efficiency",
         "LL",)

