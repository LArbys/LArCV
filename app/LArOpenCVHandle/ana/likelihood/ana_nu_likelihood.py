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
all_df = pd.read_pickle(os.path.join(BASE_PATH,"ll_dump","{}_all.pkl".format(name)))
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
nu_ass_df = pd.read_pickle(os.path.join(BASE_PATH,"ll_dump","{}_post_nue.pkl".format(name)))
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
ll_df = pd.read_pickle(os.path.join(BASE_PATH,"ll_dump","{}_post_LL.pkl".format(name)))
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

from util.plot import eff_plot

eff_plot(event_df,
         ll_df.query("LL>-16.25 & scedr<@scedr"),
         200,1500,20,
         "energyInit >= @Xmin & energyInit <= @Xmax","energyInit",
         "energyInit [MeV]","Count / 20 [MeV]",
         "energyInit")
         
