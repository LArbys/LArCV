
# coding: utf-8

# In[1]:

import os, sys
import ROOT
from ROOT import fcllite
from ROOT import geo2d
from larcv import larcv
import cv2
pygeo = geo2d.PyDraw()
from ROOT import larocv
from ROOT import std
from ROOT import cv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.rcParams['font.size']=20
matplotlib.rcParams['font.family']='serif'
import numpy as np


# In[ ]:

proc = larcv.ProcessDriver('ProcessDriver')
CFG="../cfg/prod_fullchain_ssnet_combined_beta.cfg"
#CFG="../cfg/prod_fullchain_ssnet_combined.cfg"

preprocessed=False

print "Loading config... ",CFG
proc.configure(CFG)
flist=ROOT.std.vector('std::string')()
flist.push_back("/Users/vgenty/Desktop/intrinsic_nue/out_pyroi/larcv_fcn_out.root")
proc.override_input_file(flist)
proc.override_output_file("/tmp/cacca.root")
proc.override_ana_file("/tmp/test.root")

vinroi_id  = proc.process_id("VertexInROI")
reco_id    = proc.process_id("LArbysImage")
larbysimg  = proc.process_ptr(reco_id)

proc.initialize()


# In[ ]:

event=0
proc.batch_process(event,1)

