import ROOT, sys, os
from ROOT import std

from larlite import larlite
from larcv import larcv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#0%matplotlib inline
import matplotlib.path as path
colormap=['blue','red','magenta','green','orange','yellow','pink']
nc=len(colormap)
from ROOT import geo2d
from ROOT import cv
from ROOT.cv import Point_ as Vector
DTYPE='float'


iom=larcv.IOManager(larcv.IOManager.kBOTH)
iom.reset()
iom.set_verbosity(0)
iom.add_in_file("mcc7_bnb_detsim_to_larcv_hires_v00_p00_out_0000.root")
iom.set_out_file("/tmp/trash.root")
iom.initialize()
iom.read_entry(10)#8

larbysimg=larcv.LArbysImage()
cfg=larcv.CreatePSetFromFile("unit.fcl","LArbysImage")
larbysimg.configure(cfg)
larbysimg.initialize()

larbysimg.process(iom)
