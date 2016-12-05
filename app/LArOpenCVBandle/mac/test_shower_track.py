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

proc = larcv.ProcessDriver("ProcessDriver")
proc.configure('shower_track_unit.fcl')
flist=std.vector('string')()
flist.push_back('nue_example.root')
proc.override_input_file(flist)
proc.initialize()

larbysimg = proc.process_ptr(0)

proc.batch_process(5,1)
