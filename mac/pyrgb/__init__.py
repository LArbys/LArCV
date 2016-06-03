from PyQt4 import QtGui, QtCore

import ROOT
from ROOT import larcv
import pyqtgraph as pg

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

import numpy as np

larcv.load_pyutil

try:
    import cv2
except:
    print "NO CV2"

