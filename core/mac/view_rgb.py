#!/usr/bin python
#thanks taritree(!)

import os,sys
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
from pyrgb.display import rgbdisplay as rgbd


inputs = sys.argv
def usage() :
    print("\n\t" + '\033[91m' + " view_rgb.py [root file]..." + '\033[0m' + "\n")

if len(inputs) < 2:
    usage()
    sys.exit(1)


app = QtGui.QApplication([])
rgbdisplay = rgbd.RGBDisplay(sys.argv[1])
rgbdisplay.show()

if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    print "exec called ..."
    rgbdisplay.show()
    QtGui.QApplication.instance().exec_()
