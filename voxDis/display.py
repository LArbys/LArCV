#!/usr/bin/env python
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

try:
    import pyqtgraph.opengl as gl
except:
    print("ERROR: Must have opengl for this viewer")
    exit()

from gui import voxDisGui
import argparse
import sys
import signal
from pyqtgraph.Qt import QtGui, QtCore

from ROOT import larcv

from manager import fileManager, geometry

# This is to allow key commands to work when focus is on a box




def sigintHandler(*args):
    """Handler for the SIGINT signal."""
    sys.stderr.write('\r')
    sys.exit()


def main():

    parser = argparse.ArgumentParser(description='Python based 3D event display.  Requires opengl.')
    parser.add_argument('file', nargs='*', help="Optional input file to use")

    args = parser.parse_args()

    app = QtGui.QApplication(sys.argv)

    geom = geometry.microboone()

    # If a file was passed, give it to the manager:

    manager = fileManager()
    # manager = fileManager(geom)
    manager.set_input_files(args.file)

    thisgui = voxDisGui(geom, manager)
    # manager.goToEvent(0)

    signal.signal(signal.SIGINT, sigintHandler)
    timer = QtCore.QTimer()
    timer.start(500)  # You may change this if you wish.
    timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.

    app.exec_()
    # sys.exit(app.exec_())


if __name__ == '__main__':
    main()
