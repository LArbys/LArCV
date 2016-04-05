#thanks taritree

import os,sys,copy
from . import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import time
#from ..pyfakefifo import PyFakeFifo
#import plotmanager

import datamanager

class RGBDisplay(QtGui.QWidget) :
    def __init__(self,rfile):
        super(RGBDisplay,self).__init__()
        self.resize( 1200, 700 )

        self.win = pg.GraphicsWindow(title="Fuck P100")
        self.win.setWindowTitle('Fuck P100')
        
        self.vb  = self.win.addViewBox()
        self.imi = pg.ImageItem()
        self.vb.addItem(self.imi)        
        
        # Main Layout
        self.layout = QtGui.QGridLayout()
        self.layout.addWidget( self.win, 0, 0, 1, 10 )
        self.setLayout(self.layout)
        
        # # -------------
        # # Input Widgets
        # # -------------
        
        # # Layouts
        self.lay_inputs = QtGui.QGridLayout()
        self.layout.addLayout( self.lay_inputs, 1, 0 )
        
        # # Navigation
        self.event = QtGui.QLineEdit("%d"%(0))      # event number

        self.lay_inputs.addWidget( QtGui.QLabel("Event"), 0, 0)
        self.lay_inputs.addWidget( self.event, 0, 1 )
        
        # # axis options
        self.axis_plot = QtGui.QPushButton("Plot!")
        self.lay_inputs.addWidget( self.axis_plot, 0, 2 )

    
        self.axis_plot.clicked.connect( self.plotData )

        self.dm = datamanager.DataManager(rfile)

    def plotData(self):
        event   = int(self.event.text())
        b = self.dm.get_event_image(event)
        tic = time.clock()        
    
    
        self.imi.setImage(b)

        
        toc = time.clock()
        print "added item: {} s".format(toc - tic)


        
