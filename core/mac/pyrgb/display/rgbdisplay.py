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
        
        self.co = { 0 : 'r', 1 : 'g' , 2 : 'b' }
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
        b,rois,imgs = self.dm.get_event_image(event)
        tic = time.clock()        
    
    
        self.imi.setImage(b)

        for i in xrange(3):
            meta = imgs[i].meta()
            print meta.tl().x,meta.tl().y,meta.col(meta.tl().x),meta.row(meta.tl().y)
            print meta.bl().x,meta.bl().y,meta.col(meta.bl().x),meta.row(meta.bl().y)
            print meta.tr().x,meta.tr().y,meta.col(meta.tr().x),meta.row(meta.tr().y)
            print meta.br().x,meta.br().y,meta.col(meta.br().x),meta.row(meta.br().y)
            print "))))"

        
        r1 = pg.QtGui.QGraphicsRectItem(meta.col(meta.tl().x),meta.row(meta.tl().y),meta.width(),meta.height())
        r1.setPen(pg.mkPen('w'))
        r1.setBrush(pg.mkBrush(None))
        self.vb.addItem(r1)
        
        toc = time.clock()
        print "added item: {} s".format(toc - tic)
        print rois
        for roi_p in rois:
            for plane in roi_p:
                bbox = roi_p[plane]
                r1 = pg.QtGui.QGraphicsRectItem(meta.col(bbox['tl'][0]),meta.row(bbox['tl'][1]),bbox['width'],bbox['height'])
                r1.setPen(pg.mkPen(self.co[plane]))
                r1.setBrush(pg.mkBrush(None))
                self.vb.addItem(r1)
            
            
        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        # import pdb; pdb.set_trace()
