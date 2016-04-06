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


        self.plt  = self.win.addPlot()

        
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
        
        # # event choicesoptions
        self.axis_plot = QtGui.QPushButton("Plot")
        self.lay_inputs.addWidget( self.axis_plot, 0, 2 )

        self.previous_plot = QtGui.QPushButton("Previous Event")
        self.lay_inputs.addWidget( self.previous_plot, 0, 3 )
        
        self.next_plot = QtGui.QPushButton("Next Event")
        self.lay_inputs.addWidget( self.next_plot, 0, 4 )
        

        self.axis_plot.clicked.connect( self.plotData )

        self.previous_plot.clicked.connect( self.previousEvent )
        self.next_plot.clicked.connect( self.nextEvent )

        self.dm = datamanager.DataManager(rfile)



    def previousEvent(self):

        event = int(self.event.text())

        if event == 0:
            return
        
        self.event.setText(str(event-1))

        
        self.plotData()

        
    def nextEvent(self):
        
        event = int(self.event.text())
        self.event.setText(str(event+1))

        self.plotData()
        
    def plotData(self):

        #Clear out plot
        self.plt.clear()

        #Add image
        self.imi = pg.ImageItem()
        self.plt.addItem(self.imi)        
        
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
        self.plt.addItem(r1)
        
        toc = time.clock()
        print "added item: {} s".format(toc - tic)

        delay = 0
        for roi_p in rois:
            for plane in roi_p:
                bbox = roi_p[plane]
                r1 = pg.QtGui.QGraphicsRectItem(meta.col(bbox['tl'][0]),
                                                meta.row(bbox['tl'][1] - delay),
                                                bbox['width'],
                                                bbox['height'])
                # r1 = pg.QtGui.QGraphicsRectItem(meta.col(bbox['tl'][0]),meta.row(bbox['tl'][1]-delay),20,20)
                # r2 = pg.QtGui.QGraphicsRectItem(meta.col(bbox['br'][0]),meta.row(bbox['br'][1]-delay),20,20)
                # r3 = pg.QtGui.QGraphicsRectItem(meta.col(bbox['tr'][0]),meta.row(bbox['tr'][1]-delay),20,20)
                # r4 = pg.QtGui.QGraphicsRectItem(meta.col(bbox['bl'][0]),meta.row(bbox['bl'][1]-delay),20,20)
                
                r1.setPen(pg.mkPen(self.co[plane]))
                r1.setBrush(pg.mkBrush(None))
                self.plt.addItem(r1)

                # r2.setPen(pg.mkPen(self.co[plane]))
                # r2.setBrush(pg.mkBrush(None))
                # self.plt.addItem(r2)

                # r3.setPen(pg.mkPen(self.co[plane]))
                # r3.setBrush(pg.mkBrush(None))
                # self.plt.addItem(r3)
            
                # r4.setPen(pg.mkPen(self.co[plane]))
                # r4.setBrush(pg.mkBrush(None))
                # self.plt.addItem(r4)
                                
            
        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        # import pdb; pdb.set_trace()
