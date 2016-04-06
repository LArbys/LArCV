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

        # # imin
        self.imin = QtGui.QLineEdit("%d"%(0)) 

        self.lay_inputs.addWidget( QtGui.QLabel("imin"), 0, 2)
        self.lay_inputs.addWidget( self.imin, 0, 3 )

        # # imax
        self.imax = QtGui.QLineEdit("%d"%(5))

        self.lay_inputs.addWidget( QtGui.QLabel("imax"), 0, 4)
        self.lay_inputs.addWidget( self.imax, 0, 5 )

        
        # # event choicesoptions
        self.axis_plot = QtGui.QPushButton("Plot")
        self.lay_inputs.addWidget( self.axis_plot, 0, 6 )

        self.previous_plot = QtGui.QPushButton("Previous Event")
        self.lay_inputs.addWidget( self.previous_plot, 0, 7 )
        
        self.next_plot = QtGui.QPushButton("Next Event")
        self.lay_inputs.addWidget( self.next_plot, 0, 8 )
        

        self.axis_plot.clicked.connect( self.plotData )

        self.previous_plot.clicked.connect( self.previousEvent )
        self.next_plot.clicked.connect( self.nextEvent )

        self.dm = datamanager.DataManager(rfile)



    def previousEvent(self):

        event = int(self.event.text())

        if event == 0:
            print "idiot.."
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
        imin   = int(self.imin.text())
        imax   = int(self.imax.text())
        
        b,rois,imgs = self.dm.get_event_image(event,imin,imax)
        tic = time.clock()        

        self.imi.setImage(b)

        # for i in xrange(3):
        #     meta = imgs[i].meta()
        #     print meta.tl().x,meta.tl().y,meta.col(meta.tl().x),meta.row(meta.tl().y)
        #     print meta.bl().x,meta.bl().y,meta.col(meta.bl().x),meta.row(meta.bl().y)
        #     print meta.tr().x,meta.tr().y,meta.col(meta.tr().x),meta.row(meta.tr().y)
        #     print meta.br().x,meta.br().y,meta.col(meta.br().x),meta.row(meta.br().y)
        #     print "))))"

            
        #r1 = pg.QtGui.QGraphicsRectItem(0,0,meta.tr().x,meta.tr().y)
        meta = imgs[0].meta()
        r1 = pg.QtGui.QGraphicsRectItem(0,0,meta.cols(),meta.rows())
        r1.setPen(pg.mkPen('w'))
        r1.setBrush(pg.mkBrush(None))
        self.plt.addItem(r1)
        
        toc = time.clock()
        print "added item: {} s".format(toc - tic)
        print rois
        delay = 0
        for roi_p in rois:
            for plane in roi_p:
                bbox = roi_p[plane]
                
                imm = imgs[0].meta()

                print "rows : {} cols : {} " .format(imm.rows(),imm.cols())
                
                x = bbox.bl().x - imm.bl().x
                y = bbox.bl().y - imm.bl().y

                print "bbox bl x: {} bbox bl y: {}".format(x,y)
                
                dw_i = imm.cols() / ( imm.tr().x - imm.bl().x )
                dh_i = imm.rows() / ( imm.tr().y - imm.bl().y )

                print "dw_i : {} dh_i : {}".format(dw_i,dh_i)
                
                w_b = bbox.tr().x - bbox.bl().x
                h_b = bbox.tr().y - bbox.bl().y


                print "w_b : {} h_b : {}".format(w_b,h_b)
                
                r1 = pg.QtGui.QGraphicsRectItem(x * dw_i,
                                                y * dh_i,
                                                w_b * dw_i,
                                                h_b * dh_i)
                                                
                
                r1.setPen(pg.mkPen(self.co[plane]))
                r1.setBrush(pg.mkBrush(None))
                self.plt.addItem(r1)
            
        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        # import pdb; pdb.set_trace()
