#thanks taritree

import os,sys,copy
from . import QtGui, QtCore
from . import pg
import numpy as np
import time

import datamanager

from hoverrect import HoverRect as HR

class RGBDisplay(QtGui.QWidget) :
    def __init__(self,rfile):
        super(RGBDisplay,self).__init__()
        self.resize( 1200, 700 )

        self.win = pg.GraphicsWindow(title="Fuck P100")
        self.win.setWindowTitle('Fuck P100')

        self.plt  = self.win.addPlot()

        
        self.co = { 0 : 'r', 1 : 'g' , 2 : 'b' }
        self.particles = [

            "Eminus",
            "Kminus",
            "Proton",
            "Muminus",
            "Piminus",
            "Gamma",
            "Pizero",
            "BNB",
            "Cosmic",
            "Unknown"
        ]
        
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

        
        # event choicesoptions
        self.axis_plot = QtGui.QPushButton("Plot")
        self.lay_inputs.addWidget( self.axis_plot, 0, 6 )

        self.previous_plot = QtGui.QPushButton("Previous Event")
        self.lay_inputs.addWidget( self.previous_plot, 0, 7 )

        self.next_plot = QtGui.QPushButton("Next Event")
        self.lay_inputs.addWidget( self.next_plot, 0, 8 )

        
        self.kBNB   = QtGui.QRadioButton("BNB")
        self.lay_inputs.addWidget( self.kBNB, 0, 9 )

        self.kBNB.setChecked(True)
        self.kOTHER = QtGui.QRadioButton("Particle")
        self.lay_inputs.addWidget( self.kOTHER, 0, 10 )
        
        self.kBOTH  = QtGui.QRadioButton("Both")
        self.lay_inputs.addWidget( self.kBOTH, 0, 11 )


        self.kTypes = { 'kBNB'  :  (self.kBNB,[7]),
                        'kOTHER' : (self.kOTHER,[i for i in xrange(10) if i != 7]),
                        'kBOTH'  : (self.kBOTH,[ i for i in xrange(10) ])}
        
        self.image = None
    
        self.axis_plot.clicked.connect( self.plotData )

        self.previous_plot.clicked.connect( self.previousEvent )
        self.next_plot.clicked.connect( self.nextEvent )
        
        self.kBNB.clicked.connect(  lambda: self.drawBBOX(self.kTypes['kBNB'][1]) )
        self.kOTHER.clicked.connect(lambda: self.drawBBOX(self.kTypes['kOTHER'][1]) )
        self.kBOTH.clicked.connect( lambda: self.drawBBOX(self.kTypes['kBOTH'][1]) )

        self.boxes = []
        self.dm = datamanager.DataManager(rfile)
        
        

    def which_type(self):
        for button in self.kTypes:
            if self.kTypes[button][0].isChecked():
                return self.kTypes[button][1]
        return None
    
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

        self.image = None
        
        #Clear out plot
        self.plt.clear()

        #Add image
        self.imi = pg.ImageItem()
        self.plt.addItem(self.imi)        
        
        event = int(self.event.text())
        imin  = int(self.imin.text())
        imax  = int(self.imax.text())
        
        b,self.rois,imgs = self.dm.get_event_image(event,imin,imax)

        self.imi.setImage(b)

        meta = imgs[0].meta()
        outline = pg.QtGui.QGraphicsRectItem(0,0,meta.cols(),meta.rows())
        outline.setPen(pg.mkPen('w'))
        outline.setBrush(pg.mkBrush(None))
        self.plt.addItem(outline)

        self.image = imgs[0]
        
        self.drawBBOX(self.which_type())

    def drawBBOX(self,kType):
        
        if self.image is None:
            return

        if kType is None:
            return
        
        for box in self.boxes:
            self.plt.removeItem(box)

        self.boxes = []
            
        for roi_p in self.rois:

            if roi_p['type'] not in kType:
                continue
            
            for ix,bbox in enumerate(roi_p['bbox']):
                
                imm = self.image.meta()

                x = bbox.bl().x - imm.bl().x
                y = bbox.bl().y - imm.bl().y

                # print "bbox bl x: {} bbox bl y: {}".format(x,y)
                
                dw_i = imm.cols() / ( imm.tr().x - imm.bl().x )
                dh_i = imm.rows() / ( imm.tr().y - imm.bl().y )

                # print "dw_i : {} dh_i : {}".format(dw_i,dh_i)
                
                w_b = bbox.tr().x - bbox.bl().x
                h_b = bbox.tr().y - bbox.bl().y

                # print "w_b : {} h_b : {}".format(w_b,h_b)
                ti = pg.TextItem(text=self.particles[ roi_p['type'] ])
                ti.setPos(x*dw_i,(y+h_b)*dh_i+1)
                
                r1 = HR(x * dw_i,
                        y * dh_i,
                        w_b * dw_i,
                        h_b * dh_i,
                        ti,self.plt)

                
                r1.setPen(pg.mkPen(self.co[ix]))
                r1.setBrush(pg.mkBrush(None))
                self.plt.addItem(r1)
                self.boxes.append(r1)

        def showParticle(self):
            print "aho"

                
