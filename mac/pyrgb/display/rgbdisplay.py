#thanks taritree

import os,sys,copy,re
from .. import QtGui, QtCore
from .. import pg
import numpy as np
import time

import datamanager
from ..lib.storage   import Storage

from ..lib.hoverrect import HoverRect as HR

import cv2


class RGBDisplay(QtGui.QWidget) :

    def __init__(self,argv):
        super(RGBDisplay,self).__init__()

        
        ### Hold constants
        self.st = Storage()

        ### DataManager
        self.dm = datamanager.DataManager(argv)

        self.resize( 1200, 700 )

        self.win = pg.GraphicsWindow()

        self.plt  = self.win.addPlot()
        
        ### Main Layout
        self.layout = QtGui.QGridLayout()
        self.layout.addWidget( self.win, 0, 0, 1, 10 )
        self.setLayout(self.layout)
        
        ### -------------
        ### Input Widgets
        ### -------------
        
        ### Layouts
        self.lay_inputs = QtGui.QGridLayout()
        self.layout.addLayout( self.lay_inputs, 1, 0 )
        
        ### Navigation
        self.event = QtGui.QLineEdit("%d"%0)      # event number
        self.lay_inputs.addWidget( QtGui.QLabel("Event"), 0, 0)
        self.lay_inputs.addWidget( self.event, 0, 1 )

        ### imin
        self.imin = QtGui.QLineEdit("%d"%(0)) 
        self.lay_inputs.addWidget( QtGui.QLabel("imin"), 0, 2)
        self.lay_inputs.addWidget( self.imin, 0, 3 )

        ### imax
        self.imax = QtGui.QLineEdit("%d"%(5))
        self.lay_inputs.addWidget( QtGui.QLabel("imax"), 0, 4)
        self.lay_inputs.addWidget( self.imax, 0, 5 )

        
        ### select choice options
        self.axis_plot = QtGui.QPushButton("Replot")
        self.lay_inputs.addWidget( self.axis_plot, 1, 0, 1 , 2  )

        self.previous_plot = QtGui.QPushButton("Prev. Event")
        self.lay_inputs.addWidget( self.previous_plot, 1, 2, 1, 2 )

        self.next_plot = QtGui.QPushButton("Next Event")
        self.lay_inputs.addWidget( self.next_plot, 1, 4, 1 , 2 )
        

        ### particle types
        #BNB
        self.kBNB   = QtGui.QRadioButton("BNB")
        self.lay_inputs.addWidget( self.kBNB, 0, 9 )
        self.kBNB.setChecked(True)

        #Particle
        self.kOTHER = QtGui.QRadioButton("Particle")
        self.lay_inputs.addWidget( self.kOTHER, 0, 10 )

        #Both
        self.kBOTH  = QtGui.QRadioButton("Both")
        self.lay_inputs.addWidget( self.kBOTH, 0, 11 )

        self.p0 = QtGui.QCheckBox("Plane 0")
        self.p0.setChecked(True)
        self.lay_inputs.addWidget( self.p0, 1, 9 )
        
        self.p1 = QtGui.QCheckBox("Plane 1")
        self.p1.setChecked(True)
        self.lay_inputs.addWidget( self.p1, 1, 10 )

        self.p2 = QtGui.QCheckBox("Plane 2")
        self.p2.setChecked(True)
        self.lay_inputs.addWidget( self.p2, 1, 11 )

        self.planes = [ self.p0, self.p1, self.p2 ]
        self.views = []
        
        self.lay_inputs.addWidget( QtGui.QLabel("Image2D"), 0, 12)
        self.comboBoxImage = QtGui.QComboBox()
        self.image_producer = None
        self.high_res = False
        for prod in self.dm.keys['image2d'] :
            self.comboBoxImage.addItem(prod)
            
        self.lay_inputs.addWidget( self.comboBoxImage, 1, 12 )

        self.lay_inputs.addWidget( QtGui.QLabel("ROI"), 0, 13)
        self.comboBoxROI = QtGui.QComboBox()
        self.roi_producer   = None

        if 'partroi' in self.dm.keys.keys():
            self.roi_exists = True
            for prod in self.dm.keys['partroi'] :
                self.comboBoxROI.addItem(prod)
        else:
            self.roi_exists = False
            self.comboBoxROI.addItem("None")

        self.lay_inputs.addWidget( self.comboBoxROI, 1, 13 )
        
        #Compressed or not comporessed image
        # self.compression  = QtGui.QCheckBox("Compressed")
        # self.lay_inputs.addWidget( self.compression, 0, 14 )
        # self.compression.setChecked(True)

        
        self.auto_range = QtGui.QPushButton("AutoRange")
        self.lay_inputs.addWidget( self.auto_range, 0, 14 )


        self.kTypes = { 'kBNB'  :  (self.kBNB  ,[2]), 
                        'kOTHER' : (self.kOTHER,[ i for i in xrange(10) if i != 2]),
                        'kBOTH'  : (self.kBOTH ,[ i for i in xrange(10) ])}
        
        ### The current image array, useful for meta
        self.image = None

        ### Plot button
        self.axis_plot.clicked.connect( self.plotData )

        ### Previous and Next 
        self.previous_plot.clicked.connect( self.previousEvent )
        self.next_plot.clicked.connect    ( self.nextEvent )

        ### Radio buttons
        self.kBNB.clicked.connect   ( lambda: self.drawBBOX(self.kTypes['kBNB'][1] ) )
        self.kOTHER.clicked.connect ( lambda: self.drawBBOX(self.kTypes['kOTHER'][1]) )
        self.kBOTH.clicked.connect  ( lambda: self.drawBBOX(self.kTypes['kBOTH'][1])  )

        self.auto_range.clicked.connect( self.autoRange )
        ### Set of ROI's on view
        self.boxes = []


        self.comboBoxImage.activated[str].connect(self.chosenImageProducer)
        self.comboBoxROI.activated[str].connect(self.chosenROIProducer)

        self.chosenImageProducer()
        self.chosenROIProducer()
        
        self.plotData()


    def chosenImageProducer(self):
        self.image_producer = str(self.comboBoxImage.currentText())
        self.highres=False
        #if re.search("mcint",self.image_producer) is None and re.search("segment",self.image_producer) is None:
        #    self.highres = False
        #else:
        #    self.highres = True

        
    def chosenROIProducer(self):

        if self.roi_exists == True:
            self.roi_producer = str(self.comboBoxROI.currentText())
        
    def autoRange(self):
        self.plt.autoRange()

    
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

    def setViewPlanes(self):

        self.views = []
        for ix, p in enumerate( self.planes ):
            if p.isChecked():
                self.views.append(ix)

                
    def plotData(self):

        self.image = None
        
        #Clear out plot
        self.plt.clear()

        #Add image
        self.imi = pg.ImageItem()
        self.plt.addItem(self.imi)        

        #From QT
        event = int( self.event.text())
        imin  = int( self.imin.text() )
        imax  = int( self.imax.text() )


        self.setViewPlanes()
        
        pimg, self.rois, self.image = self.dm.get_event_image(event,imin,imax,
                                                              self.image_producer,
                                                              self.roi_producer,
                                                              self.views,
                                                              self.highres)

        if pimg is None:
            self.image = None
            return

        # Display the image
        self.imi.setImage(pimg)

        if self.roi_exists == True:
            self.drawBBOX( self.which_type() )

        xmin,xmax,ymin,ymax = (1e9,0,1e9,0)
        for roi in self.rois:
            for bb in roi['bbox']:
                if xmin > bb.min_x(): xmin = bb.min_x()
                if xmax < bb.max_x(): xmax = bb.max_x()
                if ymin > bb.min_y(): ymin = bb.min_y()
                if ymax < bb.max_y(): ymax = bb.max_y()
        self.autoRange()
        print ymin,ymax,xmin,xmax
        #self.plt.setYRange(ymin,ymax,padding=0)
        #self.plt.setXRange(xmin,xmax,padding=0)


    ### For now this is fine....
    def drawBBOX(self,kType):

        self.setViewPlanes()
        
        if self.image is None: #no image was drawn
            return

        if kType is None: #no type to draw
            return
        
        for box in self.boxes:
            self.plt.removeItem(box)

        self.boxes = []
            
        for roi_p in self.rois:

            if roi_p['type'] not in kType:
                continue
            
            for ix,bbox in enumerate(roi_p['bbox']):

                if ix not in self.views: continue
                
                imm = self.image[ix].meta()

                x = bbox.bl().x - imm.bl().x
                y = bbox.bl().y - imm.bl().y
                
                dw_i = imm.cols() / ( imm.tr().x - imm.bl().x )
                dh_i = imm.rows() / ( imm.tr().y - imm.bl().y )

                
                w_b = bbox.tr().x - bbox.bl().x
                h_b = bbox.tr().y - bbox.bl().y
                
                print "bbox bl().x {} bbox bl().y {} imm bl().x {} imm bl().y {}".format(bbox.bl().x,bbox.bl().y,imm.bl().x,imm.bl().y) 
                if self.highres == True:
                    dw_i = 1.0;
                    dh_i = 1.0;
                    x = bbox.bl().x
                    #Temporary hack bbox.bl() doesn't match imm.bl() !!
                    y = bbox.bl().y

                
                #Set the text
                ti = pg.TextItem(text=self.st.particle_types[ roi_p['type'] ])
                ti.setPos( x*dw_i , ( y + h_b )*dh_i + 1 )

                # print "ix: {} bbox x {} y {} wb {} hb {} dw_i {} dh_i {}".format(ix,x,y,w_b,h_b,dw_i,dh_i)
                
                r1 = HR(x * dw_i,
                        y * dh_i,
                        w_b * dw_i,
                        h_b * dh_i,
                        ti,self.plt)

                r1.setPen(pg.mkPen(self.st.colors[ix]))
                r1.setBrush(pg.mkBrush(None))
                self.plt.addItem(r1)
                self.boxes.append(r1)

