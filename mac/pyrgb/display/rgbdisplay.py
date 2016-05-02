#thanks taritree

import os,sys,copy,re
from .. import QtGui, QtCore
from .. import pg

import numpy as np
import time

from ..lib.datamanager import DataManager
from ..lib import storage as STORAGE

from ..lib.hoverrect import HoverRect as HR


from caffelayout import CaffeLayout
from ..rgb_caffe.testwrapper import TestWrapper

from cv2display import CV2Display

class RGBDisplay(QtGui.QWidget) :

    def __init__(self,argv):
        super(RGBDisplay,self).__init__()
        
        ### DataManager for loading the plot image
        self.dm = DataManager(argv)

        ### Size the canvas
        self.resize( 1200, 700 )

        ### Sliding ROI which we will do OpenCV manipulations on
        self.swindow = pg.ROI([0, 0], [10, 10])
        self.swindow.addScaleHandle([0.0, 0.0], [0.5, 0.5])
        self.swindow.addScaleHandle([0.0, 1.0], [0.5, 0.5])
        self.swindow.addScaleHandle([1.0, 1.0], [0.5, 0.5])
        self.swindow.addScaleHandle([1.0, 0.0], [0.5, 0.5])

        self.swindow.sigRegionChanged.connect(self.regionChanged)

        ### Graphics window which will hold the image
        self.win  = pg.GraphicsWindow()
        self.plt  = self.win.addPlot()

        # Handles to the axis which we will update with wire/tick
        self.plt_x = self.plt.getAxis('bottom')
        self.plt_y = self.plt.getAxis('left')
        
        ### Main Layout
        self.layout  = QtGui.QGridLayout()
        # run information up top
        self.runinfo = QtGui.QLabel("<b>Run:</b> -1 <b>Subrun:</b> -1 <b>Event:</b> -1")
        self.layout.addWidget( self.runinfo, 0, 0)
        self.layout.addWidget( self.win, 1, 0, 1, 10 )
        self.setLayout(self.layout)
        
        ### Input Widgets
        ### Layouts
        self.lay_inputs = QtGui.QGridLayout()
        self.layout.addLayout( self.lay_inputs, 2, 0 )
        
        ### Navigation
        self.event = QtGui.QLineEdit("%d"%0)      # event number
        self.lay_inputs.addWidget( QtGui.QLabel("Entry"), 0, 0)
        self.lay_inputs.addWidget( self.event, 0, 1 )

        ### imin
        self.imin = QtGui.QLineEdit("%d"%(5)) 
        self.lay_inputs.addWidget( QtGui.QLabel("imin"), 0, 2)
        self.lay_inputs.addWidget( self.imin, 0, 3 )

        ### imax
        self.imax = QtGui.QLineEdit("%d"%(400))
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

        #Check boxes for drawing plane1/2/3 -- perhaps should
        #become tied to current image being shown (N planes...)
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
        
        #Combo box to select the image producer
        self.lay_inputs.addWidget( QtGui.QLabel("Image2D"), 0, 12)
        self.comboBoxImage = QtGui.QComboBox()
        self.image_producer = None
        self.high_res = False
        for prod in self.dm.keys['image2d'] :
            self.comboBoxImage.addItem(prod)
            
        self.lay_inputs.addWidget( self.comboBoxImage, 1, 12 )

        #and another combo box to select ROI
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
        
        #Auto range function        
        self.auto_range = QtGui.QPushButton("AutoRange")
        self.lay_inputs.addWidget( self.auto_range, 0, 14 )

        # Yes or no to draw ROI (must hit replot)
        self.draw_bbox = QtGui.QCheckBox("Draw ROI")
        self.draw_bbox.setChecked(True)
        self.lay_inputs.addWidget( self.draw_bbox, 1, 14 )

        #RGBCaffe will open and close bottom of the window
        self.rgbcaffe = QtGui.QPushButton("Enable RGBCaffe")
        self.rgbcv2   = QtGui.QPushButton("Enable OpenCV")
        
        self.rgbcaffe.setFixedWidth(130)
        self.rgbcv2.setFixedWidth(130)

        self.lay_inputs.addWidget( self.rgbcaffe, 0, 15 )
        self.lay_inputs.addWidget( self.rgbcv2, 1, 15 )
        
        #Particle types
        self.kTypes = { 'kBNB'  :  (self.kBNB  ,[2]), 
                        'kOTHER' : (self.kOTHER,[ i for i in xrange(10) if i != 2]),
                        'kBOTH'  : (self.kBOTH ,[ i for i in xrange(10) ])}
        
        ### The current image array, useful for getting meta
        self.image = None

        ### (Re)Plot button
        self.axis_plot.clicked.connect( self.plotData )

        ### Previous and Next event
        self.previous_plot.clicked.connect( self.previousEvent )
        self.next_plot.clicked.connect    ( self.nextEvent )

        ### Radio buttons for choosing type of ROI
        self.kBNB.clicked.connect   ( lambda: self.drawBBOX(self.kTypes['kBNB'][1]   ) )
        self.kOTHER.clicked.connect ( lambda: self.drawBBOX(self.kTypes['kOTHER'][1] ) )
        self.kBOTH.clicked.connect  ( lambda: self.drawBBOX(self.kTypes['kBOTH'][1]  ) )

        self.auto_range.clicked.connect( self.autoRange )

        ### Set of ROI's on the current view -- just "boxes"
        self.boxes = []

        self.comboBoxImage.activated[str].connect(self.chosenImageProducer)
        self.comboBoxROI.activated[str].connect(self.chosenROIProducer)

        self.chosenImageProducer()
        self.chosenROIProducer()

        self.pimg   = None
        self.modimg = None

        self.rgbcaffe.clicked.connect( self.expandWindow )
        self.rgbcv2.clicked.connect( self.openCVEditor )

        ### Caffe Widgets
        #wrapper for FORWARD function
        self.caffe_test   = TestWrapper() 
        #wrapper for the caffe specific layout
        self.caffe_layout = CaffeLayout(self.caffe_test)

        ### OpenCV Widgets
        #wrapper for the opencv specific window
        self.cv2_display = CV2Display()
        self.cv2_enabled = False

        

    def expandWindow(self):
        if re.search("Disable",self.rgbcaffe.text()) is None:
            self.rgbcaffe.setText("Disable RGBCaffe")
            self.resize( 1200, 900 )
            self.layout.addLayout( self.caffe_layout.grid(True), 3, 0 )
        else:
            self.rgbcaffe.setText("Enable RGBCaffe")
            self.layout.removeItem(self.caffe_layout.grid(False))
            self.resize( 1200, 700 )

    def openCVEditor(self):
        if re.search("Disable",self.rgbcv2.text()) is None:
            self.rgbcv2.setText("Disable OpenCV")
            self.cv2_display.enable()
            self.cv2_display.show()
            self.plt.addItem(self.swindow)
            self.cv2_enabled = True
        else:
            self.cv2_enabled = False
            self.rgbcv2.setText("Enable OpenCV")
            self.cv2_display.hide()
            self.plt.removeItem(self.swindow)

    def setRunInfo(self,run,subrun,event):
        self.runinfo.setText("<b>Run:</b> {} <b>Subrun:</b> {} <b>Event:</b> {}".format(run,subrun,event))
        
    def chosenImageProducer(self):
        self.image_producer = str(self.comboBoxImage.currentText())

    def chosenROIProducer(self):
        if self.roi_exists == True:
            self.roi_producer = str(self.comboBoxROI.currentText())

    def get_ticks(self):
        
        xmax,ymax,_ = self.pimg.shape
        meta        = self.image[0].meta()
        tr = meta.tr()
        bl = meta.bl()

        dy = int(tr.y - bl.y)
        dx = int(tr.x - bl.x)

        ymajor   = []
        yminor   = []
        yminor2  = []
        xmajor   = []
        xminor   = []
        xminor2  = []
        
        for y in xrange(dy):
            if y > ymax: break
            t = int(bl.y)+y
            label = (y,t)
            if y%10 != 0:
                yminor2.append(label)
                continue

            if y%25 != 0:
                yminor.append(label)
                continue
            
            ymajor.append( label )

        for x in xrange(dx):
            if x > xmax: break
            t = int(bl.x)+x
            label = (x,t)

            if x%25 != 0:
                xminor2.append(label)
                continue
            
            if x%50 != 0:
                xminor.append(label)
                continue

            xmajor.append( label )


        return ([xmajor,xminor,xminor2],[ymajor,yminor,yminor2])
    
    def autoRange(self):

        xticks, yticks = self.get_ticks()
        
        self.plt_y.setTicks(yticks)
        self.plt_x.setTicks(xticks)

        self.plt.autoRange()
        self.setRunInfo(self.dm.run,
                        self.dm.subrun,
                        self.dm.event)

        if self.cv2_enabled == True:
            self.plt.addItem(self.swindow)
            self.swindow.setZValue(10)
        self.modimage = None

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
        
        pimg, self.rois, plotimage = self.dm.get_event_image(event,imin,imax,
                                                             self.image_producer,
                                                             self.roi_producer,
                                                             self.views)

        self.image = plotimage.imgs
        
        if pimg is None:
            self.image = None
            return

        self.caffe_test.set_image(plotimage.orig_mat)
        self.pimg = pimg

        # Emplace the image on the canvas
        self.imi.setImage(self.pimg)
        self.modimage = None
        
        if self.rois is None:
            self.autoRange()
            return

        xmin,xmax,ymin,ymax = (1e9,0,1e9,0)
        for roi in self.rois:
            for bb in roi['bbox']:
                if xmin > bb.min_x(): xmin = bb.min_x()
                if xmax < bb.max_x(): xmax = bb.max_x()
                if ymin > bb.min_y(): ymin = bb.min_y()
                if ymax < bb.max_y(): ymax = bb.max_y()
        pixel_size=(None,None)
        for img in self.image:
            bb = img.meta()
            if xmin > bb.min_x(): xmin = bb.min_x()
            if xmax < bb.max_x(): xmax = bb.max_x()
            if ymin > bb.min_y(): ymin = bb.min_y()
            if ymax < bb.max_y(): ymax = bb.max_y()
            pixel_size = (bb.pixel_width(),bb.pixel_height())


        if self.roi_exists == True:
            self.drawBBOX( self.which_type() )

        self.autoRange()


    ### For now this is fine....
    def drawBBOX(self,kType):
        
        # set the planes to be drawn
        self.setViewPlanes()
        
        # no image to draw ontop of
        if self.image is None: 
            return
        
        # no type to draw
        if kType is None: 
            return

        # remove the current set of boxes
        for box in self.boxes:
            self.plt.removeItem(box)

        # if thie box is unchecked don't draw it
        if self.draw_bbox.isChecked() == False:
            return
        
        # clear boxes explicitly
        self.boxes = []
        
        # and makew new boxes
        for roi_p in self.rois:

            if roi_p['type'] not in kType:
                continue
            
            for ix,bbox in enumerate(roi_p['bbox']):

                if ix not in self.views: continue
                
                imm = self.image[ix].meta()

                # x,y below are relative coordinate of bounding-box w.r.t. image in original unit
                x = bbox.min_x() - imm.min_x()
                y = bbox.min_y() - imm.min_y()

                #dw_i is an image X-axis unit legnth in pixel. dh_i for Y-axis. (i.e. like 0.5 pixel/cm)
                dw_i = imm.cols() / ( imm.max_x() - imm.min_x() )
                dh_i = imm.rows() / ( imm.max_y() - imm.min_y() )

                #w_b is width of a rectangle in original unit
                w_b = bbox.max_x() - bbox.min_x()
                h_b = bbox.max_y() - bbox.min_y()
                
                ti = pg.TextItem(text=STORAGE.particle_types[ roi_p['type'] ])
                ti.setPos( x*dw_i , ( y + h_b )*dh_i + 1 )

                print x*dw_i,y*dh_i,w_b*dw_i,h_b*dh_i

                r1 = HR(x   * dw_i,
                        y   * dh_i,
                        w_b * dw_i,
                        h_b * dh_i,
                        ti,self.plt)

                r1.setPen(pg.mkPen(STORAGE.colors[ix]))
                r1.setBrush(pg.mkBrush(None))
                self.plt.addItem(r1)
                self.boxes.append(r1)

    def regionChanged(self):
        
        if self.modimage is None:
            self.modimage = np.zeros( list(self.pimg.shape) )

        sl = self.swindow.getArraySlice(self.pimg,self.imi)[0]
        
        # need mask if user doesn't want to overwrite
        if self.cv2_display.overwrite == False:
            idx = np.where( self.modimage == 1 )
            pcopy = self.pimg.copy()

        self.pimg[ sl ] = self.cv2_display.paint( self.pimg[ sl ] ) ##11,11,3

        # use mask to updated only pixels not already updated
        if self.cv2_display.overwrite == False:
            self.pimg[ idx ] = pcopy[ idx ]
            self.modimage[ sl ] = 1

        if self.cv2_display.transform == False:
            return
        
        self.imi.setImage(self.pimg)
        
        

        


        
        
        
