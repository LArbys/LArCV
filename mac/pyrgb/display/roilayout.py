from .. import pg
from .. import QtGui, QtCore
from ..lib.roislider import ROISliderGroup
from ..lib import storage as store
from ..lib.iomanager import IOManager
from .. import larcv

import copy

class ROIToolLayout(QtGui.QGridLayout):

    def __init__(self,plt,images,event):

        super(ROIToolLayout, self).__init__()

        # Sliding ROI which we will do OpenCV manipulations on
        self.name = "ROIToolLayout"

        self.enabled = False
        self.cv2 = None
        self.imi = None
        self.overwrite = False
        self.transform = True

        self.title  = QtGui.QLabel("<b>ROI Tool</b>")

        self.input_roi       = QtGui.QLineEdit("(Optional) Input ROI filename")
        self.input_roi_producer  = QtGui.QLineEdit("(Optional) Input ROI producer")
        # self.input_roi       = QtGui.QLineEdit("aho.root")
        # self.input_roi_producer  = QtGui.QLineEdit("boke")
        self.input_prod = None
        
        self.output_roi       = QtGui.QLineEdit("(Required) Output ROI filename")
        self.output_roi_producer  = QtGui.QLineEdit("(Required) Output ROI producer")
        # self.output_roi       = QtGui.QLineEdit("aho2.root")
        # self.output_roi_producer  = QtGui.QLineEdit("doji")
        self.output_prod = None

        self.load_files = QtGui.QPushButton("Load Files")
        
        self.add_roi = QtGui.QPushButton("Add ROIs")
        self.remove_roi = QtGui.QPushButton("Remove ROI")
        self.clear_roi = QtGui.QPushButton("Clear ROIs")
        self.capture_roi = QtGui.QPushButton("Capture ROIs")
        self.store_roi = QtGui.QPushButton("Store ROIs")
        self.reset_roi = QtGui.QPushButton("Reset ROIs")
        
        self.add_roi.clicked.connect(self.addROI)
        self.remove_roi.clicked.connect(self.removeROI)
        self.clear_roi.clicked.connect(self.clearROI)
        self.reset_roi.clicked.connect(self.resetROI)
        self.capture_roi.clicked.connect(self.captureROI)
        self.store_roi.clicked.connect(self.storeROI)

        self.load_files.clicked.connect(self.load)
                                        
        self.rois = []

        # Pointer to the current plot windows for placing ROI
        self.plt = plt

        # Pointer to current list of images for computing ROI
        self.images = images

        # Pointer to the current list of events
        self.event = event
    
        # The iomanager
        self.in_iom = None
        self.ou_iom = None

        self.user_rois       = {}
        self.user_rois_larcv = {}

    def storeROI(self):

        if self.ou_iom is None:
            "Print load a file first please!!!!"
            return

        # get the maximum key
        max_ = 0

        for event, rois in self.user_rois.iteritems():
            if int(event) > max_: max_ = event
            
        max_+=1
        for event in xrange(max_):

            roiarray = self.ou_iom.get_data(larcv.kProductROI,self.output_prod)
            self.ou_iom.set_id(1,0,event)
            
            if event not in self.user_rois.keys():
                self.ou_iom.save_entry()
                continue
            
            print "event ",event
            print "rois\n",rois

            for larcv_roi in self.user_rois_larcv[event]:
                roiarray.Append(larcv_roi)

            # save it to disk
            self.ou_iom.save_entry()
            
        self.ou_iom.finalize()
        
    def load(self):
        
        input_  = str(self.input_roi.text())
        output_ = str(self.output_roi.text())
        
        self.input_prod  = str(self.input_roi_producer.text())
        self.output_prod = str(self.output_roi_producer.text())
        
        if ".root" not in input_:
            input_ = []
            iomode = 1
            self.input_roi.setText("No input provided!")
            self.input_roi_producer.setText("None")
            self.input_prod = None
        else:
            self.in_iom = IOManager([input_],None,0)
            self.in_iom.set_verbosity(0)
            
        if ".root" not in output_:
            self.output_roi.setText("Not a valid ROOT file output name!")
            self.output_roi_producer.setText("Give me an output ROI producer!")
            self.output_prod = None
            return

        print " input is",input_," and output is",output_

        self.ou_iom = IOManager([],output_,1)
        self.ou_iom.set_verbosity(0)
    
    def captureROI(self):

        #self.user_rois[int(self.event.text())] = copy.deepcopy(self.rois) # make this this shit is right
        self.user_rois[int(self.event.text())] = self.rois # not allowed to copy qwidgets

        larcv_rois = []
        for roisg in self.rois:
            larcv_rois.append(self.roi2larcv(roisg))

        self.user_rois_larcv[int(self.event.text())] = larcv_rois # not allowed to copy qwidgets
        print "---"
        print self.user_rois
        print self.user_rois_larcv
        print "---"
        
    def addROI(self) :
        
        coords = [ [0,0,30,30] for _ in xrange(3) ]
        
        roisg = ROISliderGroup(coords,3,store.colors)

        self.rois.append(roisg)

        for roi in roisg.rois:
            self.plt.addItem(roi)

    def removeROI(self) :
        for roi in self.rois[-1].rois:
            self.plt.removeItem(roi)
            
        self.rois = self.rois[:-1]

    def clearROI(self) :

        while len(self.rois) > 0:
            for roi in self.rois[-1].rois:
                self.plt.removeItem(roi)
                
            self.rois = self.rois[:-1]

    def resetROI(self):
        self.clearROI()
        self.user_rois = {}
        self.user_rois_larcv = {}
        
    def reloadROI(self):

        self.clearROI()

        event = int(self.event.text())
        print "processing event",event,"self.user_rois.keys()",self.user_rois.keys()
        
        if event not in self.user_rois.keys():

            if self.in_iom is not None:
                if event < self.in_iom.get_n_entries():
                    self.in_iom.read_entry(event)
                    roiarray = self.in_iom.get_data(larcv.kProductROI,self.input_prod)
                    self.user_rois_larcv[event] = [roi for roi in roiarray.ROIArray()]
                    self.user_rois[event] = self.larcv2roi(self.user_rois_larcv[event])
                else:
                    return

            else:
                return
            
        self.rois = self.user_rois[event]

        for roisg in self.rois:
            for roi in roisg.rois:
                print type(roi),roi
                self.plt.addItem(roi)
    
    # add widgets to self and return 
    def grid(self, enable):

        if enable == True:
            self.enabled = True
            self.addWidget(self.title, 0, 0)
            self.addWidget(self.input_roi, 1, 0)
            self.addWidget(self.output_roi, 2, 0)
            self.addWidget(self.input_roi_producer, 1, 1)
            self.addWidget(self.output_roi_producer, 2, 1)
            
            self.addWidget(self.load_files, 3, 0)
            
            self.addWidget(self.add_roi, 1, 2)
            self.addWidget(self.remove_roi, 2, 2)
            self.addWidget(self.capture_roi, 3, 2)

            self.addWidget(self.clear_roi, 1, 3)
            self.addWidget(self.reset_roi, 2, 3)
            self.addWidget(self.store_roi, 3, 3)
            

        else:

            for i in reversed(range(self.count())):
                self.itemAt(i).widget().setParent(None)

            self.enabled = False

        return self

    def roi2larcv(self,bboxes):

        #store all ROIs
        
        larcv_roi = larcv.ROI()

        for ix,bbox in enumerate(bboxes.rois):
            size = bbox.size()
            pos = bbox.pos()

            print "ix",ix,"size",size,"pos",pos

            width,height,row_count,col_count,origin_x,origin_y = self.roi2imgcord(self.images.imgs[ix].meta(),size,pos)
            
            bbox_meta = larcv.ImageMeta(width,height,
                                        row_count,col_count,
                                        origin_x,origin_y,ix)

            larcv_roi.AppendBB(bbox_meta)

        return larcv_roi
    
    def roi2imgcord(self,imm,size,pos):

        x = pos[0]
        y = pos[1]

        dw_i = imm.cols() / (imm.max_x() - imm.min_x())
        dh_i = imm.rows() / (imm.max_y() - imm.min_y())

        x /= dw_i
        y /= dh_i

        # the origin
        origin_x = x + imm.min_x()
        origin_y = y + imm.min_y()
        
        # w_b is width of a rectangle in original unit
        w_b = size[0]
        h_b = size[1]

        # the width
        width  = w_b / dw_i
        height = h_b / dh_i

        # for now...
        row_count = 0
        col_count = 0

        # vic isn't sure why this is needed
        origin_y += height
        
        print "roi2img ROI: ",(width,height,row_count,col_count,origin_x,origin_y)
        return (width,height,row_count,col_count,origin_x,origin_y)

    def img2roicord(self,imm,bbox):
        
        # x,y  relative coordinate of bounding-box w.r.t. image in original unit
        x = bbox.min_x() - imm.min_x()
        y = bbox.min_y() - imm.min_y()

        # dw_i is an image X-axis unit legnth in pixel. dh_i for
        # Y-axis. (i.e. like 0.5 pixel/cm)
        dw_i = imm.cols() / (imm.max_x() - imm.min_x())
        dh_i = imm.rows() / (imm.max_y() - imm.min_y())

        # w_b is width of a rectangle in original unit
        w_b = bbox.max_x() - bbox.min_x()
        h_b = bbox.max_y() - bbox.min_y()

        print "img2roi ROI: ",(x*dw_i,  y*dh_i,  w_b*dw_i,  h_b*dh_i)
        return (x*dw_i,  y*dh_i,  w_b*dw_i,  h_b*dh_i)

    
    def larcv2roi(self,rois):
        converted_rois = []

        for roi in rois:
            coords = []
            
            for ix,bbox in enumerate(roi.BB()):

                coord = self.img2roicord(self.images.imgs[ix].meta(),bbox)
                coords.append(coord)
                
            roisg = ROISliderGroup(coords,len(coords),store.colors)
            
            converted_rois.append(roisg)
            

        return converted_rois
        

        
        
        
    
