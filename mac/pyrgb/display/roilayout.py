from .. import pg
from .. import QtGui, QtCore
from ..lib.roislider import ROISliderGroup
from ..lib import storage as store
from ..lib.iomanager import IOManager
from .. import larcv

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

        self.input_label = QtGui.QLabel("Input:")
        self.output_label = QtGui.QLabel("Output:")
        self.input_prod_label = QtGui.QLabel("Producer:")
        self.output_prod_label = QtGui.QLabel("Producer:")
        
        self.input_roi = QtGui.QLineEdit("Input ROI filename")
        self.input_roi_producer  = QtGui.QLineEdit("Input ROI producer")
        self.input_prod = None
        
        self.output_roi = QtGui.QLineEdit("Output ROI filename")
        self.output_roi_producer = QtGui.QLineEdit("Output ROI producer")
        self.output_prod = None

        self.load_files = QtGui.QPushButton("Load Files")
        
        self.add_roi = QtGui.QPushButton("Add ROIs")
        self.remove_roi = QtGui.QPushButton("Remove ROI")
        self.clear_roi = QtGui.QPushButton("Clear ROIs")
        self.capture_roi = QtGui.QPushButton("Capture ROIs")
        self.store_roi = QtGui.QPushButton("Store ROIs")
        self.reset_roi = QtGui.QPushButton("Reset ROIs")


        self.roi_p1 = QtGui.QCheckBox("R:")
        self.roi_p1.setChecked(True)
        self.roi_p2 = QtGui.QCheckBox("G:")
        self.roi_p2.setChecked(True)
        self.roi_p3 = QtGui.QCheckBox("B:")
        self.roi_p3.setChecked(True)
        
        self.fixed_roi_box = QtGui.QCheckBox("Fixed ROI")
        self.fixed_roi_box.setChecked(False)
        
        self.fixed_w_label = QtGui.QLabel("W:")
        self.fixed_h_label = QtGui.QLabel("H:")

        self.fixed_w = QtGui.QLineEdit("30")
        self.fixed_h = QtGui.QLineEdit("30")
        
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
        
        self.user_rois = {}
        self.user_rois_larcv = {}


        self.checked_planes = []
        
    def storeROI(self):

        if self.ou_iom is None:
            "Please load a file first please!!!!"
            return

        # get the maximum key
        max_ = 0

        for event, rois in self.user_rois.iteritems():
            if int(event) > max_: max_ = event
            
        max_+=1

        for event in xrange(max_):

            roiarray = self.ou_iom.get_data(larcv.kProductROI,self.output_prod)

            # event == TTree entry in the image file so I place that in the event number here.


            # If this event doesn't have an ROI, save a blank and continue
            if event not in self.user_rois.keys():
                
                self.ou_iom.set_id(1,0,event)
                self.ou_iom.save_entry()
                
                continue

            # User accidentally hit capture ROIs when no ROI drawn, save a blank and continue
            if len(self.user_rois[event]) == 0:

                self.ou_iom.set_id(1,0,event)
                self.ou_iom.save_entry()

                continue

            # It's a fine ROI, put a 1 in the subrun to indicate one exists
            self.ou_iom.set_id(1,1,event)

            # There is ROI so lets append the larcv converted ROIs and put them into the ROOT file
            for larcv_roi in self.user_rois_larcv[event]:
                roiarray.Append(larcv_roi)

            # Save them to the tree
            self.ou_iom.save_entry()

        # Put it on the disk
        self.ou_iom.finalize()
        
    def load(self):
        
        input_  = str(self.input_roi.text())
        output_ = str(self.output_roi.text())
        
        self.input_prod  = str(self.input_roi_producer.text())
        self.output_prod = str(self.output_roi_producer.text())

        # No ROOT file in the input, don't make a read iomanager
        if ".root" not in input_:
            input_ = []
            self.input_roi.setText("No input provided!")
            self.input_roi_producer.setText("None")
            self.input_prod = None
        else:
            self.in_iom = IOManager([input_],None,0)
            self.in_iom.set_verbosity(0)

        # No ROOT file in the output, return and complain
        if ".root" not in output_:
            self.output_roi.setText("No valid output ROOT file!")
            self.output_roi_producer.setText("Give output ROI producer!")
            self.output_prod = None
        else:
            self.ou_iom = IOManager([],output_,1)
            self.ou_iom.set_verbosity(0)
    
    def captureROI(self):

        # Get the rois on screen
        self.user_rois[int(self.event.text())] = self.rois # not allowed to ``copy" qwidgets (w/ copy.deepcopy)
        
        larcv_rois = [self.roi2larcv(roisg) for roisg in self.rois]

        self.user_rois_larcv[int(self.event.text())] = larcv_rois # not allowed to copy qwidgets
        
        print "---"
        print self.user_rois
        print self.user_rois_larcv
        print "---"
        
    def addROI(self) :

        ww = 50
        hh = 50
        allow_resize=True
        
        if self.fixed_roi_box.isChecked():
            ww = int(self.fixed_w.text())
            hh = int(self.fixed_h.text())
            allow_resize=False

            
        if self.getCheckedPlanes() == False:
            return
        
        coords = [ [0,0,ww,hh] for _ in xrange(3)]
        
        roisg = ROISliderGroup(coords,self.checked_planes,3,store.colors,allow_resize)

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

        # Clear out the image ROIs
        self.clearROI()

        # Get the event number (this is TTree entry for the displayed image)
        event = int(self.event.text())
        
        #print "processing event: ",event," and self.user_rois.keys():",self.user_rois.keys()
        
        if event not in self.user_rois.keys():

            if self.in_iom is not None:
                if event < self.in_iom.get_n_entries():
                    self.in_iom.read_entry(event)
                    roiarray = self.in_iom.get_data(larcv.kProductROI,self.input_prod)
                    self.user_rois_larcv[event] = [roi for roi in roiarray.ROIArray()]

                    print "reloading ",self.user_rois_larcv[event]," from file"
                    
                    self.user_rois[event] = self.larcv2roi(self.user_rois_larcv[event])
                else:
                    return

            else:
                return
            
        self.rois = self.user_rois[event]

        for roisg in self.rois:
            for roi in roisg.rois:
                #print type(roi),roi
                self.plt.addItem(roi)
    
    # add widgets to self and return 
    def grid(self, enable):

        if enable == True:
            self.enabled = True
            
            self.addWidget(self.title, 0, 0)

            # in/out labels
            self.addWidget(self.input_label, 1, 0)
            self.addWidget(self.output_label, 2, 0)

            # actual input/output
            self.addWidget(self.input_roi, 1, 1)
            self.addWidget(self.output_roi, 2, 1)

            # producer label
            self.addWidget(self.input_prod_label, 1, 2)
            self.addWidget(self.output_prod_label, 2, 2)

            # actual producer input/output
            self.addWidget(self.input_roi_producer, 1, 3)
            self.addWidget(self.output_roi_producer, 2, 3)

            # big button to load files into docket
            self.addWidget(self.load_files, 3, 0,1,4)
            
            self.addWidget(self.add_roi, 1, 4)
            self.addWidget(self.remove_roi, 2, 4)
            self.addWidget(self.capture_roi, 3, 4)

            self.addWidget(self.clear_roi, 1, 5)
            self.addWidget(self.reset_roi, 2, 5)
            self.addWidget(self.store_roi, 3, 5)


            self.addWidget(self.roi_p1,1,6)
            self.addWidget(self.roi_p2,2,6)
            self.addWidget(self.roi_p3,3,6)

            self.addWidget(self.fixed_w_label,1,7)
            self.addWidget(self.fixed_h_label,2,7)

            self.addWidget(self.fixed_roi_box,0,8)
            self.addWidget(self.fixed_w,1,8)
            self.addWidget(self.fixed_h,2,8)
        
            

        else:

            self.enabled = False
            
            for i in reversed(range(self.count())):
                self.itemAt(i).widget().setParent(None)


        return self

    def roi2larcv(self,bboxes):

        #store all ROIs
        
        larcv_roi = larcv.ROI()

        for ix,bbox in enumerate(bboxes.rois):
            size = bbox.size()
            pos = bbox.pos()

            #print "ix",ix,"size",size,"pos",pos

            width,height,row_count,col_count,origin_x,origin_y = self.roi2imgcord(self.images.imgs[ix].meta(),size,pos)
            
            bbox_meta = larcv.ImageMeta(width,height,
                                        row_count,col_count,
                                        origin_x,origin_y,bbox.plane)

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
        
        #print "roi2img ROI: ",(width,height,row_count,col_count,origin_x,origin_y)
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

        #print "img2roi ROI: ",(x*dw_i,  y*dh_i,  w_b*dw_i,  h_b*dh_i)
        return (x*dw_i,  y*dh_i,  w_b*dw_i,  h_b*dh_i)

    
    def larcv2roi(self,rois):

        converted_rois = []

        for roi in rois:

            coords = [None for _ in xrange(3)]

            bbs = roi.BB()

            planes = []
            
            for ix,bbox in enumerate(bbs):

                pl = int(bbox.plane())
                
                coord = self.img2roicord(self.images.imgs[ix].meta(),bbox)
                
                coords[pl] = coord
                planes.append(pl)
                
            roisg = ROISliderGroup(coords,planes,len(coords),store.colors)
            
            converted_rois.append(roisg)
            

        return converted_rois
        
        
    def getCheckedPlanes(self):

        self.checked_planes = []
            
        if self.roi_p1.isChecked() == True:
            self.checked_planes.append(0)

        if self.roi_p2.isChecked() == True:
            self.checked_planes.append(1)

        if self.roi_p3.isChecked() == True:
            self.checked_planes.append(2)
            
        if len(self.checked_planes) == 0:
            return False

        return True
        
