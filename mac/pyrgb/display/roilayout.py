from .. import pg
from .. import QtGui, QtCore
from ..lib.roislider import ROISliderGroup
from ..lib import storage as store

class ROIToolLayout(QtGui.QGridLayout):

    def __init__(self,plt):

        super(ROIToolLayout, self).__init__()

        # Sliding ROI which we will do OpenCV manipulations on
        self.name = "ROIToolLayout"

        self.enabled = False
        self.cv2 = None
        self.imi = None
        self.overwrite = False
        self.transform = True

        self.title  = QtGui.QLabel("<b>ROI Tool</b>")
        self.input_roi  = QtGui.QLineEdit("(Optional) Input ROI filename")
        self.output_roi = QtGui.QLineEdit("(Required) Output ROI filename")

        self.add_roi = QtGui.QPushButton("Add ROI")
        self.remove_roi = QtGui.QPushButton("Remove ROI")

        self.add_roi.clicked.connect(self.addROI)
        self.remove_roi.clicked.connect(self.removeROI)

        self.rois = []

        self.plt = plt

    def addROI(self) :
        roisg = ROISliderGroup(30,30,3,store.colors)
        self.rois.append(roisg)
        for roi in roisg.rois: self.plt.addItem(roi)


    def removeROI(self) :
        for roi in self.rois[-1].rois: self.plt.removeItem(roi)

        rois = rois[:-1]
        
    # add widgets to self and return 
    def grid(self, enable):

        if enable == True:

            self.enabled = True
            self.addWidget(self.title, 0, 0)
            self.addWidget(self.input_roi, 1, 0)
            self.addWidget(self.output_roi, 2, 0)

            self.addWidget(self.add_roi, 1, 1)
            self.addWidget(self.remove_roi, 2, 1)

        else:
            for i in reversed(range(self.count())):
                self.itemAt(i).widget().setParent(None)

            self.enabled = False

        return self
