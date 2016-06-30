from .. import pg
from .. import QtGui, QtCore

class ROIToolLayout(QtGui.QGridLayout):

    def __init__(self):

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

    # add widgets to self and return 
    def grid(self, enable):

        if enable == True:
            self.enabled = True
            self.addWidget(self.title, 0, 0)
            self.addWidget(self.input_roi, 1, 0)
            self.addWidget(self.output_roi, 2, 0)
        else:
            for i in reversed(range(self.count())):
                self.itemAt(i).widget().setParent(None)

            self.enabled = False

        return self
