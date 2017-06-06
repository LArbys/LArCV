import pyqtgraph
from pyqtgraph.Qt import QtGui, QtCore

from widget import viewport2D, viewport3D


class voxDisGui(QtGui.QWidget):
    """docstring for voxDisGui"""

    def __init__(self, geometry, manager):
        super(voxDisGui, self).__init__()
        self._geometry = geometry
        self._manager = manager
        self._manager.redrawRequested.connect(self.redraw)
        self.initUI()
        self.redraw()

    def quit(self):
        QtCore.QCoreApplication.instance().quit()

    def update(self):
        eventLabel = "Ev: " + str(self._manager.event())
        self._eventLabel.setText(eventLabel)
        runLabel = "Run: " + str(self._manager.run())
        self._runLabel.setText(runLabel)
        subrunLabel = "Subrun: " + str(self._manager.subrun())
        self._subrunLabel.setText(subrunLabel)

    # redraw will clear the image and redraw by getting the data again from
    # the manager
    def redraw(self):
        self.clear()

        #Redraw the 3D data:
        self._3d_display.setData(self._manager.get_image_3D())

        #Redraw the 2D data:
        for _viewport2D in self._2d_displays:
            _viewport2D.setData(self._manager.get_image_2D(_viewport2D.plane()))

        pass

    # clear will remove all the data from the viewer and leave it blank
    def clear(self):
        #Clear the 3D Data:

        #Clear the 2D Data:

        pass

    # Ask the manager to go to the specified event
    def goToEventWorker(self):
        try:
            entry = int(self._entry.text())
        except:
            print("Error, must enter an integer")
            self._entry.setText(str(self._manager.current_entry()))
            return
        self._manager.goToEvent(entry)

    # Ask the manager to go to the next event
    def nextWorker(self):
        self._manager.next()

    # Ask the manager to go to the previous event
    def prevWorker(self):
        self._manager.previous()

    def selectFileWorker(self):
        pass

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_N:
            self.nextWorker()
            return
        if e.key() == QtCore.Qt.Key_P:
            self.prevWorker()
            return
        if e.key() == QtCore.Qt.Key_C:
            # print "C was pressed"
            if e.modifiers() and QtCore.Qt.ControlModifier:
                self.quit()
                return

    # This function prepares the buttons such as prev, next, etc and returns a
    # layout
    def getEventControlButtons(self):

        # This is a box to allow users to enter an event (larlite numbering)
        self._goToLabel = QtGui.QLabel("Go to: ")
        self._entry = QtGui.QLineEdit()
        self._entry.setToolTip(
            "Enter an event to skip to that event (larcv numbering)")
        self._entry.returnPressed.connect(self.goToEventWorker)
        # These labels display current events
        self._runLabel = QtGui.QLabel("Run: 0")
        self._eventLabel = QtGui.QLabel("Ev.: 0")
        self._subrunLabel = QtGui.QLabel("Subrun: 0")

        # Jump to the next event
        self._nextButton = QtGui.QPushButton("Next")
        # self._nextButton.setStyleSheet("background-color: red")
        self._nextButton.clicked.connect(self.nextWorker)
        self._nextButton.setToolTip("Move to the next event.")
        # Go to the previous event
        self._prevButton = QtGui.QPushButton("Previous")
        self._prevButton.clicked.connect(self.prevWorker)
        self._prevButton.setToolTip("Move to the previous event.")
        # Select a file to use
        self._fileSelectButton = QtGui.QPushButton("Select File")
        self._fileSelectButton.clicked.connect(self.selectFileWorker)

        # pack the buttons into a box
        self._eventControlBox = QtGui.QVBoxLayout()

        # Make a horiztontal box for the event entry and label:
        self._eventGrid = QtGui.QHBoxLayout()
        self._eventGrid.addWidget(self._goToLabel)
        self._eventGrid.addWidget(self._entry)
        # Another horizontal box for the run/subrun
        # self._runSubRunGrid = QtGui.QHBoxLayout()
        # self._runSubRunGrid.addWidget(self._eventLabel)
        # self._runSubRunGrid.addWidget(self._runLabel)
        # Pack it all together
        self._eventControlBox.addLayout(self._eventGrid)
        self._eventControlBox.addWidget(self._eventLabel)
        self._eventControlBox.addWidget(self._runLabel)
        self._eventControlBox.addWidget(self._subrunLabel)
        self._eventControlBox.addWidget(self._nextButton)
        self._eventControlBox.addWidget(self._prevButton)
        self._eventControlBox.addWidget(self._fileSelectButton)

        return self._eventControlBox

    def initUI(self):

        # Set up the widgets:
        self.displayWidget = QtGui.QWidget()
        self._displayLayout = QtGui.QVBoxLayout()

        # Initialize a 3D display:
        self._3d_display = viewport3D(self._geometry)
        self._3d_display.quitRequested.connect(self.quit)

        # Initialize an array of 2D displays
        self._2d_displays = []
        self._2d_layout = QtGui.QHBoxLayout()
        for plane in range(self._geometry.nViews()):
            self._2d_displays.append(viewport2D(plane))
            self._2d_layout.addWidget(self._2d_displays[-1])

        # Put the 3D and 2D displays into the the layout
        self._displayLayout.addWidget(self._3d_display)
        self._displayLayout.addLayout(self._2d_layout)

        self.displayWidget.setLayout(self._displayLayout)

        # Get all of the widgets:
        # self.eastWidget  = self.getEastLayout()
        self.westWidget = self.getWestLayout()
        self.southWidget = self.getSouthWidget()

        # self._view_manager.connectStatusBar(self._statusBar)

        # Put the layout together

        self.master = QtGui.QVBoxLayout()
        self.slave = QtGui.QHBoxLayout()
        self.slave.addWidget(self.westWidget)
        # self.slave.addWidget(self.eastWidget)
        self.slave.addWidget(self.displayWidget)
        self.master.addLayout(self.slave)
        self.master.addWidget(self.southWidget)

        self.setLayout(self.master)

        # ask the view manager to draw the planes:
        # self._view_manager.drawPlanes(self._manager)

        self.setGeometry(0, 0, 2400, 1600)
        self.setWindowTitle('Event Display')
        self.setFocus()
        self.show()
        # self._view_manager.setRangeToMax()

    def screenCapture(self):
        print("Screen Capture!")
        dialog = QtGui.QFileDialog()
        r = self._manager.current_run()
        e = self._manager.current_event()
        s = self._manager.current_subrun()
        name = "voxel_" + self._geometry.name() + "_R" + str(r)
        name = name + "_S" + str(s)
        name = name + "_E" + str(e) + ".png"
        f = dialog.getSaveFileName(self, "Save File", name,
                                   "PNG (*.png);;JPG (*.jpg);;All Files (*)")

        try:
            pixmapImage = QtGui.QPixmap.grabWidget(self)
            pixmapImage.save(f, "PNG")
        except:
            pixmapImage = super(gui, self).grab()
            pixmapImage.save(f[0], "PNG")

    # This function prepares the quit buttons layout and returns it
    def getQuitButton(self):
        self._quitButton = QtGui.QPushButton("Quit")
        self._quitButton.setToolTip("Close the viewer.")
        self._quitButton.clicked.connect(self.quit)
        return self._quitButton

    # This function combines the control button layouts, range layouts, and
    # quit button
    def getWestLayout(self):

        event_control = self.getEventControlButtons()
        # draw_control = self.getDrawingControlButtons()

        self._westLayout = QtGui.QVBoxLayout()
        self._westLayout.addLayout(event_control)
        self._westLayout.addStretch(1)
        # self._westLayout.addLayout(draw_control)
        # self._westLayout.addStretch(1)

        self._westLayout.addStretch(1)
        self._westWidget = QtGui.QWidget()
        self._westWidget.setLayout(self._westLayout)
        self._westWidget.setMaximumWidth(150)
        self._westWidget.setMinimumWidth(100)
        return self._westWidget

    def getSouthWidget(self):
        # This layout contains the status bar, message bar, and the capture
        # screen buttons

        # The screen capture button:
        self._screenCaptureButton = QtGui.QPushButton("Capture Screen")
        self._screenCaptureButton.setToolTip(
            "Capture the entire screen to file")
        self._screenCaptureButton.clicked.connect(self.screenCapture)
        self._southWidget = QtGui.QWidget()
        self._southLayout = QtGui.QHBoxLayout()
        # Add a status bar
        self._statusBar = QtGui.QStatusBar()
        self._statusBar.showMessage("Test message")

        # And the quit button:
        quit_control = self.getQuitButton()

        self._southLayout.addWidget(quit_control)
        self._southLayout.addWidget(self._statusBar)
        self._southLayout.addWidget(self._screenCaptureButton)
        self._southWidget.setLayout(self._southLayout)

        return self._southWidget
