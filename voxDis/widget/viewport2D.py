
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy
import math
from larcv import larcv

colorScheme = [(
    {'ticks': [(1, (22, 30, 151, 255)),
               (0.791, (0, 181, 226, 255)),
               (0.645, (76, 140, 43, 255)),
               (0.47, (0, 206, 24, 255)),
               (0.33333, (254, 209, 65, 255)),
               (0, (255, 0, 0, 255))],
     'mode': 'rgb'})]
colorScheme.append(
    {'ticks': [(0, (22, 30, 151, 255)),
               (0.33333, (0, 181, 226, 255)),
               (0.47, (76, 140, 43, 255)),
               (0.645, (0, 206, 24, 255)),
               (0.791, (254, 209, 65, 255)),
               (1, (255, 0, 0, 255))],
     'mode': 'rgb'})
colorScheme.append(
    {'ticks': [(0, (22, 30, 151, 255)),
               (0.33333, (0, 181, 226, 255)),
               (0.47, (76, 140, 43, 255)),
               (0.645, (0, 206, 24, 255)),
               (0.791, (254, 209, 65, 255)),
               (1, (255, 0, 0, 255))],
     'mode': 'rgb'})


class viewport2D(QtGui.QWidget):

    def __init__(self, plane=-1):
        super(viewport2D, self).__init__()

        # add a view box, which is a widget that allows an image to be shown
        self._view = pg.GraphicsLayoutWidget()
        self._plot = self._view.addPlot(border=None)
        self._plot.getViewBox().invertY()

        # add an image item which handles drawing (and refreshing) the image
        self._item = pg.ImageItem(useOpenGL=True)

        self._plot.addItem(self._item)

        # connect the scene to click events, used to get wires
        self._view.scene().sigMouseClicked.connect(self.mouseClicked)

        # connect the views to mouse move events, used to update the info box
        # at the bottom
        self._view.scene().sigMouseMoved.connect(self.mouseMoved)

        # Save the plane:
        self._plane = plane

        # Set up the blank data:
        # self._blankData = np.ones((self._geometry.wRange(self._plane),self._geometry.tRange()))
        self._view.setBackground(None)

        # Meta is provided by Image2D, helps us map pixel to absolute location
        self._meta = None

        # each drawer contains its own color gradient and levels
        # this class can return a widget containing the right layout for
        # everything

        # Define some color collections:
        self._colorMap = colorScheme[self._plane]

        self._cmap = pg.GradientWidget(orientation='right')
        self._cmap.restoreState(self._colorMap)
        self._cmap.sigGradientChanged.connect(self.refreshGradient)
        self._cmap.resize(1, 1)

        # These boxes control the levels.
        self._upperLevel = QtGui.QLineEdit()
        self._lowerLevel = QtGui.QLineEdit()

        self._upperLevel.returnPressed.connect(self.levelChanged)
        self._lowerLevel.returnPressed.connect(self.levelChanged)

        self._lowerLevel.setText(str(-1))
        self._upperLevel.setText(str(1))

        self.setMaximumHeight(300)

        # Fix the maximum width of the widgets:
        self._upperLevel.setMaximumWidth(35)
        self._cmap.setMaximumWidth(25)
        self._lowerLevel.setMaximumWidth(35)

        self._layout = QtGui.QHBoxLayout()
        self._layout.addWidget(self._view)

        colors = QtGui.QVBoxLayout()
        colors.addWidget(self._upperLevel)
        colors.addWidget(self._cmap)
        colors.addWidget(self._lowerLevel)
        self._layout.addLayout(colors)
        self.setLayout(self._layout)
        self.show()


    def setData(self, image):
        if image is None:
            return
        self._meta = image.meta()
        
        # Generate ticks for this image:
        ticks_x = [[], []]
        for i in range(self._meta.cols()):
            val = self._meta.pos_x(i)
            # Most major:
            if self._meta.cols() % (i+1) == 10:
                ticks_x[0].append((i, str(val)))
            ticks_x[-1].append((i, str(val)))
        self._plot.getAxis('bottom').setTicks(ticks_x)

        ticks_y = [[], []]
        for i in range(self._meta.rows()):
            val = self._meta.pos_y(i)
            # Most major:
            if self._meta.rows() % (i+1) == 10:
                ticks_y[0].append((i, str(val)))
            ticks_y[-1].append((i, str(val)))
        self._plot.getAxis('left').setTicks(ticks_y)


        # self.plt_y = self.plt.getAxis('left')
        # self._plot.setRange(xRange=[self._meta.min_x(), self._meta.max_x()])

        AR = (self._meta.cols()) / (self._meta.rows())
        self._plot.setAspectLocked(True, ratio=AR)
        wrapper = larcv.as_ndarray(image)
        self._item.setImage(wrapper)
        self._item.setVisible(False)
        self._item.setVisible(True)
        # Make sure the levels are actually set:
        # self.levelChanged()

    def restoreDefaults(self):
        self._lowerLevel.setText(str(-1))
        self._upperLevel.setText(str(1))

        self._cmap.restoreState(self._colorMap)

    def mouseDrag(self):
        print "mouse was dragged"

    def getWidget(self):

        return self._widget, self._totalLayout

    def levelChanged(self):
        # First, get the current values of the levels:
        lowerLevel = int(self._lowerLevel.text())
        upperLevel = int(self._upperLevel.text())

        # set the levels as requested:
        levels = (lowerLevel, upperLevel)
        # next, set the levels in the geometry:

        # last, update the levels in the image:
        self._item.setLevels(levels)

    def refreshGradient(self):
        self._item.setLookupTable(self._cmap.getLookupTable(255))

    def mouseMoved(self, pos):
        # self.q = self._item.mapFromScene(pos)
        # self._lastPos = self.q
        # print self.q
        # if (pg.Qt.QT_LIB == 'PyQt4'):
        #     message = QtCore.QString()
        # else:
        #     message = str()

        # if type(message) != str:
        #     message.append("W: ")
        #     message.append(str(int(self.q.x())))
        # else:
        #     message += "W: "
        #     message += str(int(self.q.x()))

        # if type(message) != str:
        #     message.append(", T: ")
        #     message.append(str(int(self.q.y())))
        # else:
        #     message += ", T: "
        #     message += str(int(self.q.y()))

        # # print message
        # if self.q.x() > 0 and self.q.x() < self._geometry.wRange(self._plane):
        #     if self.q.y() > 0 and self.q.y() < self._geometry.tRange():
        #         self._statusBar.showMessage(message)
        pass

    def mouseClicked(self, event):
        pass
        # # Get the Mouse position and print it:
        # # print "Image position:", self.q.x()
        # # use this method to try drawing rectangles
        # # self.drawRect()
        # # pdi.plot()
        # # For this function, a click should get the wire that is
        # # being hovered over and draw it at the bottom
        # if event.modifiers() == QtCore.Qt.ShiftModifier:
        #     if event.pos() is not None:
        #         self.processPoint(self._lastPos)

        # #
        # wire = int(self._lastPos.x())
        # if self._item.image is not None:
        #     # get the data from the plot:
        #     data = self._item.image
        #     self._wireData = data[wire]
        #     self._wdf(self._wireData)
        #     # print "Plane: " + str(self._plane) + ", Wire: " + str(wire)
        #     # return self.plane,self.wire

        # # Make a request to draw the hits from this wire:
        # self.drawHitsRequested.emit(self._plane, wire)

    def connectStatusBar(self, _statusBar):
        self._statusBar = _statusBar

    def setRangeToMax(self):
        xR = (0, self._geometry.wRange(self._plane))
        yR = (0, self._geometry.tRange())
        self._plot.setRange(xRange=xR, yRange=yR, padding=0.000)

    def autoRange(self, xR, yR):
        self._plot.setRange(xRange=xR, yRange=yR, padding=0.000)

    def plane(self):
        return self._plane

    def drawPlane(self, image):
        self._item.setImage(image, autoLevels=False)
        self._item.setLookupTable(self._cmap.getLookupTable(255))
        self._cmap.setVisible(True)
        self._upperLevel.setVisible(True)
        self._lowerLevel.setVisible(True)
        self._item.setVisible(False)
        self._item.setVisible(True)
        # Make sure the levels are actually set:
        self.levelChanged()

    def drawBlank(self):
        self._item.clear()
        self._cmap.setVisible(False)
        self._upperLevel.setVisible(False)
        self._lowerLevel.setVisible(False)
