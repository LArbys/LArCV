
try:
    import pyqtgraph.opengl as gl
except:
    print "Error, must have open gl to use this viewer."
    exit(-1)

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy
import math

red = numpy.asarray((1.0, 0.0, 0.0, 0.75))
white = numpy.asarray((1.0, 1.0, 1.0, 0.75))

colorMap = {'ticks': [(1, (22, 30, 151, 255)),
                      (0.791, (0, 181, 226, 255)),
                      (0.645, (76, 140, 43, 255)),
                      (0.47, (0, 206, 24, 255)),
                      (0.33333, (254, 209, 65, 255)),
                      (0, (255, 255, 255, 128))],
            'mode': 'rgb'}

class connectedGLViewWidget(gl.GLViewWidget):

    keyPressSignal = QtCore.pyqtSignal(QtGui.QKeyEvent)

    def __init__(self):
        super(connectedGLViewWidget, self).__init__()


    def keyPressEvent(self, event):
        if event.modifiers():
            self.keyPressSignal.emit(event)
        else:
            super(connectedGLViewWidget, self).keyPressEvent(event)

class viewport3D(QtGui.QWidget):

    quitRequested = QtCore.pyqtSignal()
    keyPressSignal = QtCore.pyqtSignal(QtGui.QKeyEvent)

    def __init__(self, geometry):
        super(viewport3D, self).__init__()
        # add a view box, which is a widget that allows an image to be shown
        # add an image item which handles drawing (and refreshing) the image
        self._view = connectedGLViewWidget()
        self._view.keyPressSignal.connect(self.keyPressEvent)
        self._view.show()
        # self._view.setBackgroundColor((50, 50, 50, 255))
        self._geometry = geometry

        # The data storage names:
        self._points = numpy.ndarray((0))
        self._values = numpy.ndarray((0))
        self._color = numpy.ndarray((0))
        self._glPointsCollection = None

        # Define some color collections:

        self._cmap = pg.GradientWidget(orientation='right')
        self._cmap.restoreState(colorMap)
        self._cmap.sigGradientChanged.connect(self.refreshGradient)
        self._cmap.resize(1, 1)

        self._lookupTable = self._cmap.getLookupTable(255, alpha=0.75)

        # These boxes control the levels.
        self._upperLevel = QtGui.QLineEdit()
        self._lowerLevel = QtGui.QLineEdit()

        self._upperLevel.returnPressed.connect(self.levelChanged)
        self._lowerLevel.returnPressed.connect(self.levelChanged)

        self._lowerLevel.setText(str(0.0))
        self._upperLevel.setText(str(2.0))

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



        # # self.pan(0,0,self._geometry.length())
        # print "Finish"

    def setData(self, voxels):
        if voxels is None:
            return
        self._meta = voxels.GetVoxelMeta()

        _len_x = self._meta.MaxX() - self._meta.MinX()
        _len_y = self._meta.MaxY() - self._meta.MinY()
        _len_z = self._meta.MaxZ() - self._meta.MinZ()

        # This section prepares the 3D environment:
        # Add an axis orientation item:
        self._axis = gl.GLAxisItem()
        self._axis.setSize(x=0.25*_len_x, y=0.25*_len_y, z=0.25*_len_z)
        self._view.addItem(self._axis)

        # Add a set of grids along x, y, z:
        self._xy_grid = gl.GLGridItem()
        self._xy_grid.setSize(x=_len_x, y=_len_y, z=0.0)
        self._xy_grid.translate(_len_x*0.5, _len_y * 0.5, 0.0)
        # self._xy_grid.setSize(x=_len_x, y=_len_y, z=0.0)
        # self._x_grid.setSize(x=self._meta.MaxX(), y=self._meta.MaxY(), z=self._meta.MaxZ())
        self._yz_grid = gl.GLGridItem()
        self._yz_grid.setSize(x=_len_z, y=_len_y)
        self._yz_grid.rotate(-90, 0, 1, 0)
        self._yz_grid.translate(0, _len_y*0.5, _len_z*0.5)
        self._xz_grid = gl.GLGridItem()
        self._xz_grid.setSize(x=_len_x, y=_len_z)
        self._xz_grid.rotate(90, 1, 0, 0)
        self._xz_grid.translate(_len_x*0.5, 0, _len_z*0.5)

        self._view.addItem(self._xy_grid)
        self._view.addItem(self._yz_grid)
        self._view.addItem(self._xz_grid)

        # Move the center of the camera to the center of the view:
        self._view.pan(_len_x*0.5, _len_y * 0.5, _len_z*0.5)

        # Dummy test, use spheres:
        self._points = numpy.ndarray((voxels.GetVoxelSet().size(), 3))
        self._color = numpy.ndarray((voxels.GetVoxelSet().size(), 4))
        self._values = numpy.ndarray((voxels.GetVoxelSet().size()))

        # # This section draws voxels onto the environment:
        i = 0
        for voxel in voxels.GetVoxelSet():
            _pos = self._meta.Position(voxel.ID())
            self._points[i][0] = _pos[0] - self._meta.MinX()
            self._points[i][1] = _pos[1] - self._meta.MinY()
            self._points[i][2] = _pos[2] - self._meta.MinZ()
            self._values[i] = voxel.Value()
            i += 1
            # print voxel.Value()

        self.setColors()

        self.redrawPoints()

    def setColors(self):
        _max = float(self._upperLevel.text())
        _min = float(self._lowerLevel.text())

        for i in xrange(len(self._values)):
            if self._values[i] >= _max:
                # print "Max " + str(self._values[i])
                self._color[i] = self._lookupTable[-1]
            elif self._values[i] <= _min:
                # print "Min "  + str(self._values[i])
                self._color[i] = self._lookupTable[0]
            else:
                index = 255*(self._values[i] - _min) / (_max - _min)
                self._color[i] = self._lookupTable[int(index)]
            # print self._color[i]

    def redrawPoints(self):
        if self._glPointsCollection is not None:
            self._view.removeItem(self._glPointsCollection)

        self._glPointsCollection = gl.GLScatterPlotItem(pos=self._points,
            size=10, color=self._color)
        self._view.addItem(self._glPointsCollection)

    def refreshGradient(self):
        self._lookupTable = (1./255)*self._cmap.getLookupTable(255, alpha=0.75)
        if len(self._points) == 0:
            return
        else:
            self.setColors()
            self.redrawPoints()


    def levelChanged(self):
        if len(self._points) == 0:
            return
        else:
            self.setColors()
            self.redrawPoints()

    def getColor(self, val):
        print self._cmap.getLookupTable(255)
        exit()
        if val <= 10.:
            return white
        if val >= 100.:
            return red
        else:
            # Change from white to red over a scale of 100
            return red
            return

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_C:
            # print "C was pressed"
            if e.modifiers() and QtCore.Qt.ControlModifier:
                self.quitRequested.emit()
                return
        elif e.modifiers():
            if QtCore.Qt.ShiftModifier:
                if e.key() == QtCore.Qt.Key_Up:
                    # This is supposed to pan upwards in the view
                    self._view.pan(0, 20, 0, True)
                if e.key() == QtCore.Qt.Key_Down:
                    self._view.pan(0, -20, 0, True)
                if e.key() == QtCore.Qt.Key_Left:
                    self._view.pan(20, 0, 0, True)
                if e.key() == QtCore.Qt.Key_Right:
                    self._view.pan(-20, 0, 0, True)


        # Pass this signal to the main gui, too
        self.keyPressSignal.emit(e)

    def setCenter(self, center):
        if len(center) != 3:
            return
        cVec = QtGui.QVector3D(center[0], center[1], center[2])
        self.opts['center'] = cVec
        self.update()

    def worldCenter(self):
        return self.opts['center']

    def getAzimuth(self):
        return self.opts['azimuth']

    def getElevation(self):
        return self.opts['elevation']

    def setCameraPos(self, pos):
        # calling set camera with pos doesn't actually do anything.  Convert
        # spherical coordinates:
        if pos is not None:
            # Convert to relative coordinates to always leave the world center
            # as the center point
            worldCenter = self.opts['center']
            # Check the type:
            if type(worldCenter) is QtGui.QVector3D:
                X = pos[0] - worldCenter.x()
                Y = pos[1] - worldCenter.y()
                Z = pos[2] - worldCenter.z()
            else:
                X = pos[0] - worldCenter[0]
                Y = pos[1] - worldCenter[1]
                Z = pos[2] - worldCenter[2]

            distance = X**2 + Y**2 + Z**2
            distance = math.sqrt(distance)
            if X != 0:
                azimuth = math.atan2(Y, X)
            else:
                azimuth = math.pi
                if Y < 0:
                    azimuth = -1 * azimuth
            if distance != 0:
                elevation = math.asin(Z / distance)
            else:
                elevation = math.copysign(Z)
            azimuth *= 180./math.pi
            elevation *= 180./math.pi
            self.setCameraPosition(
                distance=distance, elevation=elevation, azimuth=azimuth)