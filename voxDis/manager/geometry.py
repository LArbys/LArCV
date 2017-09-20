
import ROOT
import numpy as np

import os

class geoBase(object):

    """docstring for geometry"""

    def __init__(self):
        super(geoBase, self).__init__()
        self._nViews = 2
        self._tRange = 1600
        self._wRange = [240, 240]
        self._aspectRatio = 4
        self._time2Cm = 0.1
        self._wire2Cm = 0.4
        self._levels = [(-15, 15), (-10, 30)]
        self._name = "null"
        self._offset = [0, 0]
        self._halfwidth = 1.0
        self._halfheight = 1.0
        self._length = 1.0
        self._haslogo = False
        self._logo = None
        self._path = os.path.dirname(os.path.realpath(__file__))
        self._logopos = [0,0]
        self._logoscale = 1.0
        self._triggerOffset = 60
        self._readoutWindowSize = 2408
        self._planeOriginX = [-0.2, -0.6] 
        self._planeOriginXTicks = [-0.2/0.4, -0.6/0.4] 
        self._readoutPadding = 0
        self._timeOffsetTicks = 0
        self._timeOffsetCm = 0

    def halfwidth(self):
       return self._halfwidth

    def halfheight(self):
       return self._halfheight

    def length(self):
       return self._length

    def nViews(self):
        return self._nViews

    def tRange(self):
        return self._tRange

    def wRange(self, plane):
        return self._wRange[plane]

    def getLevels(self, plane):
        return self._levels[plane]

    def aspectRatio(self):
        return self._aspectRatio

    def getBlankData(self, plane):
        return np.ones((self._wRange[plane], self._tRange))

    def wire2cm(self):
        return self._wire2Cm

    def time2cm(self):
        return self._time2Cm

    def name(self):
        return self._name

    def offset(self, plane):
        return self._offset[plane]

    def hasLogo(self):
        return self._haslogo

    def logo(self):
        return self._logo

    def logoScale(self):
        return self._logoscale

    def logoPos(self):
        return self._logopos

    def readoutWindowSize(self):
        return self._readoutWindowSize

    def readoutPadding(self):
        return self._readoutPadding

    def triggerOffset(self):
        return self._triggerOffset
    
    def planeOriginX(self, plane):
        return self._planeOriginX[plane]

    def timeOffsetTicks(self, plane):
        return self._timeOffsetTicks
        # return self._timeOffsetTicks + self._planeOriginXTicks[plane]

    def timeOffsetCm(self, plane):
        return self._timeOffsetCm

class geometry(geoBase):

    def __init__(self):
        super(geometry, self).__init__()
        self._defaultColorScheme = []

    def configure(self):
        # self._halfwidth = larutil.Geometry.GetME().DetHalfWidth()
        # self._halfheight = larutil.Geometry.GetME().DetHalfHeight()
        # self._length = larutil.Geometry.GetME().DetLength()      
        # self._time2Cm = larutil.GeometryHelper.GetME().TimeToCm()
        # self._wire2Cm = larutil.GeometryHelper.GetME().WireToCm()
        # self._aspectRatio = self._wire2Cm / self._time2Cm
        # self._nViews = larutil.Geometry.GetME().Nviews()
        # self._tRange = larutil.DetectorProperties.GetME().ReadOutWindowSize()
        self._wRange = []
        self._offset = []
        # for v in range(0, self._nViews):
            # self._wRange.append(larutil.Geometry.GetME().Nwires(v))

    def colorMap(self, plane):
        return self._defaultColorScheme[plane]


class microboone(geometry):

    def __init__(self):
        # Try to get the values from the geometry file.  Configure for microboone
        # and then call the base class __init__
        super(microboone, self).__init__()
        self._halfwidth = 128.175
        self._halfheight = 116.5
        self._length = 1036.8
        self._time2Cm = 1
        self._wire2Cm = 1
        self._aspectRatio = self._wire2Cm / self._time2Cm
        self._nViews = 3
        self._tRange = 3072
        # larutil.LArUtilManager.Reconfigure(galleryfmwk.geo.kMicroBooNE)
        self.configure()
        self._levels = [(-100, 10), (-10, 100), (-10, 200)]
        # self._colorScheme =
        # self._time2Cm = 0.05515
        self._pedestals = [2000, 2000, 440]
        self._name = "uboone"
        self._logo = self._path + "/logos/uboone_logo_bw_transparent.png"
        self._logoRatio = 1.0
        self._haslogo = True
        self._logopos = [1250,10]
        self._logoscale = 0.1
        self._tRange = 9600
        self._triggerOffset = 3200
        self._readoutWindowSize = 9600
        self._planeOriginX = [0.0, -0.3, -0.6] 
        self._planeOriginXTicks = [0.0, -0.3/self._time2Cm, -0.6/self._time2Cm] 

        self._defaultColorScheme = [(
            {'ticks': [(1, (22, 30, 151, 255)),
                       (0.791, (0, 181, 226, 255)),
                       (0.645, (76, 140, 43, 255)),
                       (0.47, (0, 206, 24, 255)),
                       (0.33333, (254, 209, 65, 255)),
                       (0, (255, 0, 0, 255))],
             'mode': 'rgb'})]
        self._defaultColorScheme.append(
            {'ticks': [(0, (22, 30, 151, 255)),
                       (0.33333, (0, 181, 226, 255)),
                       (0.47, (76, 140, 43, 255)),
                       (0.645, (0, 206, 24, 255)),
                       (0.791, (254, 209, 65, 255)),
                       (1, (255, 0, 0, 255))],
             'mode': 'rgb'})
        self._defaultColorScheme.append(
            {'ticks': [(0, (22, 30, 151, 255)),
                       (0.33333, (0, 181, 226, 255)),
                       (0.47, (76, 140, 43, 255)),
                       (0.645, (0, 206, 24, 255)),
                       (0.791, (254, 209, 65, 255)),
                       (1, (255, 0, 0, 255))],
             'mode': 'rgb'})

        self._offset = []
        for v in range(0, self._nViews):
            # Set up the correct drift time offset.
            # Offset is returned in terms of centimeters.

            self._offset.append(
                self.triggerOffset()
                * self.time2cm()
                - self.planeOriginX(v) )


class microboonetruncated(microboone):

    def __init__(self):
        super(microboonetruncated, self).__init__()

        # The truncated readouts change the trigger offset and 
        self._tRange = 9600
        self._triggerOffset = 3200
        self._planeOriginX = [0.3, -0.3, -0.6] 
        self._planeOriginXTicks = [0.3/self.time2cm(), -0.3/self.time2cm(), -0.6/self.time2cm()] 
        self._readoutWindowSize = 9600
        self._readoutPadding = 2400
        self._offset = []
        self._timeOffsetTicks = 2400
        self._timeOffsetCm = 2400 * self._time2Cm
        for v in range(0, self._nViews):
            # Set up the correct drift time offset.
            # Offset is returned in terms of centimeters.

            self._offset.append(
                self.triggerOffset()
                * self.time2cm()
                - self.planeOriginX(v) )

class argoneut(geometry):

    def __init__(self):
        # Try to get the values from the geometry file.  Configure for microboone
        # and then call the base class __init__
        super(argoneut, self).__init__()
        # larutil.LArUtilManager.Reconfigure(galleryfmwk.geo.kArgoNeuT)
        self.configure()
        self._levels = [(-15, 60), (-25, 100)]
        self._pedestals = [0, 0]
        self._name = "argoneut"
        self._offset = []

        self._tRange = 1800
        self._triggerOffset = 60
        self._planeOriginX = [-0.2, -0.6] 
        self._readoutWindowSize = 2048

        self._offset = []

        for v in range(0, self._nViews):
            # Set up the correct drift time offset.
            # Offset is returned in terms of centimeters.

            self._offset.append(
                self.triggerOffset()
                * self.time2cm()
                - self.planeOriginX(v) )

        self._defaultColorScheme = [
            {'ticks': [(0.0,  (30,  30, 255, 255)),
                       (0.32,  (0,  255, 255, 255)),
                       (0.8,  (0,  255, 0,   255)),
                       (1,    (255,  0, 0,   255))],
             'mode': 'rgb'}]
        self._defaultColorScheme.append(
            {'ticks': [(0.0,  (30,  30, 255, 255)),
                       (0.4,  (0,  255, 255, 255)),
                       (0.8,  (0,  255, 0,   255)),
                       (1,    (255,  0, 0,   255))],
             'mode': 'rgb'})
        # self._offset = [1.7226813611, 2.4226813611]


class lariat(geometry):

    def __init__(self):
        # Try to get the values from the geometry file.  Configure for microboone
        # and then call the base class __init__
        super(lariat, self).__init__()
        # larutil.LArUtilManager.Reconfigure(galleryfmwk.geo.kArgoNeuT)
        self.configure()
        # lariat has a different number of time ticks
        # fix it directly:
        self._tRange = 3072
        self._levels = [(-40, 160), (-80, 320)]
        self._pedestals = [0, 0]
        self._name = "lariat"
        # Get the logo too!
        self._logo = self._path + "/logos/LArIAT_simple_outline.png"
        self._haslogo = True
        self._logopos = [1200,10]
        self._logoscale = 0.2
        # Make default color schemes here:
        self._defaultColorScheme = [
            {'ticks': [(0, (30, 30, 255, 255)),
                       (0.33333, (0, 255, 255, 255)), 
                       (0.66666, (255,255,100,255)), 
                       (1, (255, 0, 0, 255))], 
             'mode': 'rgb'}]
        self._defaultColorScheme.append(
            {'ticks': [(0, (30, 30, 255, 255)),
                       (0.33333, (0, 255, 255, 255)), 
                       (0.66666, (255,255,100,255)), 
                       (1, (255, 0, 0, 255))], 
             'mode': 'rgb'})

