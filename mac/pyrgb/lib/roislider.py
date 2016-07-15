from .. import pg

class ROISlider(pg.ROI):

    def __init__(self,xx,yy,pen='w',allow_resize=True,plane=0): # need to look up kwargs

        #scaleSnap and translateSnap == the box can only scanle and slide to integer values
        super(ROISlider, self).__init__(xx,yy,scaleSnap=True,translateSnap=True,pen=pen)

        # which dimension is it?
        self.plane = plane

        # add the toggles to the edge of the ROI slider
        if allow_resize == True:
            #self.addScaleHandle([0.0, 0.0], [0.5, 0.5])
            #self.addScaleHandle([0.0, 1.0], [0.5, 0.5])
            #self.addScaleHandle([1.0, 1.0], [0.5, 0.5])
            #self.addScaleHandle([1.0, 0.0], [0.5, 0.5])
            self.addScaleHandle([0.0, 0.0], [1.0, 1.0])
            self.addScaleHandle([0.0, 1.0], [1.0, 0.0])
            self.addScaleHandle([1.0, 1.0], [0.0, 0.0])
            self.addScaleHandle([1.0, 0.0], [0.0, 1.0])

    
class ROISliderGroup:

    def __init__(self,coords,planes,N,pencolors,allow_resize=True,func_setlabel=None):
        N = int(N)
        self.rois = []
        self.planes = planes
        self.use_same_times = False   # forces all plane ROIs to have the same time coordinates (Y-axis)

        # optional labels for U,V,Y: report back the positions of the boxes
        self.setROIlabel = func_setlabel

        for ix in self.planes:

            coord = coords[ix]
            x,y,w,h = coord
            
            roi = ROISlider([x,y], [w, h],pencolors[ix],allow_resize,ix)

            # can't use sigRegionChangeFinished since we encounter an infinite loop
            # upon progammatic change of ROI size
            #roi.sigRegionChangeFinished.connect(self.resizeROI)
            
            # huge python complaint!
            #roi.sigRegionChanged.connect(self.resizeROI)

            # works, but is annoying. Will just tell user what to do in the README
            #roi.sigRegionChangeStarted.connect(self.resizeROI)
            roi.sigRegionChangeFinished.connect(self.resizeROI)
            roi.sigRegionChangeFinished.connect(self.reportPositions)

            self.rois.append(roi)

    def useSameTimes(self):
        self.use_same_times = True

    def useDifferentTimes(self):
        self.use_same_times = False
            
    def resizeROI(self):
        if not self.use_same_times:
            sender = self.rois[0].sender()
            size = sender.size()
            for roi in self.rois:
                if roi == sender: continue
                roi.setSize(size, finish=False)
        else:
            #print "same-time resize. ROI set "%(self)," : "
            sender = self.rois[0].sender()
            sender_pos = sender.pos()
            sender_shape = sender.size()
            tstart = sender_pos[1]
            tend   = sender_pos[1]+sender_shape[1]
            #print "  ",sender
            #print "  ",
            for roi in self.rois:
                pos = roi.pos()
                roi.setPos( [pos[0],tstart], finish=False, update=False )
                s   = roi.size()
                roi.setSize( [s[0], sender_shape[1] ], finish=False, update=False )
                #print roi.pos(),
            #print
    
    def reportPositions(self):
        # this is a function pointer to the owner ROIToolkit class. 
        if self.setROIlabel is not None:
            self.setROIlabel( self.rois )
                
                
    # no real need yet for __iter__ and next()
                
