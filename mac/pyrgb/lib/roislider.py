from .. import pg

class ROISlider(pg.ROI):

    def __init__(self,xx,yy,pen='w',allow_resize=True,plane=0): # need to look up kwargs

        #scaleSnap and translateSnap == the box can only scanle and slide to integer values
        super(ROISlider, self).__init__(xx,yy,scaleSnap=True,translateSnap=True,pen=pen)

        # which dimension is it?
        self.plane = plane

        # add the toggles to the edge of the ROI slider
        if allow_resize == True:
            self.addScaleHandle([0.0, 0.0], [0.5, 0.5])
            self.addScaleHandle([0.0, 1.0], [0.5, 0.5])
            self.addScaleHandle([1.0, 1.0], [0.5, 0.5])
            self.addScaleHandle([1.0, 0.0], [0.5, 0.5])

    
class ROISliderGroup:

    def __init__(self,coords,planes,N,pencolors,allow_resize=True):
        N = int(N)
        self.rois = []
        self.planes = planes
    
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
            roi.sigRegionChangeStarted.connect(self.resizeROI)

            self.rois.append(roi)

            
    def resizeROI(self):
        sender = self.rois[0].sender()
        size = sender.size()
        for roi in self.rois:
            if roi == sender: continue
            roi.setSize(size)
            
    # no real need yet for __iter__ and next()
