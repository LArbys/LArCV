from .. import pg

class ROISlider(pg.ROI):

    def __init__(self,xx,yy,pen='w'): # need to look up kwargs

        #scaleSnap and translateSnap == the box can only scanle and slide to integer values

        super(ROISlider, self).__init__(xx,yy,scaleSnap=True,translateSnap=True,pen=pen)

        # add the toggles to the edge of the ROI slider
        self.addScaleHandle([0.0, 0.0], [0.5, 0.5])
        self.addScaleHandle([0.0, 1.0], [0.5, 0.5])
        self.addScaleHandle([1.0, 1.0], [0.5, 0.5])
        self.addScaleHandle([1.0, 0.0], [0.5, 0.5])

    
class ROISliderGroup:

    def __init__(self,width,height,N,pencolors):
        N = int(N)
        self.rois = []

        for ix in xrange(N):
            roi = ROISlider([0, 0], [width, height],pencolors[ix])

            #can't use sigRegionChangeFinished since we encounter an infinite loop
            #upon progammatic change of ROI size
            #roi.sigRegionChangeFinished.connect(self.resizeROI)
            
            #huge python complaint
            #roi.sigRegionChanged.connect(self.resizeROI)

            #works but is annoying, will just tell user what to do
            roi.sigRegionChangeStarted.connect(self.resizeROI)

            self.rois.append(roi)

            
    def resizeROI(self):
        sender = self.rois[0].sender()
        size = sender.size()
        for roi in self.rois:
            if roi == sender: continue
            roi.setSize(size)
            
    # no real need yet for __iter__ and next()
