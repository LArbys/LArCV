from .. import pg

class ROISlider(pg.ROI):

    def __init__(self,xx,yy): # need to look up kwargs
        #scaleSnap and translateSnap == the box can only scanle and slide to integer values
        super(ROISlider, self).__init__(xx,yy,scaleSnap=True,translateSnap=True)

        # add the toggles to the edge of the ROI slider
        self.addScaleHandle([0.0, 0.0], [0.5, 0.5])
        self.addScaleHandle([0.0, 1.0], [0.5, 0.5])
        self.addScaleHandle([1.0, 1.0], [0.5, 0.5])
        self.addScaleHandle([1.0, 0.0], [0.5, 0.5])

    
