from .. import pg


class ROISlider(pg.ROI):

    def __init__(self,xx,yy): # need to look up kwargs
        super(ROISlider, self).__init__(xx,yy,scaleSnap=True,translateSnap=True)

        #add sliders
        self.addScaleHandle([0.0, 0.0], [0.5, 0.5])
        self.addScaleHandle([0.0, 1.0], [0.5, 0.5])
        self.addScaleHandle([1.0, 1.0], [0.5, 0.5])
        self.addScaleHandle([1.0, 0.0], [0.5, 0.5])

    