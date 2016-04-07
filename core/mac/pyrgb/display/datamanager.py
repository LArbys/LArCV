import numpy as np
import time

from ..lib.iomanager        import IOManager
from ..lib.compressed_image import CompressedImage

from .. import larcv

def get_max_size(imgs) :
    xb=99999
    yb=99999
    xt=0
    yt=0
    imxt = None
    imyt = None
    imxb = None
    imyb = None
    
    for img in imgs:
        if img.meta().tr().x > xt:
            xt = img.meta().tr().x
            imxt = img.meta()
        if img.meta().tr().y > yt:
            yt = img.meta().tr().y
            imyt = img.meta()
        if img.meta().bl().x < xb:
            xb = img.meta().bl().x
            imxb = img.meta()
        if img.meta().bl().y < yb:
            yb = img.meta().bl().y
            imyb = img.meta()

        # if img.meta().cols() > x:
        #     x = img.meta().cols()
        #     imx = img.meta()
        # if img.meta().rows() > y:
        #     y = img.meta().rows()
        #     imy = img.meta()

        x = xt - xb + imxb.bl().x
        y = yt - yb + imyb.bl().y
        
    return (x,y)

class DataManager(object):

    def __init__(self,rfiles):

        self.ROI_PRODUCER   ='event_roi'
        self.LR_IMG_PRODUCER='event_image'
        self.HR_IMG_PRODUCER='mcint00'

        self.iom = IOManager(rfiles)

        ### set of loaded images, we actually read them into
        ### memory with as_ndarray, but probably lose
        ### meta information
        
        self.loaded = {}
        
    def get_event_image(self,ii,imin,imax,lr=True) :

        #Load data in TChain
        self.iom.iom.read_entry(ii)

        imdata  = self.iom.iom.get_data( larcv.kProductImage2D, self.LR_IMG_PRODUCER )
        roidata = self.iom.iom.get_data( larcv.kProductROI    , self.ROI_PRODUCER    )

        imdata = imdata.Image2DArray()

        if imdata.size() == 0:
            return (None,None,None)
        
        cimage = CompressedImage(imdata)
        

        roi_v = roidata.ROIArray()

        #list of ROIs
        rois = []

        # loop over event ROIs
        for ix in xrange(roi_v.size()):
            #this ROI
            roi = roi_v[ix]

            if roi.BB().size() == 0: #there was no ROI continue...
                continue

            r = {}
                        
            r['type'] = roi.Type()
            r['bbox'] = []
            for iy in xrange(3):
                r['bbox'].append( roi.BB(iy) )
                
            rois.append(r)

        # self.loaded[(ii,imin,imax,lr)]  =  (b,rois,imgs)
        
        return ( cimage.treshold_mat(imin,imax), rois, imdata )
