import numpy as np
import time

from ..lib.iomanager        import IOManager
from ..lib.compressed_image import CompressedImage

from .. import larcv


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
        
    def get_event_image(self,ii,imin,imax) :

        #Load data in TChain
        self.iom.iom.read_entry(ii)

        roidata = self.iom.iom.get_data( larcv.kProductROI    , self.ROI_PRODUCER    )
        roidata = roidata.ROIArray()

        imdata, image = None, None
        
        imdata  = self.iom.iom.get_data( larcv.kProductImage2D, self.LR_IMG_PRODUCER )
        imdata  = imdata.Image2DArray()

        if imdata.size() == 0:
            return (None,None,None)

        image   = CompressedImage(imdata,roidata)

        

        return ( image.treshold_mat(imin,imax),
                 image.parse_rois(),
                 imdata )
