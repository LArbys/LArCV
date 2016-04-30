import numpy as np
import time, re

from ..lib.iomanager        import IOManager

from ..lib.compressed_image   import CompressedImage
from ..lib.uncompressed_image import UnCompressedImage
from ..lib.vic_image          import VicImage
from .. import ROOT
from .. import larcv

class DataManager(object):

    def __init__(self,argv):

        if len(argv) < 3:
            print '\033[93mERROR\033[00m requires at least 3 arguments: image_producer roi_producer file1 [file2 file3 ...]'
            raise Exception

        img_producer = argv[0]
        roi_producer = argv[1]
        self.iom = IOManager(argv[2:])

        self.keys = { larcv.ProductName(larcv.kProductImage2D) : [img_producer,'segment_'+img_producer],
                      larcv.ProductName(larcv.kProductROI)     : [roi_producer] }

        #get keys from rootfile

        # set of loaded images, we actually read them into
        # memory with as_ndarray, but probably lose
        # meta information
        
        self.loaded = {}
        
    def get_event_image(self,ii,imin,imax,imgprod,roiprod,planes,highres) :

        #Load data in TChain
        self.iom.iom.read_entry(ii)

        imdata, roidata, image = None, None, None

        if roiprod == "None":
            roiprod = None
            
        if roiprod is not None:
            roidata = self.iom.iom.get_data(larcv.kProductROI,roiprod)
            roidata = roidata.ROIArray()

        imdata  = self.iom.iom.get_data(larcv.kProductImage2D,imgprod)
        print imdata.event_key()
        imdata  = imdata.Image2DArray()
        if imdata.size() == 0 : return (None,None,None)
        image   = VicImage(imdata,roidata,planes)

        if roiprod is None:
            return ( image.treshold_mat(imin,imax),
                     None,
                     image.imgs )
    
        return ( image.treshold_mat(imin,imax),
                 image.parse_rois(),
                 image.imgs )
