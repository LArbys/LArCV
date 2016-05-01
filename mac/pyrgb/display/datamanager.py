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
        
        self.iom = IOManager(argv)
        self.keys ={}
        # self.keys = { larcv.ProductName(larcv.kProductImage2D) : [img_producer,'segment_'+img_producer],
        #               larcv.ProductName(larcv.kProductROI)     : [roi_producer] }

        # get keys from rootfile
        for i in xrange(larcv.kProductUnknown):
            product = larcv.ProductName(i)

            self.keys[product] = []

            producers=self.iom.iom.producer_list(i)
            
            for p in producers:
                self.keys[product].append(p)
                    
        self.run    = -1
        self.subrun = -1
        self.event  = -1
        
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


        self.run    = imdata.run()
        self.subrun = imdata.subrun()
        self.event  = imdata.event()
        
        print "imdata.event_key() {}".format(imdata.event_key())

        imdata  = imdata.Image2DArray()

        print "imdata.size(): {}".format(imdata.size())
        
        if imdata.size() == 0 : return (None,None,None)
        image   = VicImage(imdata,roidata,planes)

        if roiprod is None:
            return ( image.treshold_mat(imin,imax),
                     None,
                     image.imgs )
    
        return ( image.treshold_mat(imin,imax),
                 image.parse_rois(),
                 image.imgs )
