import numpy as np
import time, re

from .. import ROOT
from .. import larcv

from iomanager import IOManager
from imagefactory import ImageFactory

class DataManager(object):

    def __init__(self,argv):
        
        self.iom = IOManager(argv)
        self.keys ={}

        self.IF = ImageFactory()
        
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
        
    def get_event_image(self,ii,imin,imax,imgprod,roiprod,planes) :

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
        
        imdata  = imdata.Image2DArray()
        if imdata.size() == 0 : return (None,None,None)

        image  = self.IF.get(imdata,roidata,planes,imgprod)

        if roiprod is None:
            return ( image.treshold_mat(imin,imax),
                     None,
                     image )
    
        return ( image.treshold_mat(imin,imax),
                 image.parse_rois(),
                 image )
    









