import numpy as np
import time, re

from ..lib.iomanager        import IOManager

from ..lib.compressed_image   import CompressedImage
from ..lib.uncompressed_image import UnCompressedImage

from .. import ROOT

from .. import larcv



class DataManager(object):

    def __init__(self,rfiles):

        self.ROI_PRODUCER   ='event_roi'
        self.LR_IMG_PRODUCER='event_image'
        self.HR_IMG_PRODUCER='mcint00'
        
        # for rf in rfiles:
        #     ROOT.TFile.Open((rf
            
        
        self.iom = IOManager(rfiles)
        
        keys = [ key.GetName() for key in ROOT.gDirectory.GetListOfKeys() ]
        self.keys= {}

        for key in keys:
            key  = key.split("_")
            prod = key[0]

            if prod not in self.keys.keys():
                self.keys[ prod ] = []
                
            producer = ""

            for k in key[1:]:

                if k == "tree":
                    producer = producer[:-1]
                    break
                
                producer += k + "_"

            self.keys[ prod ].append(producer)
            
        ### set of loaded images, we actually read them into
        ### memory with as_ndarray, but probably lose
        ### meta information
        
        self.loaded = {}
        
    def get_event_image(self,ii,imin,imax,imgprod,roiprod,highres) :

        #Load data in TChain
        self.iom.iom.read_entry(ii)

        imdata, roidata, image = None, None, None

        if roiprod is not None:
            roidata = self.iom.iom.get_data( larcv.kProductROI , roiprod    )
            roidata = roidata.ROIArray()

        #Awkward true false
        if highres == False:
            imdata  = self.iom.iom.get_data( larcv.kProductImage2D, imgprod )
            imdata  = imdata.Image2DArray()
            if imdata.size() == 0 : return (None,None,None)
            image   = CompressedImage(imdata,roidata)
            
        else:
            imdata  = self.iom.iom.get_data( larcv.kProductImage2D, imgprod )
            imdata  = imdata.Image2DArray()
            if imdata.size() == 0 : return (None,None,None)
            image   = UnCompressedImage(imdata,roidata)

        if roiprod is None:
            return ( image.treshold_mat(imin,imax),
                     None,
                     imdata )
    
        return ( image.treshold_mat(imin,imax),
                 image.parse_rois(),
                 imdata )
