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
        """
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

            if not producer.startswith('op') and not producer.startswith('segment'):
                self.keys[ prod ].append(producer)
        """
        print self.keys

        ### set of loaded images, we actually read them into
        ### memory with as_ndarray, but probably lose
        ### meta information
        
        self.loaded = {}
        
    def get_event_image(self,ii,imin,imax,imgprod,roiprod,planes,highres) :

        #Load data in TChain
        self.iom.iom.read_entry(ii)

        imdata, roidata, image = None, None, None

        if roiprod is not None:
            print roiprod
            roidata = self.iom.iom.get_data(larcv.kProductROI,roiprod)
            roidata = roidata.ROIArray()

        imdata  = self.iom.iom.get_data(larcv.kProductImage2D,imgprod)
        imdata  = imdata.Image2DArray()
        if imdata.size() == 0 : return (None,None,None)
        image   = VicImage(imdata,roidata,planes)

        #Awkward true false
        #if highres == False:
        #    print imgprod
        #    imdata  = self.iom.iom.get_data(larcv.kProductImage2D,imgprod)
        #    imdata  = imdata.Image2DArray()
        #    if imdata.size() == 0 : return (None,None,None)
        #    image   = CompressedImage(imdata,roidata,planes)
            
        #else:
        #    print imgprod
        #    imdata  = self.iom.iom.get_data( larcv.kProductImage2D,imgprod)
        #    imdata  = imdata.Image2DArray()
        #    if imdata.size() == 0 : return (None,None,None)
        #    image   = UnCompressedImage(imdata,roidata,planes)

        if roiprod is None:
            return ( image.treshold_mat(imin,imax),
                     None,
                     image.imgs )
    
        return ( image.treshold_mat(imin,imax),
                 image.parse_rois(),
                 image.imgs )
