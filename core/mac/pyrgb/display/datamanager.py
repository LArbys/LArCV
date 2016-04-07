from ROOT import TChain, larcv
import numpy as np
import time

class DataManager(object):
    def __init__(self,rfile):

        ROI_PRODUCER='event_roi'
        IMG_PRODUCER='event_image'

        self.img_tree_name='image2d_%s_tree' % IMG_PRODUCER
        self.img_br_name='image2d_%s_branch' % IMG_PRODUCER
        
        self.img_ch = TChain(self.img_tree_name)
        self.img_ch.AddFile(rfile)
        self.img_ch.GetEntry(0)
        
        self.roi_tree_name='partroi_%s_tree' % ROI_PRODUCER
        self.roi_br_name='partroi_%s_branch' % ROI_PRODUCER

        self.roi_ch = TChain(self.roi_tree_name)
        self.roi_ch.AddFile(rfile)
        self.roi_ch.GetEntry(0)

        self.co = { 0 : 'r', 1 : 'g' , 2 : 'b' }


        self.loaded = {}
        
    def get_event_image(self,ii,imin,imax) :

        if (ii,imin,imax) in self.loaded.keys():
            print "\t>> Already loaded this image return it\n"
            return self.loaded[(ii,imin,imax)] 
        
        self.img_ch.GetEntry(ii)
        img_br=None
        exec('img_br=self.img_ch.%s' % self.img_br_name)

        img_v = img_br.Image2DArray()

        imgs      = [ img_v[i] for i in xrange(img_v.size()) ]
        img_array = [ larcv.as_ndarray(img) for img in imgs ]

        if ( len(img_array) == 0 ) :
            return (None,None,None)
        
        b = np.zeros(list(img_array[0].shape) + [3])

        for ix,img in enumerate(img_array):
            img[img < imin] = 0
            img[img > imax] = imax

            img = img[:,::-1]
            
            b[:,:,ix] = img


        b[:,:,0][ b[:,:,1] > 0.0 ] = 0.0
        b[:,:,0][ b[:,:,2] > 0.0 ] = 0.0

        b[:,:,1][ b[:,:,2] > 0.0 ] = 0.0

        #event ROIs
        self.roi_ch.GetEntry(ii)
        roi_br=None
        exec('roi_br=self.roi_ch.%s' % self.roi_br_name)

        roi_v = roi_br.ROIArray()

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

        self.loaded[(ii,imin,imax)]  =  (b,rois,imgs)
        return (b,rois,imgs)
