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

    def get_event_image(self,ii) :
        tic = time.clock()

        self.img_ch.GetEntry(ii)
        img_br=None
        exec('img_br=self.img_ch.%s' % self.img_br_name)
        toc = time.clock()
        print "Get entry time: {} s".format(toc - tic)

        tic = time.clock()        
        img_v = img_br.Image2DArray()
        toc = time.clock()
        print "Set ev_image and get Image2DArray: {} s".format(toc - tic)

        tic = time.clock()        
        imgs = [ img_v[i] for i in xrange(img_v.size()) ]
        toc = time.clock()
        print "imgs list: {} s".format(toc - tic)
        
        tic = time.clock()        
        img_array = [ larcv.as_ndarray(img) for img in imgs ]
        toc = time.clock()
        print "img_array list: {} s".format(toc - tic)

        tic = time.clock()        

        b = np.zeros(list(img_array[0].shape) + [3])
        toc = time.clock()
        print "create b: {} s".format(toc - tic)
        
        imin = 1
        imax = 2
        tic = time.clock()        
        for ix,img in enumerate(img_array):
            img[img < imin] = 0
            img[img > imax] = imax
            
            #img -= np.min(img)
            
            #b[:,:,ix] = img[:,::-1]
            b[:,:,ix] = img
            
        toc = time.clock()
        print "fill data: {} s".format(toc - tic)

        tic = time.clock()        
        b[:,:,0][ b[:,:,1] > 0.0 ] = 0.0
        b[:,:,0][ b[:,:,2] > 0.0 ] = 0.0

        b[:,:,1][ b[:,:,2] > 0.0 ] = 0.0
        toc = time.clock()
        print "slice on data: {} s".format(toc - tic)
        
        print "~~~~~~~~~~~"

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
            if roi.Type() != 7:
                continue
            
            #Three ROIs, one for each plane
            r = {}
            for iy in xrange(3):
                r[iy] = larcv.as_bbox(roi,iy)
            
            rois.append(r)

        return (b,rois,imgs)
