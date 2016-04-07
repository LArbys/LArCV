from ROOT import TChain, larcv
import numpy as np
import time


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
    def __init__(self,rfile):

        ROI_PRODUCER='event_roi'
        LR_IMG_PRODUCER='event_image'
        HR_IMG_PRODUCER='mcint00'

        self.LR_img_tree_name='image2d_%s_tree' % LR_IMG_PRODUCER
        self.LR_img_br_name='image2d_%s_branch' % LR_IMG_PRODUCER

        self.HR_img_tree_name='image2d_%s_tree' % HR_IMG_PRODUCER
        self.HR_img_br_name='image2d_%s_branch' % HR_IMG_PRODUCER
        
        self.LR_img_ch = TChain(self.LR_img_tree_name)
        self.HR_img_ch = TChain(self.HR_img_tree_name)
        
        self.LR_img_ch.AddFile(rfile)
        self.HR_img_ch.AddFile(rfile)

        self.LR_img_ch.GetEntry(0)
        self.HR_img_ch.GetEntry(0)
        
        self.roi_tree_name='partroi_%s_tree' % ROI_PRODUCER
        self.roi_br_name='partroi_%s_branch' % ROI_PRODUCER

        self.roi_ch = TChain(self.roi_tree_name)
        self.roi_ch.AddFile(rfile)
        self.roi_ch.GetEntry(0)

        self.co = { 0 : 'r', 1 : 'g' , 2 : 'b' }


        self.loaded = {}
        
    def get_event_image(self,ii,imin,imax,lr=True) :

        if (ii,imin,imax,lr) in self.loaded.keys():
            print "\t>> Already loaded this image return it\n"
            return self.loaded[(ii,imin,imax,lr)] 
        
        if lr == True :
            self.LR_img_ch.GetEntry(ii)
            img_br=None
            exec('img_br=self.LR_img_ch.%s' % self.LR_img_br_name)
        else:
            self.HR_img_ch.GetEntry(ii)
            img_br=None
            exec('img_br=self.HR_img_ch.%s' % self.HR_img_br_name)
            
        img_v = img_br.Image2DArray()

        imgs      = [ img_v[i] for i in xrange(img_v.size()) ]
        img_array = [ larcv.as_ndarray(img) for img in imgs ]

        if ( len(img_array) == 0 ) :
            return (None,None,None)

        
        if lr == False:
            trmax = get_max_size(imgs)
            print "trmax is {}".format(trmax)
            
            b = np.zeros([ trmax[0] + 1, trmax[1] + 1,3 ])
            print "bshape {}".format(b.shape)
            print "trmax is {}".format(trmax)
        else:
            b = np.zeros(list(img_array[0].shape) + [3])



        for ix,img in enumerate(img_array):
            img[img < imin] = 0
            img[img > imax] = imax

            if lr == False:

                aa = (0,trmax[0]+1 - imgs[ix].meta().cols())
                bb = (trmax[1]+1 - imgs[ix].meta().rows(),0)

                print "aa : {} bb : {}".format(aa,bb)
                print "IX: {} img shape: {}".format(ix,img.shape)
                print "aa: {} bb: {}".format(aa,bb)
                img = np.pad(img,(aa,bb),
                             mode='constant',
                             constant_values=0)
                print "after padding img shape: {}".format(img.shape)
                
                
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

        self.loaded[(ii,imin,imax,lr)]  =  (b,rois,imgs)
        return (b,rois,imgs)
