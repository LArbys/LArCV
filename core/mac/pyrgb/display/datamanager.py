import ROOT
from ROOT import larcv
import numpy as np
import time

class DataManager(object):
    def __init__(self,rfile):
        self.tf = ROOT.TFile.Open(rfile,"READ")
        self.tree = self.tf.image2d_event_image_tree
        self.co = { 0 : 'r', 1 : 'g' , 2 : 'b' }



    def get_event_image(self,ii) :
        tic = time.clock()
        self.tree.GetEntry(ii)
        toc = time.clock()
        print "Get entry time: {} s".format(toc - tic)

        tic = time.clock()        
        ev_image = self.tree.image2d_event_image_branch
        img_v = ev_image.Image2DArray()
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
        #b = [np.zeros(list(img_array[0].shape) + [3]) for _ in xrange(3)]
        

        b = np.zeros(list(img_array[0].shape) + [3])
        toc = time.clock()
        print "create b: {} s".format(toc - tic)
        
        imin = 5

        tic = time.clock()        
        for ix,img in enumerate(img_array):
            img[img < imin] = 0
            #img -= np.min(img)
            
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
        return b
