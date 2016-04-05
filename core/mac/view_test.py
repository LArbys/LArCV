import pyqtgraph as pg

import ROOT, sys,os
from ROOT import larcv

import numpy as np

tf = ROOT.TFile.Open("/Users/vgenty/Downloads/out.root","READ")
tree = tf.image2d_event_image_tree

co = { 0 : 'r', 1 : 'g' , 2 : 'b' }
ii=0 
while(True):

    print ii
    tree.GetEntry(ii)
    ev_image = tree.image2d_event_image_branch
    print ev_image
    img_v = ev_image.Image2DArray()


    imgs      = [ img_v[i] for i in xrange(img_v.size()) ]
    img_array = [ larcv.as_ndarray(img) for img in imgs ]

    # roi  = ROOT.larcv.ROI()
    # bb_v = [ ROOT.larcv.ImageMeta(img.meta().cols()/2, img.meta().rows()/2,
    #                               img.meta().rows()/2, img.meta().cols()/2,
    #                               img.meta().min_x()+25*ix,
    #                               img.meta().max_y()-25*ix,ix) for ix,img in enumerate(imgs) ]

    # for bb in bb_v:
    #     roi.AppendBB(bb)

    b = np.zeros(list(img_array[0].shape) + [3])

    # rr = []
    # ts = []


    imin = 5

    for ix,img in enumerate(img_array):
        # Retreive it
        # bbox = ROOT.larcv.as_bbox(roi,ix)
    
        #r1 = pg.QtGui.QGraphicsRectItem(bbox["xy"][0],bbox["xy"][1],bbox["width"],bbox["height"])
        #r1.setPen(pg.mkPen(co[ix]))
        #r1.setBrush(pg.mkBrush(None))
        #rr.append(r1)

        img[img < imin] = 0
        
        img -= np.min(img)
        
        b[:,:,ix] = img
        
    #app = QtGui.QApplication([])
    win = pg.GraphicsWindow(title="Fuck P100")
    win.setWindowTitle('Fuck P100')

    vb  = win.addViewBox()
    imi = pg.ImageItem()
    imi.setImage(b)
    vb.addItem(imi)
    # for r in rr:
    #     vb.addItem(r)

    raw_input('')
    ii+=1
