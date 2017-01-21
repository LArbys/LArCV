import os,sys
import ROOT
from ROOT import std
from larcv import larcv
import cv2
import numpy as np


hitvar = larcv.HitVariation( "varyhits.cfg" )

# if you want a constructor that takes in these variables, you need to make one in the C++
# but I think you should set these in the cfg file
#hitvar = larcv.HitVariation( larcv_file, sig_translation, sig_amp, sig_width, sig_thresh )

# print addres of HitVariation class
print hitvar

num_images = 10
pars = std.vector("double")(3)
pars[0] = 1.0
pars[1] = 2.0
pars[2] = -10.0
# note the parameters do nothing, but this above is an example of how to make std::vector in python enviroment

labels = std.vector("int")()
entrynumbers = std.vector("int")()

print "Call GenerateImages"
imgset_vo = std.vector( "vector<larcv::Image2D>" )()
imgset_vm = std.vector( "vector<larcv::Image2D>" )()

hitvar.GenerateImages( num_images, pars, imgset_vo, imgset_vm, labels, entrynumbers)

# draw imgs
nbatches = imgset_vo.size()
print "images in batch: ",nbatches
for ibatch in range(0,nbatches):
    img_v = imgset_vo.at(ibatch)
    nimgs = img_v.size()
    out = np.transpose(larcv.as_ndarray(img_v.at(0)), (1,0))
    out[...] = 0
    print "label: ",labels.at(ibatch)
    for iimg in range(2,3):
        img2d = img_v.at(iimg)
        imgnp = np.transpose(larcv.as_ndarray( img2d ), (1,0))
        out +=imgnp
#    out[out>0]=255
    print entrynumbers.at(ibatch)
    imgname=  "testimg_batch%dcombo.jpg" % ( entrynumbers.at(ibatch))
    cv2.imwrite(imgname, out)

#def get_batch(


