import os,sys
import ROOT
from ROOT import std
from larcv import larcv
import cv2



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
imgset_v = std.vector( "vector<larcv::Image2D>" )()
hitvar.GenerateImages( num_images, pars, imgset_v, labels, entrynumbers)

# draw imgs
nbatches = imgset_v.size()
print "images in batch: ",nbatches
for ibatch in range(0,nbatches):
    img_v = imgset_v.at(ibatch)
    nimgs = img_v.size()
    for iimg in range(0,nimgs):
        img2d = img_v.at(iimg)
        imgnp = larcv.as_ndarray( img2d )
        imgname=  "testimg_batch%d_plane%d.jpg" % ( ibatch, iimg )
        cv2.imwrite(imgname, imgnp)




