import os,sys
import ROOT
from ROOT import std
from larcv import larcv
import cv2



hitvar = larcv.HitVariation( "varyhits.cfg" )
#hitvar = larcv.HitVariation( larcv_file, sig_translation, sig_amp, sig_width, sig_thresh )

# print addres of HitVariation class
print hitvar


#img_v = hivar.genimages(nimgs)
#nimgs = img_v.size()
#for iimg in range(0,nimgs):
#    img2d = img_v.at(iimg)
#    imgnp = larcv.as_ndarray( img2d )
#    imgname=  "test"
#    cv2.imwrite(imgname, imgnp)
# draw imgs




