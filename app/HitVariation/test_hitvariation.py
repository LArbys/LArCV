import os,sys
import ROOT
from ROOT import std
from larcv import larcv
import cv2

#hitvar = larcv.HitVariation()
hitvar = larcv.HitVariation( larcv_file, sig_translation, sig_amp, sig_width, sig_thresh )

#img_v = hivar.genimages(nimgs)
nimgs = img_v.size()
for iimg in range(0,nimgs):
    img2d = img_v.at(iimg)
    imgnp = larcv.as_ndarray( img2d )


    cv2.imwrite(imgname, imgnp)
# draw imgs

print hitvar


