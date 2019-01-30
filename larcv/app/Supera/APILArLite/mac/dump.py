import sys
from ROOT import TChain,larcv
larcv.load_pyutil
import cv2,numpy
import matplotlib.pyplot as plt

ch=TChain("image2d_segment_bnbnu_mc_tree")

ch.AddFile(sys.argv[1])

ch.GetEntry(0)

img_v=ch.image2d_segment_bnbnu_mc_branch.Image2DArray()

for i in xrange(3):
    img=larcv.as_ndarray(img_v[i])

    for col in xrange(len(img)):
        for row in xrange(len(img[col])):
            if img[col][row]: print '%-3d %-3d' % (row,col), img[col][row]
        print
    print
    plt.imshow(img)
    aho=input()
