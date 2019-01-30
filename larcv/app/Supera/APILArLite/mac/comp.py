import sys
from ROOT import TChain,larcv
larcv.load_pyutil
import cv2,numpy
import matplotlib.pyplot as plt

ch1=TChain("image2d_bnbnu_mc_int00_tree")
ch2=TChain("image2d_segment_bnbnu_mc_tree")

ch1.AddFile(sys.argv[1])
ch2.AddFile(sys.argv[1])

ch1.GetEntry(0)
ch2.GetEntry(0)

img1_v=ch1.image2d_bnbnu_mc_int00_branch.Image2DArray()
img2_v=ch2.image2d_segment_bnbnu_mc_branch.Image2DArray()

for i in xrange(3):
    img1=larcv.as_ndarray(img1_v[i])
    img2=larcv.as_ndarray(img2_v[i])

    for col in xrange(len(img1)):
        for row in xrange(len(img1[col])):
            v1=img1[col][row]
            if v1<5: v1=0
            if v1>5: v1=1
            v2=img2[col][row]
            if v2<0: v2=0
            if v2>0: v2=1
            if not v1 == v2: print v1-v2,
        print
    print
    plt.imshow(img1)
    #cv2.imshow('img1',img1)
    #cv2.imshow('img2',img2)
    aho=input()
