from ROOT import TFile, TChain
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

c = TChain("pixel_tree")
c.AddFile("ana.root")

p0 = []
p1 = []
p2 = []

for im in c:

  if im.plane == 0 and im.pixel_intens > 0.1:
    p0.append(im.pixel_intens) 

  if im.plane == 1 and im.pixel_intens > 0.1:
    p1.append(im.pixel_intens) 

  if im.plane == 2 and im.pixel_intens > 0.1:
    print "Finally on plane 2"
    p2.append(im.pixel_intens) 

print "plotting now"
plt.hist(p0,100,label='Plane 0',histtype='stepfilled',alpha=0.5,color='r')
plt.hist(p1,100,label='Plane 1',histtype='stepfilled',alpha=0.5,color='g')
plt.hist(p2,100,label='Plane 2',histtype='stepfilled',alpha=0.5,color='b')

plt.legend(loc='upper right');
plt.grid(True)
#plt.ylim(0,200)
plt.xlabel('Pixel Intensity')
plt.title('Pixel Intensity Per Image in 100 Cosmic Events')
plt.show()

