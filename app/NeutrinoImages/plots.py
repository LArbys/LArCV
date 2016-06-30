from ROOT import TFile, TChain
#from ROOT import larlite
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from root_numpy import root2array
#myfile = TFile("ana.root","READ")
#df = pd.DataFrame( root2array( myfile, 'image_tree' ) )
#print df.columns.values

c = TChain("image_tree")
c.AddFile("ana.root")


p0 = []
p1 = []
p2 = []

mpix0 = []
mpix1 = []
mpix2 = []

for im in c:

  if im.plane == 0:
    p0.append(im.pixel_sum) 
    mpix0.append(im.max_pixel) 

  if im.plane == 1:
    p1.append(im.pixel_sum) 
    mpix1.append(im.max_pixel) 

  if im.plane == 2:
    p2.append(im.pixel_sum) 
    mpix2.append(im.max_pixel) 

print len(p2)

#fig = plt.figure()
#
#plt.hist(p0,label='Plane 0',histtype='stepfilled',alpha=0.5,color='r')
#plt.hist(p1,label='Plane 1',histtype='stepfilled',alpha=0.5,color='g')
#plt.hist(p2,label='Plane 2',histtype='stepfilled',alpha=0.5,color='b')
#
#plt.legend(loc='upper left');
#plt.grid(True)
#plt.ylim(0,30)
#plt.xlabel('Total Pixel Intensity')
#plt.xlabel('Total Pixel Intensity Per Image in 100 Cosmic Overlay Events')
#plt.show()

fig2 = plt.figure()
plt.hist(mpix0,45,label='Plane 0',histtype='stepfilled',alpha=0.5,color='r')
plt.hist(mpix1,45,label='Plane 1',histtype='stepfilled',alpha=0.5,color='g')
plt.hist(mpix2,45,label='Plane 2',histtype='stepfilled',alpha=0.5,color='b')

plt.legend(loc='upper right');
plt.grid(True)
plt.ylim(0,200)
#plt.xlim(0,10000)
plt.xlabel('Max Pixel Intensity')
plt.title('Max Pixel Intensity Per Image in 1000 Cosmic Events')
plt.show()

