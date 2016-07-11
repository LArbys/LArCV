from ROOT import TChain, TTree
from pylab import * 
import math as m
#import matplotlib.pyplot as plt 
#import numpy as np

c = TChain("image_tree")
c.AddFile("ana.root")

test3_v = [1, 6, 9, 19, 25, 36, 47, 58, 63, 92, 102, 128, 219, 224, 232, 243, 261, 262, 279, 314, 316, 324, 326, 331, 340, 343, 361, 372, 393, 415, 432, 445, 452, 458, 501, 503, 528, 581, 588, 600, 623, 643, 644, 661, 663, 674, 677, 719, 720, 723, 750, 759, 763, 804, 837, 841, 883, 895, 906, 921, 933, 949, 956, 961, 972, 974, 975, 978, 991]

bad_v = [1,6,36,47,128,219,232,458,581,644,674,719,921,949]

final = []

for im in c: 

  density_v = [] 
  radius_v = [] 

  if im.event not in test3_v:
    continue 

  for radius in xrange(5,500,1):
    pix_sum = 0.
    n_pix = 0.
   
    if im.plane == 0:
      for p in xrange(len(im.pix_intens_v)):
        if im.dist_v[p] <= radius:
          pix_sum += im.pix_intens_v[p] 
          n_pix += 1
  

      density_v.append( (n_pix * pix_sum)/ radius**2 )
      radius_v.append(radius)
      if radius == 28 and density_v[-1] < 1.1:
        final.append(im.event)
 
  if len(radius_v) == 0 : 
    continue         

 
  if im.event in bad_v: # and im.event in final: 
    plot(radius_v,density_v,'k+',label=('Event %g' % im.event))
  else:
    plot(radius_v,density_v,label=('Event %g' % im.event))

  #legend(loc=1, ncol=1, borderaxespad=0.)
  title('Density of Pix Intensity vs Dist from max Pix Intensity')
  xlabel('Radius')
  ylabel('Density')


print "Final events are  : ", len(final), ": ", final

grid()
#plt.semilogy()
#plt.semilogx()
plt.ylim(0.1,3)
show()
