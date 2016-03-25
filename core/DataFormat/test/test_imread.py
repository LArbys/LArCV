import ROOT,sys

k=ROOT.larcv.Image2D('Sakura.png')

#k.imshow("aho")
#sys.stdin.readline()

j=k.as_ndarray()
print type(j)
print len(j),len(j[0])

import matplotlib.pyplot as plt
img=plt.imshow(j)

img.write_png('kazu.png')
