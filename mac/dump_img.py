from larcv import larcv
larcv.IOManager
import matplotlib.pyplot as plt
from ROOT import TChain
import sys

IMAGE_PRODUCER=sys.argv[1]

img_tree_name='image2d_%s_tree' % IMAGE_PRODUCER
img_br_name='image2d_%s_branch' % IMAGE_PRODUCER
img_ch = TChain(img_tree_name)
img_ch.AddFile(sys.argv[2])

cutoff=0
if len(sys.argv) > 3:
    cutoff = int(sys.argv[3])

for entry in xrange(img_ch.GetEntries()):
    img_ch.GetEntry(entry)
    img_br=None
    exec('img_br=img_ch.%s' % img_br_name)
    event_key = img_br.event_key()
    for img in img_br.Image2DArray():
        mat=larcv.as_ndarray(img)
        mat_display=plt.imshow(mat)
        mat_display.write_png('%s_plane%d.png' % (event_key,img.meta().plane()))
    if cutoff and cutoff <= entry:
        break 


