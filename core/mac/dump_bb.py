from ROOT import larcv
larcv.IOManager
from ROOT import TChain
import matplotlib.pyplot as plt
import sys

ROI_PRODUCER='event_roi'
IMG_PRODUCER='event_image'

roi_tree_name='partroi_%s_tree' % ROI_PRODUCER
roi_br_name='partroi_%s_branch' % ROI_PRODUCER
roi_ch = TChain(roi_tree_name)
roi_ch.AddFile(sys.argv[1])
roi_ch.GetEntry(0)
roi_br=None
exec('roi_br=roi_ch.%s' % roi_br_name)
bb_array = roi_br.ROIArray()

for b in bb_array:
    print b.dump()

print 'Showing Y plane image...'
img_tree_name='image2d_%s_tree' % IMG_PRODUCER
img_br_name='image2d_%s_branch' % IMG_PRODUCER
img_ch = TChain(img_tree_name)
img_ch.AddFile(sys.argv[1])
img_ch.GetEntry(0)
img_br=None
exec('img_br=img_ch.%s' % img_br_name)
img_array = img_br.Image2DArray()
for img in img_array:
    if not img.meta().plane() == 2: continue
    print 'Image meta info'
    print img.meta().dump()
    #img_copy = larcv.Image2D(img)
    #img_copy.compress(768,768)

    mat=larcv.as_ndarray(img)
    plt_img=plt.imshow(mat)
    plt_img.write_png('plane2.png')
    
    sys.exit(1)

    print img.meta().cols()
    print img.meta().rows()
    cols = len(mat)
    rows = len(mat[0])
    print cols
    print rows
    #sys.exit(1)
    for y in xrange(rows):
        for x in xrange(cols):
            if mat[x][y] > 5: print x,y


