import os,sys

print "-----------------------"
print "| DL SSNET IMG DUMPER |"
print "-----------------------"

if len(sys.argv) != 9: 
    print ""
    print "PKL_FILE     = str(sys.argv[1])"
    print "STAGE1_DIR   = str(sys.argv[2])"
    print "STAGE2_DIR   = str(sys.argv[3])"
    print "RUN          = int(sys.argv[4])"
    print "SUBRUN       = int(sys.argv[5])"
    print "EVENT        = int(sys.argv[6])"
    print "VTXID        = int(sys.argv[7])"
    print "OUTDIR       = str(sys.argv[8])"
    print ""
    sys.exit(1)
    

PKL_FILE     = str(sys.argv[1])
STAGE1_DIR   = str(sys.argv[2])
STAGE2_DIR   = str(sys.argv[3])
RUN          = int(sys.argv[4])
SUBRUN       = int(sys.argv[5])
EVENT        = int(sys.argv[6])
VTXID        = int(sys.argv[7])
OUTDIR       = str(sys.argv[8])

import pandas as pd

df = pd.read_pickle(PKL_FILE);

row = df.query("run==@RUN&subrun==@SUBRUN&event==@EVENT").iloc[0]
ENTRY = int(row['entry'])
fname = str(row['fname'])

runmod100    = RUN%100
rundiv100    = RUN/100
subrunmod100 = SUBRUN%100
subrundiv100 = SUBRUN/100

jobtag         = 10000*RUN + SUBRUN
SSNET_DIR      = os.path.join(STAGE1_DIR,"%03d/%02d/%03d/%02d/"%(rundiv100,runmod100,subrundiv100,subrunmod100))
VERTEXOUT_DIR  = os.path.join(STAGE2_DIR,"%03d/%02d/%03d/%02d/"%(rundiv100,runmod100,subrundiv100,subrunmod100))


WIRE_FILE = os.path.join(SSNET_DIR,"ssnetout-larcv-%s.root" % fname)
if os.path.exists(WIRE_FILE) == False:
    print
    print "Could _not_ find SSNET file!!!"
    print WIRE_FILE
    print
    
    sys.exit(1)

PGRAPH_FILE = os.path.join(VERTEXOUT_DIR,"vertexout_%05d.root" % jobtag)
if os.path.exists(PGRAPH_FILE) == False:
    print
    print "Could _not_ find PGRAPH file!!!"
    print PGRAPH_FILE
    print
    sys.exit(1)

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

from util.dutil import segment_image

from larcv import larcv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ROOT
from ROOT import geo2d
pygeo = geo2d.PyDraw()

mycmap = matplotlib.cm.get_cmap('jet')
mycmap.set_under('w')
mycmap.set_over('b')

#
# ssnet image readout
#
proc = larcv.ProcessDriver('ProcessDriver') 
proc.configure(os.path.join(BASE_PATH,"cfg","ssnet_config.cfg"))

flist_v = ROOT.std.vector("std::string")()
flist_v.push_back(WIRE_FILE)
flist_v.push_back(PGRAPH_FILE)
proc.override_input_file(flist_v)
proc.initialize()

ext_id   = proc.process_id("LArbysImageExtract")
print "GOT: LArbysImageExtract @ ",ext_id
ext_proc = proc.process_ptr(ext_id)   
print "reading ssnet..."
proc.process_entry(ENTRY)
print "...read"

#
# vertex readout
#
iom = larcv.IOManager()
iom.add_in_file(WIRE_FILE)
iom.add_in_file(PGRAPH_FILE)
iom.initialize()
iom.read_entry(ENTRY)

ev_img    = iom.get_data(larcv.kProductImage2D,"wire")
ev_pgraph = iom.get_data(larcv.kProductPGraph,"test")
ev_ctor   = iom.get_data(larcv.kProductPixel2D,"test_ctor")

print "@run=",ev_img.run(),"subrun=",ev_img.subrun(),"event=",ev_img.event()

print "GOT:",ev_img.Image2DArray().size(),"images"
print "GOT:",ev_pgraph.PGraphArray().size(),"vertices"

print "ev_pgraph sz=",ev_pgraph.PGraphArray().size()
pgraph = ev_pgraph.PGraphArray().at(VTXID)

parray = pgraph.ParticleArray()
pgroi  = parray.front()
meta_v = pgroi.BB()

X = pgroi.X()
Y = pgroi.Y()
Z = pgroi.Z()

colors = ['magenta','white','yellow']
ctor_v = []

for parid,id_ in enumerate(np.array(pgraph.ClusterIndexArray())):
    pix2d_vv=[None,None,None]
    print "@parid=",parid,"&id_=",id_
    
    for plane in xrange(3):
        pix2d_v = ev_ctor.Pixel2DClusterArray(plane).at(id_)
        pix2d_v = [[pix2d_v[ii].X(),pix2d_v[ii].Y()] for ii in xrange(pix2d_v.size())]
        if len(pix2d_v)!=0: pix2d_v.append(pix2d_v[0])
        pix2d_vv[plane] = np.array(pix2d_v)

    ctor_v.append(pix2d_vv)

#
# fill images
#
print "fill images..."
ext_proc.FillcvMat(pgroi)
print "...filled"

print "pythonize image..."
img_v = segment_image(ext_proc,pygeo,1)
print "...pythonized"

for plane in xrange(3):
    
    xpixel = ROOT.Double()
    ypixel = ROOT.Double()

    SS="Plane {}".format(plane)
    fig,ax=plt.subplots(figsize=(20,22))
    ax.imshow(img_v[plane],cmap=mycmap,vmin=0,vmax=255,interpolation='none')

    larcv.Project3D(meta_v[plane],X,Y,Z,0.0,plane,xpixel,ypixel)

    ax.plot(xpixel,meta_v[plane].rows()-ypixel,'*',color='yellow',markersize=30)

    xmin =  1e9
    xmax = -1e9

    ymin =  1e9
    ymax = -1e9

    for ix,ctor in enumerate(ctor_v):

        ctor=ctor[plane]
        if ctor.size==0: continue
        ax.plot(ctor[:,0],ctor[:,1],'-',lw=5,color=colors[ix],alpha=1.0)
        
        if ctor[:,0].min() < xmin : xmin = ctor[:,0].min()
        if ctor[:,0].max() > xmax : xmax = ctor[:,0].max()

        if ctor[:,1].min() < ymin : ymin = ctor[:,1].min()
        if ctor[:,1].max() > ymax : ymax = ctor[:,1].max()

    
    if xmin != 1e9:
        ax.set_xlim(xmin-50,xmax+50)
        ax.set_ylim(ymin-50,ymax+50)

    SS = "{}_{}_{} Plane={}".format(ev_img.run(),ev_img.subrun(),ev_img.event(),plane)
    ax.set_title(SS,fontweight='bold',fontsize=50)        

    this_num = os.path.basename(PGRAPH_FILE).split(".")[0].split("_")[-1]
    SS=os.path.join(OUTDIR,"{}_{}_{}_{}_{}_{}_{}_{}_SSNET.png".format(ev_img.run(),
                                                                      ev_img.subrun(),
                                                                      ev_img.event(),
                                                                      this_num,
                                                                      ENTRY,
                                                                      VTXID,
                                                                      parid,
                                                                      plane))
    

    plt.savefig(SS)
    plt.clf()
    plt.cla()
    plt.close()

sys.exit(0)
