import os, sys
import ROOT
from ROOT import geo2d
from larcv import larcv
import cv2
pygeo = geo2d.PyDraw()
from ROOT import larocv
from ROOT import cv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.rcParams['font.size']=20
matplotlib.rcParams['font.family']='serif'
import numpy as np

pgraph_prod  = 'test'
pixel2d_prod = 'test_ctor'

proc = larcv.ProcessDriver('ProcessDriver')
proc.configure("nothing.cfg")

FILE1=sys.argv[1]

flist = ROOT.std.vector("std::string")()
flist.push_back(str(FILE1))

proc.override_input_file(flist)

proc.override_ana_file("/tmp/t0.root")
proc.override_output_file("/tmp/t1.root")

proc.initialize()

larbysid = proc.process_id("LArbysImage")
larbys = proc.process_ptr(larbysid)
print "GOT: ",larbys,"@ id=",larbysid

mgr = larbys.Manager()# LArbysImage Manager Instance which has a data manager to data fo LArOpenCV modules
dm  = mgr.DataManager()# DataManager to access LArOpenCV data
idx = 0
for module in dm.Names():# Name function to show all LArOpenCV modules
    print idx, module
    idx+=1

dwp = larocv.DeadWirePatch()
dwp._bondage = True

mycmap = matplotlib.cm.get_cmap('jet')
mycmap.set_under('w')
mycmap.set_over('w')

for entry in [7]:
    print "@entry=",entry
    proc.process_entry(entry)

    for plane in [0,1,2]:
        print "... @plane=",plane
        
        
        aimage =  mgr.InputImages(0).at(plane).clone()
        timage =  mgr.InputImages(1).at(plane).clone()
        simage =  mgr.InputImages(2).at(plane).clone()
        dimage =  mgr.InputImages(5).at(plane).clone()

        aimg =  pygeo.image(aimage)
        timg =  pygeo.image(timage)
        simg =  pygeo.image(simage)
        dimg =  pygeo.image(dimage)
        
        timg = np.where(timg>10,200,1.0)
        simg = np.where(simg>10,100,1.0)

        fig,ax = plt.subplots(figsize=(20,20))

        plt.imshow(timg+simg,vmin=0,vmax=255,interpolation='none',cmap=mycmap)

        plt.savefig("dump/00_%04d_plane_%04d.png" % (entry,plane))
        plt.close()
        plt.cla()
        plt.clf()

        c_aimage = dwp.Patch(aimage,dimage);
        c_aimg   = pygeo.image(c_aimage)

        c_aimg  -= aimg
        c_aimg   = np.where(c_aimg>10,256,0.0)

        fig,ax = plt.subplots(figsize=(20,20))

        plt.imshow(timg+simg+c_aimg,vmin=0,vmax=255,interpolation='none',cmap=mycmap)

        plt.savefig("dump/01_%04d_plane_%04d.png" % (entry,plane))
        plt.close()
        plt.cla()
        plt.clf()
        
