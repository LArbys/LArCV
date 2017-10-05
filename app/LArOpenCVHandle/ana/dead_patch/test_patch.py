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
for entry in xrange(int(proc.io().get_n_entries())):
    print "@entry=",entry
    proc.process_entry(entry)

    plane=1
    
    img1 =  mgr.InputImages(0).at(plane).clone()
    img2 =  mgr.InputImages(5).at(plane).clone()
    
    img  = dwp.Patch(img1,img2);

    fig,ax = plt.subplots(figsize=(20,20))
    simg=pygeo.image(img1)
    img_01  = np.where(simg >10.0,50.0 ,0.0).astype(np.uint8)# Threshold shower image
    img_02  = np.where(simg >254.0,90,0.0).astype(np.uint8)# Threshold track  image
    plt.imshow(img_01+img_02,vmin=0,vmax=255,interpolation='none')
    plt.savefig("dump/00_%04d.png" % entry)
    plt.close()
    plt.cla()
    plt.clf()

    fig,ax = plt.subplots(figsize=(20,20))
    simg=pygeo.image(img)
    img_01  = np.where(simg >10.0,50.0 ,0.0).astype(np.uint8)# Threshold shower image
    img_02  = np.where(simg >254.0,90,0.0).astype(np.uint8)# Threshold track  image
    plt.imshow(img_01+img_02,vmin=0,vmax=255,interpolation='none')
    plt.savefig("dump/01_%04d.png" % entry)
    plt.close()
    plt.cla()
    plt.clf()
