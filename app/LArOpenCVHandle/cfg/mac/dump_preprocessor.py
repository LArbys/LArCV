from larcv import larcv
larcv.load_pyutil
larcv.load_cvutil
import cv2
import os,sys

import ROOT
from ROOT import fcllite
from ROOT import geo2d
from ROOT import larocv
from ROOT import std

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

proc = larcv.ProcessDriver('ProcessDriver')

CFG="../pre_process_ssnet.cfg"
print "Loading config... ",CFG
proc.configure(CFG)

flist=ROOT.std.vector('std::string')()
flist.push_back("/Users/vgenty/Desktop/numu_8000.root")
proc.override_input_file(flist)
reco_id = proc.process_id("LArbysImage")
larbysimg = proc.process_ptr(reco_id)
proc.override_ana_file("/tmp/test.root")
proc.initialize()

pygeo = geo2d.PyDraw()
pp=larbysimg.PProcessor()

def extractimage(mgr):
    img_v = []
    oimg_v = []
    track_img_v=[]
    otrack_img_v=[]
    shower_img_v=[]
    oshower_img_v=[]

    for mat in mgr.OriginalInputImages(0):
        oimg_v.append(pygeo.image(mat))
    for mat in mgr.OriginalInputImages(1):
        otrack_img_v.append(pygeo.image(mat))
    for mat in mgr.OriginalInputImages(2):
        oshower_img_v.append(pygeo.image(mat))
    
    for mat in mgr.InputImages(0):
        img_v.append(pygeo.image(mat))
    for mat in mgr.InputImages(1):
        track_img_v.append(pygeo.image(mat))
    for mat in mgr.InputImages(2):
        shower_img_v.append(pygeo.image(mat))

    return (img_v,track_img_v,shower_img_v,oimg_v,otrack_img_v,oshower_img_v)

for event in xrange(100):
    proc.batch_process(event,1)
    mgr=larbysimg.Manager()
    (img_v,track_img_v,shower_img_v,oimg_v,otrack_img_v,oshower_img_v)=extractimage(mgr)

    for plane in xrange(len(track_img_v)):                                                                                                    
        oshower_img = np.where(oshower_img_v[plane]>10.0,85.0,0.0).astype(np.uint8)
        otrack_img = np.where(otrack_img_v[plane]>10.0,160.0,0.0).astype(np.uint8)
        shower_img = np.where(shower_img_v[plane]>10.0,85.0,0.0).astype(np.uint8)
        track_img = np.where(track_img_v[plane]>10.0,160.0,0.0).astype(np.uint8)

        timg=larcv.as_ndarray(proc.io().get_data(larcv.kProductImage2D,"segment_hires_crop").Image2DArray()[plane])
        true_shower_img = np.where((timg==3) | (timg==4) | (timg==5),85.0,0.0).astype(np.uint8)
        true_track_img = np.where((timg==6) | (timg==7) | (timg==8) | (timg==9),160.0,0.0).astype(np.uint8)

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True,figsize=(30,10))


        true_shower_img=true_shower_img[:,::-1]
        true_track_img=true_track_img[:,::-1]
        
        timg = true_shower_img + true_track_img
        oimg = oshower_img + otrack_img
        img  = shower_img  + track_img
        
        ax1.imshow(timg,cmap='jet',interpolation='none',vmin=0.,vmax=255.)                                                      
        ax2.imshow(oimg,cmap='jet',interpolation='none',vmin=0.,vmax=255.)
        ax3.imshow(img ,cmap='jet',interpolation='none',vmin=0.,vmax=255.)

        #label the fraction correct#
        #sum total true pixels
        tot_true_label = float(timg[timg>0].size)


        #find total number of shower pixels correctly labeled out of SSNET!
        res_shower = true_shower_img & oshower_img
        res_shower = float(res_shower[res_shower>0].size)
        res_track  = true_track_img  & otrack_img
        res_track = float(res_track[res_track>0].size)
        ss_frac=(res_track+res_shower)/tot_true_label
        ax2.text(156,456,
                 "%f + %f = %f"%(res_track/tot_true_label,
                                 res_shower/tot_true_label,
                                 (res_track+res_shower)/tot_true_label),
                 fontweight='bold',
                 color='white',
                 fontsize=20)
        
        
        #find total number of shower pixels correctly labeled after pre process
        res_shower = true_shower_img & shower_img
        res_shower = float(res_shower[res_shower>0].size)
        res_track  = true_track_img  & track_img
        res_track = float(res_track[res_track>0].size)
        pre_frac = (res_track+res_shower)/tot_true_label
        ax3.text(156,456,
                 "%f + %f = %f"%(res_track/tot_true_label,
                                 res_shower/tot_true_label,
                                 pre_frac),
                 fontweight='bold',
                 color='white',
                 fontsize=20)

        ax3.text(156,430,
                 "delta = %f"%(pre_frac - ss_frac),
                 color='white',
                 fontsize=20)

        ax2.set_xlabel('Time [6 ticks]',fontsize=20)
        ax1.set_xlim(-150,650)
        ax2.set_xlim(-150,650)
        ax3.set_xlim(-150,650)
        ax1.set_ylim(0,512)
        ax2.set_ylim(0,512)
        ax3.set_ylim(0,512)
        ax1.set_ylabel('Wire',fontsize=20)                                                                                                          
        ax1.tick_params(labelsize=20)                                                                                                           
        plt.tight_layout()        
        SS="out1/00_pre_process_%d_%d_.png"%(event,plane)
        plt.savefig(SS,bbox_inches='tight',pad_inches=0.0)
        plt.cla()
        plt.clf()
        plt.close()
