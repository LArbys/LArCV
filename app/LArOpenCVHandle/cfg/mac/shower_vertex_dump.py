from larcv import larcv
larcv.load_pyutil
larcv.load_cvutil
import sys,os
import cv2
import ROOT
from ROOT import fcllite
from ROOT import geo2d
from ROOT import larocv

import matplotlib
import matplotlib.pyplot as plt
plt.rc('grid', linestyle="-", color='black')

import numpy as np

proc = larcv.ProcessDriver("ProcessDriver")
proc.configure('../reco_shower_debug.cfg')
f=ROOT.std.string("/Users/vgenty/Desktop/nue_track_shower.root")
flist=ROOT.std.vector('string')()
flist.push_back(f)
proc.override_input_file(flist)
proc.initialize()

mcinfo_id = proc.process_id("LArbysImageMC")                                                                                                        
reco_id   = proc.process_id("LArbysImage")
                                                                                                                                                    
mcinfo_proc   = proc.process_ptr(mcinfo_id)
larbysimg     = proc.process_ptr(reco_id)
#array=np.array([157, 173, 222, 228, 240, 271, 366, 372])
array = np.array([304])
for event in array:

    print "Processing event: ",event
    proc.batch_process(event,1)
    print "Done"
    pygeo   = geo2d.PyDraw()
    iom=proc.io()
    print iom.current_entry()

    mgr=larbysimg.Manager()
    img_v = []
    shower_img_v = []
    track_img_v  = []

    print "My name is ",mgr.Name()
    print "Requesting image id: ",0
    adc_mat_v    = mgr.InputImages(0)
    print "Requesting image id: ",1
    track_mat_v  = mgr.InputImages(1)
    print "Requesting image id: ",2
    shower_mat_v = mgr.InputImages(2)

    for plane in xrange(track_mat_v.size()):
        shower_img_v.append(pygeo.image(shower_mat_v[plane]))
        track_img_v.append(pygeo.image(track_mat_v[plane]))
        img_v.append(shower_img_v[-1] + track_img_v[-1])
        shower_img=np.where(shower_img_v[plane]>10.0,85.0,0.0).astype(np.uint8)
        track_img=np.where(track_img_v[plane]>10.0,160.0,0.0).astype(np.uint8)
        fig,ax=plt.subplots(figsize=(12,12),facecolor='w')
        plt.imshow(shower_img+track_img,cmap='jet',interpolation='none',vmin=0.,vmax=255.)
        plt.xlabel('Time [6 ticks]',fontsize=20)
        plt.ylabel('Wire',fontsize=20)
        plt.tick_params(labelsize=20)
        ax.set_aspect(0.8)
        plt.tight_layout()
        SS="out/%04d_00_plane_%02d.png"%(event,plane)
        ax.set_title(SS,fontsize=30)
        plt.savefig(SS)
        plt.cla()
        plt.clf()
        plt.close()

    #["LinearTrackFinder","SuperClusterMaker","ShowerVertexSeeds","ShowerVertexEstimate"]
    dm=mgr.DataManager()
    colors=['red','green','blue','orange','magenta','cyan','pink']
    colors*=10

    dm=mgr.DataManager()
    data=dm.Data(0,0)
    lintrk_v=data.as_vector()
    print "Found ",lintrk_v.size()," linear track clusters"
    strack_n=-1
    for strack in lintrk_v:
        strack_n+=1
        # the only good one...
        e13d=strack.edge1
        e23d=strack.edge2
        print e13d,e23d
        for plane in xrange(3):
            strack2d = strack.get_cluster(plane)
            fig,ax=plt.subplots(figsize=(12,12),facecolor='w')
            shape_img = img_v[plane]
            shape_img=np.where(img_v[plane]>0.0,1.0,0.0).astype(np.uint8)
            plt.imshow(shape_img,cmap='Greys',interpolation='none')
            nz_pixels=np.where(shape_img>0.0)
            if strack2d.ctor.size()>0:
                ctor = [[pt.x,pt.y] for pt in strack2d.ctor]
                ctor.append(ctor[0])
                ctor=np.array(ctor)
    
                plt.plot(ctor[:,0],ctor[:,1],'-o',lw=3)
    
                e1=strack2d.edge1
                e2=strack2d.edge2
                
                plt.plot(e1.x,e1.y,'*',color='orange',markersize=20)
                plt.plot(e2.x,e2.y,'*',color='yellow',markersize=20)

                try:
                    vtx2d=e13d.vtx2d_v[plane]
                    pt=vtx2d.pt
                    plt.plot(pt.x,pt.y,'o',color='green',markersize=40,alpha=0.7)
                except:
                    pass

                try:
                    vtx2d=e23d.vtx2d_v[plane]
                    pt=vtx2d.pt
                    plt.plot(pt.x,pt.y,'o',color='green',markersize=40,alpha=0.7)
                except:
                    pass
            
            ax.set_aspect(1.0)
            ax.set_ylim(np.min(nz_pixels[0])-10,np.max(nz_pixels[0])+10)
            ax.set_xlim(np.min(nz_pixels[1])-10,np.max(nz_pixels[1])+10)
            plt.xlabel('Time [6 ticks]',fontsize=20)
            plt.ylabel('Wire [2 wires]',fontsize=20)
            plt.tick_params(labelsize=20)
            ax.set_aspect(0.8)

            SS="out/%04d_01_%04d_plane_%02d.png"%(event,strack_n,plane)
            ax.set_title(SS,fontsize=30)
            plt.savefig(SS)
            plt.cla()
            plt.clf()
            plt.close()


    dm=mgr.DataManager()
    data=dm.Data(3,0)
    print data
    for vtx3d in data.as_vector():
        for plane in xrange(3):
            fig,ax=plt.subplots(figsize=(12,12),facecolor='w')
            shape_img = img_v[plane]
            shape_img=np.where(shape_img>0.0,1.0,0.0).astype(np.uint8)
            plt.imshow(shape_img,cmap='Greys',interpolation='none')
            nz_pixels=np.where(shape_img>0.0)
    
            cvtx=vtx3d.cvtx2d_v[plane]
            plt.plot(cvtx.center.x,cvtx.center.y,'o',color='red',markersize=10)
            circl=matplotlib.patches.Circle((cvtx.center.x,cvtx.center.y),
                                      cvtx.radius,fc='none',ec='cyan',lw=5)
            for xs in cvtx.xs_v:
                plt.plot(xs.pt.x,xs.pt.y,'o',color='orange',markersize=10)
                
            ax.add_patch(circl)
            ax.set_aspect(1.0)
            ax.set_ylim(np.min(nz_pixels[0])-10,np.max(nz_pixels[0])+10)
            ax.set_xlim(np.min(nz_pixels[1])-10,np.max(nz_pixels[1])+10)
            plt.xlabel('Time [6 ticks]',fontsize=20)
            plt.ylabel('Wire [2 wires]',fontsize=20)
            plt.tick_params(labelsize=20)
            ax.set_aspect(0.8)

        SS="out/%04d_02_plane_%02d.png"%(event,plane)
        ax.set_title(SS,fontsize=30)
        plt.savefig(SS)
        plt.cla()
        plt.clf()
        plt.close()
        
    assman=dm.AssManager()
    #New VertexCluster
    vtx_data=dm.Data(3,0).as_vector()
    
    for vtx in vtx_data:
        for plane in xrange(3):
            fig,ax = plt.subplots(figsize=(12,12),facecolor='w')
            shape_img = trk_img_v[plane]+shr_img_v[plane]
            shape_img=np.where(shape_img>0.0,1.0,0.0).astype(np.uint8)
            plt.imshow(shape_img,cmap='Greys',interpolation='none')
            nz_pixels=np.where(shape_img>0.0)
            
            par_data=dm.Data(4,plane)
            
            ass_t = np.array(assman.GetManyAss(vtx,par_data.ID()))
            if ass_t.size==0:continue
                
            par_data_v=par_data.as_vector()
            for id_ in ass_t:
                ctor=np.array([[pt.x,pt.y] for pt in par_data_v[id_]._ctor])
                plt.plot(ctor[:,0],ctor[:,1],'-o',lw=2)
            
            ax.set_ylim(np.min(nz_pixels[0])-10,np.max(nz_pixels[0])+10)
            ax.set_xlim(np.min(nz_pixels[1])-10,np.max(nz_pixels[1])+10)
            SS="out/%04d_03_plane_%02d.png"%(event,plane)
            ax.set_title(SS,fontsize=30)
            plt.savefig(SS)
            plt.cla()
            plt.clf()
            plt.close()

