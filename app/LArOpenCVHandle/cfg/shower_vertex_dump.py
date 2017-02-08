from larcv import larcv
larcv.load_pyutil
larcv.load_cvutil

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
proc.configure('shr_vtx.cfg')
f=ROOT.std.string("/Users/vgenty/Desktop/nue_track_shower.root")
flist=ROOT.std.vector('string')()
flist.push_back(f)
proc.override_input_file(flist)
proc.initialize()

mcinfo_id = proc.process_id("LArbysImageMC")                                                                                                        
reco_id   = proc.process_id("LArbysImage")                                                                                                          
                                                                                                                                                    
mcinfo_proc   = proc.process_ptr(mcinfo_id)                                                                                                         
#mcinfo_proc.SetFilter(filter_proc)                                                                                                                  
                                                                                                                                                    
larbysimg     = proc.process_ptr(reco_id)

for event in xrange(0,50):
    proc.batch_process(event,1)
    
    pygeo   = geo2d.PyDraw()
    
    mgr=larbysimg.Manager()
    img_v = []
    shower_img_v = []
    track_img_v  = []

    track_mat_v  = mgr.InputImages(0)
    shower_mat_v = mgr.InputImages(1)

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
        
    dm=mgr.DataManager()
    colors=['red','green','blue','orange','magenta','cyan','pink']
    colors*=10

    dm=mgr.DataManager()
    data=dm.Data(0)
    lintrk_v=data.get_clusters()
    colors=['red','green','blue','orange','magenta','cyan','pink']
    colors*=10

    dm=mgr.DataManager()
    data=dm.Data(0)
    lintrk_v=data.get_clusters()
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

    mgr=larbysimg.Manager()
    dm=mgr.DataManager()
    lintrack_data=dm.Data(0)
    sshower_data=dm.Data(2)
    track_v  = lintrack_data.get_clusters()
    shower_v = sshower_data.get_showers() 
    for plane in xrange(3):
        print 'Plane',plane
        fig,ax=plt.subplots(figsize=(12,12),facecolor='w')
    
        #Get the image
        shape_img = img_v[plane]
        shape_img=np.where(img_v[plane]>10.0,1.0,0.0).astype(np.uint8)

        plt.imshow(shape_img,cmap='Greys',interpolation='none')
        nz_pixels=np.where(shape_img>0.0)
        
        for track_id in xrange(track_v.size()):
            track = track_v[track_id]
            edge1=track.edge1
            edge2=track.edge2
            if edge1.vtx2d_v.empty() or edge2.vtx2d_v.empty():
                print 'Track',track_id,'has one edge'
            else:
                print 'Track',track_id,'has two edge'

            strack = track.get_cluster(plane)
            if strack.ctor.empty(): continue
                
            pts=np.array([[pt.x,pt.y] for pt in strack.ctor])
            plt.plot(pts[:,0],pts[:,1],'-o',color='blue')
            
            plt.plot([strack.edge1.x],[strack.edge1.y],marker='$\\star$',color='magenta',markersize=24)
            plt.plot([strack.edge2.x],[strack.edge2.y],marker='$\\star$',color='magenta',markersize=24)
            
        for shower_id in xrange(shower_v.size()):
            ctor  = shower_v[shower_id].get_cluster(plane).ctor        
            pts=[[pt.x,pt.y] for pt in ctor]
            if len(pts)>0:
                pts.append(pts[0])
                pts = np.array(pts)
                plt.plot(pts[:,0],pts[:,1],'-o',color='red',lw=3)
            
            start = shower_v[shower_id].get_cluster(plane).start
            plt.plot([start.x],[start.y],marker='$\\star$',color='yellow',markersize=24)
            
        ax.set_aspect(1.0)
        plt.tight_layout()

        plt.xlabel('Time [6 ticks]',fontsize=20)
        plt.ylabel('Wire',fontsize=20)

        y0 = np.min(nz_pixels[0])-10
        y1 = np.max(nz_pixels[0])+10
        x0 = np.min(nz_pixels[1])-10
        x1 = np.max(nz_pixels[1])+10
        
        ax.set_ylim(y0,y1)
        ax.set_xlim(x0,x1)

        major_ticks_x = np.arange(x0,x1+5,5)
        major_ticks_y = np.arange(y0,y1+5,5)

        minor_ticks_x = np.arange(x0+0.5,x1+0.5+1,1)
        minor_ticks_y = np.arange(y0+0.5,y1+0.5+1,1)
 
        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)

        # and a corresponding grid
        ax.grid(which='minor')

        ax.set_xticklabels(major_ticks_x,rotation=45,fontsize=10,minor=False)
        ax.set_yticklabels(major_ticks_y,rotation=45,fontsize=10,minor=False)
        
        plt.tick_params(labelsize=20)
        ax.set_aspect(0.8)
        plt.tight_layout()
        SS="out/%04d_02_plane_%02d.png"%(event,plane)
        ax.set_title(SS,fontsize=30)
        plt.savefig(SS)
        plt.cla()
        plt.clf()
        plt.close()
