from larcv import larcv
larcv.load_pyutil
larcv.load_cvutil
import cv2
import ROOT
from ROOT import fcllite
from ROOT import geo2d
from ROOT import larocv
from ROOT import std
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os,sys

proc = larcv.ProcessDriver('ProcessDriver')

#if NOT mc

#CFG="../reco_combined_ssnet_fullchain.cfg"
#CFG="../reco_combined_ssnet_nue.cfg"
CFG="../reco_combined_ssnet_nue_nopp.cfg"
ISMC=True
if ISMC:
    print "This is MC"
else:
    print "This is NOT MC"

cfg_=os.path.basename(CFG)
truth=cfg_.split(".")[0].split("_")[-2]
processed=True
if truth=="true":
    processed=False
processed=False

print "Loading config... ",CFG
proc.configure(CFG)
flist=ROOT.std.vector('std::string')()
flist.push_back("/Users/vgenty/Desktop/nue_8000.root")
#flist.push_back("/Users/vgenty/Desktop/numu_8000.root")
proc.override_input_file(flist)

if ISMC:
    filter_id = proc.process_id("NuFilter")
    mcinfo_id = proc.process_id("LArbysImageMC")
reco_id   = proc.process_id("LArbysImage")
ana_id    = proc.process_id("LArbysImageAna")

if ISMC:
    filter_proc   = proc.process_ptr(filter_id)
    mcinfo_proc   = proc.process_ptr(mcinfo_id)
    mcinfo_proc.SetFilter(filter_proc)

larbysimg     = proc.process_ptr(reco_id)
larbysimg_ana = proc.process_ptr(ana_id)
larbysimg_ana.SetManager(larbysimg.Manager())

proc.override_ana_file("/tmp/test.root")
proc.initialize()
from numpy import array

ignore_list_v=np.array([54,208,405,608,699,748,772,773,798,865,873,
                        918,1085,1450,1547,1702,1844,1970,2368,2470])

event_v=np.array([34, 74, 143, 186, 1119, 1955, 1992, 2004, 2008, 2016, 293, 300, 306, 418, 458, 510, 552, 719, 725, 825, 865, 884, 923, 929, 938, 959, 987, 1013, 1056, 1067, 1220, 1225, 1280, 1291, 1335, 1348, 1364, 1380, 1412, 1438, 1449, 1477, 1514, 1523, 1561, 1566, 1582, 1628, 1677, 1702, 1747, 1799, 1807, 1844, 1883, 1935, 2133, 2194, 2216, 2219, 2224, 2228, 2238])

for event in event_v:
    if event in ignore_list_v: continue
    
    proc.batch_process(event,1)

    if ISMC:
        if (filter_proc.selected()==False): continue
    print "Event is... ",event
    mgr=larbysimg.Manager()
    pygeo = geo2d.PyDraw()
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

    for plane in xrange(len(track_img_v)):
        oshower_img = np.where(oshower_img_v[plane]>10.0,85.0,0.0).astype(np.uint8)
        otrack_img = np.where(otrack_img_v[plane]>10.0,160.0,0.0).astype(np.uint8)
        shower_img = np.where(shower_img_v[plane]>10.0,85.0,0.0).astype(np.uint8)
        track_img = np.where(track_img_v[plane]>10.0,160.0,0.0).astype(np.uint8)
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(15,10))
        oimg = oshower_img + otrack_img
        img  = shower_img  + track_img
        if processed==True:
            ax1.imshow(oimg,cmap='jet',interpolation='none',vmin=0.,vmax=255.)
            ax1.set_xlabel('Time [6 ticks]',fontsize=20)
            ax1.set_ylabel('Wire',fontsize=20)
            ax1.tick_params(labelsize=20)
            ax1.set_xbound(lower=0,upper=640)
            ax1.set_ybound(lower=512,upper=0)
            ax1.set_ylim(512,0)
        else:
            ax2.set_xlabel('Time [6 ticks]',fontsize=20)
            
        ax2.imshow(img,cmap='jet',interpolation='none',vmin=0.,vmax=255.)
        ax2.set_xbound(lower=0,upper=640)
        ax2.set_ybound(lower=512,upper=0)
        ax2.set_ylim(512,0)
        ax2.set_xlabel('Time [6 ticks]',fontsize=20)
        ax2.tick_params(labelsize=20)
        plt.tight_layout()
        SS="out3/%04d_00_track_shower_%d.png"%(event,plane)
        plt.savefig(SS)
        plt.cla()
        plt.clf()
        plt.close()


    colors=['red','green','blue','orange','magenta','cyan','pink']
    colors*=10
    plane=0
    print 'Plane',plane
    for plane in xrange(3):
        fig,ax=plt.subplots(figsize=(12,12),facecolor='w')

        shape_img = track_img_v[plane]
        shape_img=np.where(track_img_v[plane]>0.0,1.0,0.0).astype(np.uint8)
        plt.imshow(shape_img,cmap='Greys',interpolation='none')
        nz_pixels=np.where(shape_img>0.0)

        dm=mgr.DataManager()

        ix=0

        vertex_seeds_v = dm.Data(1,plane).as_vector()
        cluscomp_v     = dm.Data(1,3+plane).as_vector()

        #plot the atomics
        for cluscomp_id in xrange(cluscomp_v.size()):
            cluscomp = cluscomp_v[cluscomp_id]
            for atomic_id in xrange(cluscomp.size()):
                atomic = cluscomp[atomic_id]
                pts=[[atomic[p_id].x,atomic[p_id].y] for p_id in xrange(atomic.size())]
                pts.append(pts[0])
                pts=np.array(pts)
                plt.plot(pts[:,0],pts[:,1],'-o',lw=3,color=colors[ix], alpha = 0.5)
                ix+=1
                

        pts_v = np.array([[vertex_seeds_v[i].x,
                           vertex_seeds_v[i].y] for i in xrange(vertex_seeds_v.size())])
        if pts_v.size>0:
            plt.plot(pts_v[:,0],pts_v[:,1],'*',markersize=20,color='yellow')

        ax.set_aspect(1.0)
        plt.tight_layout()
        try:
            ax.set_ylim(np.min(nz_pixels[0])-10,np.max(nz_pixels[0])+10)
            ax.set_xlim(np.min(nz_pixels[1])-10,np.max(nz_pixels[1])+10)
        except ValueError:
            pass
        
        plt.xlabel('Time [6 ticks]',fontsize=20)
        plt.ylabel('Wire [2 wires]',fontsize=20)
        plt.tick_params(labelsize=20)
        ax.set_aspect(0.8)
        plt.grid()
        SS="out3/%04d_01_atomics_%d.png"%(event,plane)
        print "Saving ",SS
        plt.savefig(SS)
        plt.cla()
        plt.clf()
        plt.close()
        

    if ISMC:
        tru_vtx_w_v = mcinfo_proc._vtx_2d_w_v;
        tru_vtx_t_v = mcinfo_proc._vtx_2d_t_v;
    colors=['red','green','blue','orange','magenta','cyan','pink']
    colors*=10
    plane=0
    print 'Plane',plane
    for plane in xrange(3):
        fig,ax=plt.subplots(figsize=(12,12),facecolor='w')

        shape_img = track_img_v[plane]
        shape_img=np.where(track_img_v[plane]>0.0,1.0,0.0).astype(np.uint8)
        plt.imshow(shape_img,cmap='Greys',interpolation='none')
        nz_pixels=np.where(shape_img>0.0)

        dm=mgr.DataManager()

        ix=0

        vertex_vv  = dm.Data(2,0).as_vector()
        cluscomp_v = dm.Data(1,3+plane).as_vector()

        for cluscomp_id in xrange(cluscomp_v.size()):
            cluscomp = cluscomp_v[cluscomp_id]
            for atomic_id in xrange(cluscomp.size()):
                atomic = cluscomp[atomic_id]
                pts=[[atomic[p_id].x,atomic[p_id].y] for p_id in xrange(atomic.size())]
                pts.append(pts[0])
                pts=np.array(pts)
                plt.plot(pts[:,0],pts[:,1],'-o',lw=3,color=colors[ix], alpha = 0.5)
                ix+=1


        pts_v = np.array([[vertex_vv[i].cvtx2d_v[plane].center.x,
                           vertex_vv[i].cvtx2d_v[plane].center.y] for i in xrange(vertex_vv.size())])
        if pts_v.size>0:
            print "Vertex Candidates @\n",pts_v
            plt.plot(pts_v[:,0],pts_v[:,1],'*',markersize=30,color='cyan')

        if ISMC:
            tru_vtx_w = tru_vtx_w_v[plane]
            tru_vtx_t = tru_vtx_t_v[plane]
            plt.plot(tru_vtx_t,tru_vtx_w,marker='*',markersize=35,color='yellow',alpha=0.5)
        
        ax.set_aspect(1.0)
        plt.tight_layout()
        try:
            ax.set_ylim(np.min(nz_pixels[0])-10,np.max(nz_pixels[0])+10)
            ax.set_xlim(np.min(nz_pixels[1])-10,np.max(nz_pixels[1])+10)
        except ValueError:
            pass
        plt.xlabel('Time [6 ticks]',fontsize=20)
        plt.ylabel('Wire [2 wires]',fontsize=20)
        plt.tick_params(labelsize=20)
        ax.set_aspect(0.8)
        SS="out3/%04d_02_vertex_%d.png"%(event,plane)
        print "Saving ",SS
        plt.savefig(SS)
        plt.cla()
        plt.clf()
        plt.close()


    # In[ ]:

    import matplotlib.patches as patches
    alg=mgr.GetClusterAlg(2).Algo()

    tickscore0_y=[]
    tickscore0_x=[]

    score0_v = alg.TimeBinnedScore0Mean()
    for idx in xrange(score0_v.size()):
        v = score0_v[idx]
        tickscore0_y.append(v)
        tickscore0_x.append(idx*1 + alg.TimeBinMin())

    tickscore1_y=[]
    tickscore1_x=[]
    score1_v = alg.TimeBinnedScore1Mean()
    for idx in xrange(score1_v.size()):
        v = score1_v[idx]
        tickscore1_y.append(v)
        tickscore1_x.append(idx*1 + alg.TimeBinMin())

    tickscore0_x = np.array(tickscore0_x)
    tickscore0_y = np.array(tickscore0_y)
    tickscore1_x = np.array(tickscore1_x)
    tickscore1_y = np.array(tickscore1_y)
    ymin = tickscore0_y.min()
    ymax = tickscore0_y.max()
    if ymin > tickscore1_y.min(): ymin = tickscore1_y.min()
    if ymax < tickscore1_y.max(): ymax = tickscore1_y.max()

    fig,ax = plt.subplots(figsize=(16,8),facecolor='w')
    plt.plot(tickscore0_x,tickscore0_y,marker='o',linestyle='-',color='red',markersize=10)
    plt.plot(tickscore1_x,tickscore1_y,marker='o',linestyle='--',
             markeredgewidth=1,markeredgecolor='blue',markerfacecolor='None',markersize=10)


    minimum_v  = alg.TimeBinnedScoreMinIndex()
    minrange_v = alg.TimeBinnedScoreMinRange()
    for idx in xrange(minimum_v.size()):
        xval = tickscore0_x[minimum_v[idx]]
        plt.plot([xval,xval],[0,360],marker='',linestyle='--',color='black',linewidth=2)
        xstart = tickscore0_x[minrange_v[idx].first]
        xend   = tickscore0_x[minrange_v[idx].second]
        ax.axvspan(xstart,xend, alpha=0.3, color='orange')

    plt.xlabel('Time [6 ticks]',fontsize=20,fontweight='bold')
    plt.ylabel('Summed Angle Difference [deg.]',fontsize=20,fontweight='bold')
    plt.tick_params(labelsize=20)
    plt.tight_layout()
    ax=plt.gca()
    ax.set_ylim(-1,ymax*1.1)
    ax.set_xlim(tickscore0_x.min(),tickscore0_x.max())
    plt.grid()
    SS="out3/%04d_03_score_%d.png"%(event,plane)
    print "Saving ",SS
    plt.savefig(SS)
    plt.cla()
    plt.clf()
    plt.close()


    assman=dm.AssManager()
    #New VertexCluster
    vtx_data=dm.Data(2,0).as_vector()

    #Simply a list of VertexCluster (i.e. vertex-wise list of clusters)
    print "There are ",vtx_data.size(),"vertex found in this event. Note that this is vertex-wise"
    ix=-1
    for vtx in vtx_data:
        ix+=1
        print "<===================Start Vertex3D number ",ix," ==========================>"
        cvtx_v = vtx.cvtx2d_v
        for plane in xrange(3):
            cvtx = cvtx_v[plane]
            fig,ax = plt.subplots(figsize=(12,12),facecolor='w')
            shape_img = img_v[plane]
            shape_img=np.where(img_v[plane]>10.0,1.0,0.0).astype(np.uint8)
            ax.imshow(shape_img,cmap='Greys',interpolation='none')
            nz_pixels=np.where(shape_img>0.0)

            par_data=dm.Data(4,plane)

            ass_t = np.array(assman.GetManyAss(vtx,par_data.ID()))
            if ass_t.size==0:continue

            par_data_v=par_data.as_vector()
            for id_ in ass_t:
                ctor=np.array([[pt.x,pt.y] for pt in par_data_v[id_]._ctor])
                ax.plot(ctor[:,0],ctor[:,1],'-o',lw=2)

            ax.plot(cvtx.center.x,cvtx.center.y,'o',color='red',markersize=10)
            circl=matplotlib.patches.Circle((cvtx.center.x,cvtx.center.y),
                                            cvtx.radius,fc='none',ec='cyan',lw=5,alpha=0.5)
            print "Vertex ",ix," plane ",plane,"..."
            for xs in cvtx.xs_v:
                print  "xs @ [",xs.pt.x,",",xs.pt.y,"]"
                ax.plot(xs.pt.x,xs.pt.y,'o',color='orange',markersize=10,alpha=0.7)
            print
            ax.add_patch(circl)
            
            ax.set_aspect(1.0)
            plt.tight_layout()
            ax.set_ylim(np.min(nz_pixels[0])-10,np.max(nz_pixels[0])+10)
            ax.set_xlim(np.min(nz_pixels[1])-10,np.max(nz_pixels[1])+10)
            plt.xlabel('Time [6 ticks]',fontsize=20)
            plt.ylabel('Wire [2 wires]',fontsize=20)
            plt.tick_params(labelsize=20)
            plt.grid()
            SS="out3/%04d_04_particle_%d_%d.png"%(event,ix,plane)
            print "Saving ",SS
            plt.savefig(SS)
            plt.cla()
            plt.clf()
            plt.close()

        print "<===================End   Vertex3D number ",ix," ==========================>"

    dm=mgr.DataManager()
    colors=['red','green','blue','orange','magenta','cyan','pink']
    colors*=10

    dm=mgr.DataManager()
    data=dm.Data(5,0)
    lintrk_v=data.as_vector()
    print "Found ",lintrk_v.size()," linear track clusters"
    strack_n=-1
    for strack in lintrk_v:
        strack_n+=1
        print "<===================Start LinearTrack number ",strack_n," ==========================>"
        # the only good one...
        e13d=strack.edge1
        e23d=strack.edge2
        print e13d,e23d
        for plane in xrange(3):
            strack2d = strack.get_cluster(plane)
            fig,ax=plt.subplots(figsize=(12,12),facecolor='w')
            shape_img = img_v[plane]
            shape_img=np.where(img_v[plane]>10.0,1.0,0.0).astype(np.uint8)
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

            SS="out3/%04d_05_strack_%02d_%02d.png"%(event,strack_n,plane)
            ax.set_title(SS,fontsize=30)
            plt.savefig(SS)
            plt.cla()
            plt.clf()
            plt.close()
        print "<===================End   LinearTrack number ",strack_n," ==========================>"
        
    dm=mgr.DataManager()
    data=dm.Data(9,0)
    print data
    vtxid=-1;
    print "Got ",data.as_vector().size()," shower 3D vertex estimate"
    for vtx3d in data.as_vector():
        vtxid+=1
        print "<===================Start concrete vertex ",vtxid," ==========================>"
        for plane in xrange(3):
            fig,ax=plt.subplots(figsize=(12,12),facecolor='w')
            shape_img = img_v[plane]
            shape_img=np.where(shape_img>10.0,1.0,0.0).astype(np.uint8)
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

            SS="out3/%04d_06_shower_cvtx_%02d_%02d.png"%(event,vtxid,plane)
            ax.set_title("Vertex Type: %d\n"%vtx3d.type + SS,fontsize=30)
            plt.savefig(SS)
            plt.cla()
            plt.clf()
            plt.close()
        print "<===================end concrete vertex ",vtxid," ==========================>"
        
    assman=dm.AssManager()

    #New VertexCluster
    vtx_data=dm.Data(9,0).as_vector()
    vtxid=-1
    for vtx in vtx_data:
        vtxid+=1
        for plane in xrange(3):
            fig,ax = plt.subplots(figsize=(12,12),facecolor='w')
            shape_img = img_v[plane]
            shape_img=np.where(shape_img>10.0,1.0,0.0).astype(np.uint8)
            plt.imshow(shape_img,cmap='Greys',interpolation='none')
            nz_pixels=np.where(shape_img>0.0)
            
            par_data=dm.Data(11,plane)
            
            ass_t = np.array(assman.GetManyAss(vtx,par_data.ID()))
            if ass_t.size==0:continue
                
            par_data_v=par_data.as_vector()
            for id_ in ass_t:
                ctor=np.array([[pt.x,pt.y] for pt in par_data_v[id_]._ctor])
                plt.plot(ctor[:,0],ctor[:,1],'-o',lw=2)
            
            ax.set_ylim(np.min(nz_pixels[0])-10,np.max(nz_pixels[0])+10)
            ax.set_xlim(np.min(nz_pixels[1])-10,np.max(nz_pixels[1])+10)
            SS="out3/%04d_07_shower_%02d_%02d_.png"%(event,vtxid,plane)
            ax.set_title("Vertex Type: %d\n"%vtx.type + SS,fontsize=30)
            plt.savefig(SS)
            plt.cla()
            plt.clf()
            plt.close()

    #track output
    vtx_data=dm.Data(12,0).as_vector()
    vtxid=-1
    for vtx in vtx_data:
        vtxid+=1
        for plane in xrange(3):
            fig,ax = plt.subplots(figsize=(12,12),facecolor='w')
            shape_img = img_v[plane]
            shape_img=np.where(shape_img>10.0,1.0,0.0).astype(np.uint8)
            plt.imshow(shape_img,cmap='Greys',interpolation='none')
            nz_pixels=np.where(shape_img>0.0)

            vtx2d=vtx.cvtx2d_v[plane].center
            
            ax.plot(vtx2d.x,vtx2d.y,'*',color='red',markersize=35,alpha=0.8)            
            if ISMC:
                tru_vtx_w = tru_vtx_w_v[plane]
                tru_vtx_t = tru_vtx_t_v[plane]
                plt.plot(tru_vtx_t,tru_vtx_w,marker='*',markersize=35,color='yellow',alpha=0.5)
            
            ax.set_ylim(np.min(nz_pixels[0])-10,np.max(nz_pixels[0])+10)
            ax.set_xlim(np.min(nz_pixels[1])-10,np.max(nz_pixels[1])+10)
            SS="out3/%04d_08_trackvtx_%02d_%02d_.png"%(event,vtxid,plane)
            ax.set_title("Vertex Type: %d\n"%vtx.type + SS,fontsize=30)
            plt.savefig(SS)
            plt.cla()
            plt.clf()
            plt.close()

            
    #Shower output
    vtx_data=dm.Data(13,0).as_vector()
    vtxid=-1
    for vtx in vtx_data:
        vtxid+=1
        for plane in xrange(3):
            fig,ax = plt.subplots(figsize=(12,12),facecolor='w')
            shape_img = img_v[plane]
            shape_img=np.where(shape_img>10.0,1.0,0.0).astype(np.uint8)
            plt.imshow(shape_img,cmap='Greys',interpolation='none')
            nz_pixels=np.where(shape_img>0.0)

            vtx2d=vtx.cvtx2d_v[plane].center
            
            ax.plot(vtx2d.x,vtx2d.y,'*',color='red',markersize=35,alpha=0.8)            

            if ISMC:
                tru_vtx_w = tru_vtx_w_v[plane]
                tru_vtx_t = tru_vtx_t_v[plane]
                plt.plot(tru_vtx_t,tru_vtx_w,marker='*',markersize=35,color='yellow',alpha=0.5)
            
            ax.set_ylim(np.min(nz_pixels[0])-10,np.max(nz_pixels[0])+10)
            ax.set_xlim(np.min(nz_pixels[1])-10,np.max(nz_pixels[1])+10)
            SS="out3/%04d_09_showervtx_%02d_%02d_.png"%(event,vtxid,plane)
            ax.set_title("Vertex Type: %d\n"%vtx.type + SS,fontsize=30)
            plt.savefig(SS)
            plt.cla()
            plt.clf()
            plt.close()


    #Shower output
    vtx_data=dm.Data(14,0).as_vector()
    vtxid=-1
    for vtx in vtx_data:
        vtxid+=1
        for plane in xrange(3):
            fig,ax = plt.subplots(figsize=(12,12),facecolor='w')
            shape_img = img_v[plane]
            shape_img=np.where(shape_img>10.0,1.0,0.0).astype(np.uint8)
            plt.imshow(shape_img,cmap='Greys',interpolation='none')
            nz_pixels=np.where(shape_img>0.0)

            vtx2d=vtx.cvtx2d_v[plane].center            
            ax.plot(vtx2d.x,vtx2d.y,'*',color='red',markersize=35,alpha=0.8)
            circl=matplotlib.patches.Circle((vtx2d.x,vtx2d.y),vtx.cvtx2d_v[plane].radius,fc='none',ec='cyan',lw=5)
            ax.add_patch(circl)
            for xs in vtx.cvtx2d_v[plane].xs_v:
                ax.plot(xs.pt.x,xs.pt.y,'o',color='orange',markersize=10)
                
            if ISMC:
                tru_vtx_w = tru_vtx_w_v[plane]
                tru_vtx_t = tru_vtx_t_v[plane]
                ax.plot(tru_vtx_t,tru_vtx_w,marker='*',markersize=35,color='yellow',alpha=0.5)

            par_data=dm.Data(14,plane+1)

            ass_t = np.array(assman.GetManyAss(vtx,par_data.ID()))
            if ass_t.size==0:
                print "No particle data found for vertex ID ",vtxid," plane ",plane
                continue

            par_data_v=par_data.as_vector()
            for id_ in ass_t:
                ctor=np.array([[pt.x,pt.y] for pt in par_data_v[id_]._ctor])
                ax.plot(ctor[:,0],ctor[:,1],'-',lw=5)

            
            ax.set_ylim(np.min(nz_pixels[0])-10,np.max(nz_pixels[0])+10)
            ax.set_xlim(np.min(nz_pixels[1])-10,np.max(nz_pixels[1])+10)
            SS="out3/%04d_10_par_%02d_%02d_.png"%(event,vtxid,plane)
            ax.set_title("Vertex Type: %d\ndist=%f"%(vtx.type,np.sqrt(np.power(tru_vtx_t-vtx2d.x,2)+np.power(tru_vtx_w-vtx2d.y,2))),fontsize=30)
            plt.savefig(SS)
            plt.cla()
            plt.clf()
            plt.close()

            
proc.finalize()
