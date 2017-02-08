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
import numpy as np

proc = larcv.ProcessDriver("ProcessDriver")
proc.configure('trk_vtx.cfg')
f=ROOT.std.string("/Users/vgenty/Desktop/numu_ccqe_p00_p07.root")
flist=ROOT.std.vector('string')()
flist.push_back(f)
proc.override_input_file(flist)
proc.initialize()

filter_id = proc.process_id("NuFilter")                                                                                                             
mcinfo_id = proc.process_id("LArbysImageMC")                                                                                                        
reco_id   = proc.process_id("LArbysImage")                                                                                                          
ana_id    = proc.process_id("LArbysImageAna")                                                                                                       
                                                                                                                                                    
filter_proc   = proc.process_ptr(filter_id)                                                                                                         
mcinfo_proc   = proc.process_ptr(mcinfo_id)                                                                                                         
mcinfo_proc.SetFilter(filter_proc)                                                                                                                  
                                                                                                                                                    
larbysimg     = proc.process_ptr(reco_id)                                                                                                           
larbysimg_ana = proc.process_ptr(ana_id)                                                                                                            
larbysimg_ana.SetManager(larbysimg.Manager())   

for event in xrange(50):
    proc.batch_process(event,1)

    if (filter_proc.selected() == False):
        print "Not selected ",event," continue"
        continue

    print "Selected ",event
    
    mgr=larbysimg.Manager()
    img_v = []
    pygeo = geo2d.PyDraw()

    plane=0
    for mat in mgr.InputImages():
        img_v.append(pygeo.image(mat))

    colors=['red','green','blue','orange','magenta','cyan','pink']
    colors*=10

    dm=mgr.DataManager()
    data=dm.Data(0)

    for plane in xrange(3):
        ix=0
        fig,ax=plt.subplots(figsize=(12,12),facecolor='w')
        print 'Plane',plane
        shape_img = img_v[plane]
        shape_img=np.where(img_v[plane]>10.0,1.0,0.0).astype(np.uint8)
        plt.imshow(shape_img,cmap='Greys',interpolation='none')

        nz_pixels=np.where(shape_img>0.0)

        vertex_seeds = data.harvest_seeds(plane)
        print "Plane ",plane," w/ ",vertex_seeds
        cluscomp_v = vertex_seeds.get_compounds()

        #plot the atomics
        for cluscomp_id in xrange(cluscomp_v.size()):
            cluscomp = cluscomp_v[cluscomp_id]
            for atomic_id in xrange(cluscomp.size()):
                atomic = cluscomp[atomic_id]
                pts=[[atomic[p_id].x,atomic[p_id].y] for p_id in xrange(atomic.size())]
                if len(pts)>0: 
                    pts.append(pts[0])
                    pts=np.array(pts)
                    plt.plot(pts[:,0],pts[:,1],'-o',lw=5,color=colors[ix])
                    ix+=1
                    
        points = data.harvest_seed_points(plane)
        print "Found number of seeds ",points.size()
        pts_v = np.array([[points[i].x,points[i].y] for i in xrange(points.size())])
        if pts_v.size>0:
            plt.plot(pts_v[:,0],pts_v[:,1],'*',markersize=20,color='yellow')

        ax.set_aspect(1.0)
        #plt.tight_layout()
        ax.set_ylim(np.min(nz_pixels[0])-10,np.max(nz_pixels[0])+10)
        ax.set_xlim(np.min(nz_pixels[1])-10,np.max(nz_pixels[1])+10)
        plt.xlabel('Time [6 ticks]',fontsize=20)
        plt.ylabel('Wire [2 wires]',fontsize=20)
        plt.tick_params(labelsize=20)
        ax.set_aspect(0.8)
        # plt.show()
        SS="out/%04d_00_seeds_%d.png"%(event,plane)
        print "Saving ",SS
        plt.savefig(SS)
        plt.cla()
        plt.clf()
        plt.close()


    # In[ ]:

    import matplotlib.patches as patches
    alg=mgr.GetClusterAlg(1).Algo()

    dm=mgr.DataManager()
    ref_data = dm.Data(1)
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
    #ax.set_xlim(450,480)
    #plt.show()
    SS="out/%04d_01_tickscore.png"%(event)
    print "Saving ",SS
    plt.savefig(SS)
    plt.cla()
    plt.clf()
    plt.close()


    import matplotlib.patches as patches
    alg=mgr.GetClusterAlg(1).Algo()

    dm=mgr.DataManager()
    ref_data = dm.Data(1)
    for plane in xrange(3):
        wirescore_y = []
        wirescore_x = []
        score_v = alg.WireBinnedScoreMean(plane)
        for idx in xrange(score_v.size()):
            wirescore_y.append(score_v[idx])
            wirescore_x.append(idx + alg.WireBinMin(plane))

        wirescore_y = np.array(wirescore_y)
        wirescore_x = np.array(wirescore_x)


        xmin=0
        xmax=1
        ymin=0
        ymax=1

        if wirescore_y.size > 0 :
            ymin = wirescore_y.min()
            ymax = wirescore_y.max()

        if wirescore_x.size > 0 :
            xmin = wirescore_x.min()
            xmax = wirescore_x.max()

        fig,ax = plt.subplots(figsize=(16,8),facecolor='w')
        plt.plot(wirescore_x,wirescore_y,marker='o',linestyle='-',color='blue',markersize=10)


        minimum_v  = alg.WireBinnedScoreMinIndex(plane)
        minrange_v = alg.WireBinnedScoreMinRange(plane)
        for idx in xrange(minimum_v.size()):
            xval = wirescore_x[minimum_v[idx]]
            plt.plot([xval,xval],[0,360],marker='',linestyle='--',color='black',linewidth=2)
            xstart = wirescore_x[minrange_v[idx].first]
            xend   = wirescore_x[minrange_v[idx].second]
            ax.axvspan(xstart,xend, alpha=0.3, color='orange')

        plt.xlabel('Wire [2 wires]',fontsize=20,fontweight='bold')
        plt.ylabel('Summed Angle Difference [deg.]',fontsize=20,fontweight='bold')
        plt.tick_params(labelsize=20)
        plt.tight_layout()
        ax=plt.gca()
        ax.set_ylim(-1,ymax*1.1)
        ax.set_xlim(xmin,xmax)
        plt.grid()
        #plt.show()
        SS="out/%04d_02_wirescore.png"%(event)
        print "Saving ",SS
        plt.savefig(SS)
        plt.cla()
        plt.clf()
        plt.close()



    # In[ ]:

    import matplotlib.patches as patches

    tru_vtx_w_v = mcinfo_proc._vtx_2d_w_v;
    tru_vtx_t_v = mcinfo_proc._vtx_2d_t_v;
    
    for plane in xrange(3):
        fig,ax=plt.subplots(figsize=(12,12),facecolor='w')
        shape_img=np.where(img_v[plane]>10.0,1.0,0.0).astype(np.uint8)
        plt.imshow(shape_img,cmap='Greys',interpolation='none')
        nz_pixels=np.where(shape_img>0.0)

        alg = mgr.GetClusterAlg(1).Algo()

        cv=larbysimg.Manager().GetClusterAlg(1)
        mgr=larbysimg.Manager()
        dm=mgr.DataManager()

        #Getting seeds
        seeds_data=dm.Data(0)
        seeds = seeds_data.harvest_seeds(plane)
        for compound in seeds.get_compounds():
            for defect in compound.get_defects():
                pt = defect._pt_defect
                plt.plot([pt.x],[pt.y],marker='o',color='blue',markersize=10)
        for pt in seeds.get_pcaxs():
            plt.plot([pt.x],[pt.y],marker='o',color='red',markersize=10)

        ref_data   = dm.Data(1)
        plane_scan = alg.PlaneInfo(plane)

        for circle in plane_scan._circle_scan_v:
            c=patches.Circle((circle.center.x,circle.center.y),circle.radius,ec='cyan',alpha=0.05,fc='none',lw=10)
            ax.add_patch(c)

        minimum_v = alg.TimeBinnedScoreMinIndex()
        for idx in xrange(minimum_v.size()):
            xval = minimum_v[idx] + alg.TimeBinMin()
            plt.plot([xval,xval],[0,2000],marker='',linestyle='--',color='magenta',linewidth=2)

        vtx_vv = ref_data.get_circle_vertex()
        print "Got ",vtx_vv.size(), " circle vertex"
        for vtx_idx in xrange(len(vtx_vv)):
            vtx = vtx_vv[vtx_idx][plane]
            color='magenta'
            if ref_data.get_type(vtx_idx) == 1:
                color='yellow'
            plt.plot([vtx.center.x],[vtx.center.y],marker='$\star$',color=color,markersize=24)

        ymin,ymax = (np.min(nz_pixels[0])-10,np.max(nz_pixels[0])+10)
        xmin,xmax = (np.min(nz_pixels[1])-10,np.max(nz_pixels[1])+10)

        ax.set_ylim(ymin,ymax)
        ax.set_xlim(xmin,xmax)
        plt.xlabel('Time [6 ticks]',fontsize=20)
        plt.ylabel('Wire [2 wires]',fontsize=20)
        plt.tick_params(labelsize=20)
        ax.set_aspect(0.8)
        plt.tight_layout()
        ax=plt.gca()

        tru_vtx_w = tru_vtx_w_v[plane]
        tru_vtx_t = tru_vtx_t_v[plane]

        plt.plot(tru_vtx_t,tru_vtx_w,marker='*',markersize=35,color='yellow',alpha=0.5)
        
        SS="out/%04d_04_vertex_%d.png"%(event,plane)
        print SS
        plt.savefig(SS)
        plt.cla()
        plt.clf()
        plt.close()


