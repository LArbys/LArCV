import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size']=20
matplotlib.rcParams['font.family']='serif'

def energy_plotter(sig_mctree_s,sig_mctree_t,Emin,Emax,deltaE,key,Name):
    fig,ax = plt.subplots(figsize=(10,6))
    bins_  = np.arange(0,Emax+deltaE,deltaE)
    a1 = plt.hist(sig_mctree_s.query("energyInit >=@Emin & energyInit <= @Emax")[key].values,
                  bins=bins_,color='blue',label='Signal')
    a3 = plt.hist(sig_mctree_t.query("energyInit >=@Emin & energyInit <= @Emax")[key].values,
                  bins=bins_,color='red',label='Reconstruction')
    ax.set_xlabel("%s Energy [MeV]"%Name,fontweight='bold')
    ax.set_ylabel("Count / %d [MeV]"%deltaE,fontweight='bold')
    ax.set_title("Good Reconstructed Vertex",fontweight='bold')
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.grid()
    SS="{}_energy.pdf".format(Name)
    print "Write {}".format(SS)
    plt.savefig(SS,format='pdf')
    plt.cla() 
    plt.clf()
    plt.close()

    fig,ax=plt.subplots(figsize=(10,6))
    b=np.nan_to_num(a3[0] / a1[0])
    bidx=np.nonzero(b)
    signal_v=a1[0]
    energy_v=a1[1][:-1]+float(deltaE)/2.0
    eff_v   =b

    res_v=np.where(eff_v==0)[0]
    eff_vv   =[]
    signal_vv=[]
    energy_vv=[]
    eff_v_   =[]
    signal_v_=[]
    energy_v_=[]

    
    for ix in xrange(eff_v.size):
        if ix in res_v:
            eff_vv.append(np.array(eff_v_))
            signal_vv.append(np.array(signal_v_))
            energy_vv.append(np.array(energy_v_))
            eff_v_   =[]
            signal_v_=[]
            energy_v_=[]
            continue

        eff_v_.append(eff_v[ix])
        signal_v_.append(signal_v[ix])
        energy_v_.append(energy_v[ix])

    eff_vv.append(np.array(eff_v_))
    signal_vv.append(np.array(signal_v_))
    energy_vv.append(np.array(energy_v_))
    
    for energy_v_,eff_v_,signal_v_ in zip(energy_vv,eff_vv,signal_vv):
        ax.plot(energy_v_,eff_v_,'o',color='blue',markersize=8)
        ax.errorbar(energy_v_,eff_v_,yerr=np.sqrt(eff_v_*(1-eff_v_)/signal_v_),lw=2,color='blue')
        
    ax.set_ylim(0,1.0)
    ax.set_xlabel("%s Energy [MeV]"%Name,fontweight='bold')
    ax.set_ylabel("Reconstruction Efficiency",fontweight='bold')
    plt.grid()
    plt.tight_layout()
    SS="{}_energy_eff.pdf".format(Name)
    print "Write {}".format(SS)
    plt.savefig(SS,format='pdf')
    plt.cla() 
    plt.clf()
    plt.close()


def angle_plotter(data1,data2,Tmin,Tmax,deltaT,Name):
    fig,ax=plt.subplots(figsize=(10,6))
    bins_=np.arange(Tmin,Tmax+deltaT,deltaT)

    a1=plt.hist(data1,
                bins=bins_,color='blue',label='Signal')
    a3=plt.hist(data2,
                bins=bins_,color='red',label='Reconstruction')
    
    ax.set_xlabel("%s Cos$\Theta$"%Name,fontweight='bold')
    ax.set_ylabel("Count / %d"%deltaT,fontweight='bold')
    ax.set_title("Good Reconstructed Vertex",fontweight='bold')
    ax.set_xlim(-1.0,1.0)
    ax.legend(loc='best',fontsize=20)
    plt.tight_layout()
    plt.grid()
    SS="{}_angle.pdf".format(Name)
    print "Write {}".format(SS)
    plt.savefig(SS,format='pdf')
    plt.cla() 
    plt.clf()
    plt.close()
    
    fig,ax=plt.subplots(figsize=(10,6))
    b=np.nan_to_num(a3[0] / a1[0])
    bidx=np.nonzero(b)
    signal_v = a1[0]
    angle_v  = a1[1][:-1]+float(deltaT)/2.0
    eff_v    = b

    res_v    = np.where(eff_v==0)[0]
    eff_vv   = []
    signal_vv= []
    angle_vv = []
    eff_v_   = []
    signal_v_= []
    angle_v_ = []

    for ix in xrange(eff_v.size):
        if ix in res_v:
            eff_vv.append(np.array(eff_v_))
            signal_vv.append(np.array(signal_v_))
            angle_vv.append(np.array(angle_v_))
            eff_v_=[]
            signal_v_=[]
            angle_v_=[]
            continue

        eff_v_.append(eff_v[ix])
        signal_v_.append(signal_v[ix])
        angle_v_.append(angle_v[ix])

    eff_vv.append(np.array(eff_v_))
    signal_vv.append(np.array(signal_v_))
    angle_vv.append(np.array(angle_v_))    

    for angle_v_,eff_v_,signal_v_ in zip(angle_vv,eff_vv,signal_vv):
        ax.plot(angle_v_,eff_v_,'o',color='blue',markersize=8)
        ax.errorbar(angle_v_,eff_v_,yerr=np.sqrt(eff_v_*(1-eff_v_)/signal_v_),lw=2,color='blue')

    ax.set_xlim(-1.0,1.0)
    ax.set_ylim(0,1.0)
    ax.set_xlabel("%s Cos$\Theta$"%Name,fontweight='bold')
    ax.set_ylabel("Reconstruction Efficiency",fontweight='bold')
    plt.grid()
    plt.tight_layout()
    SS="{}_angle_eff.pdf".format(Name)
    print "Write {}".format(SS)
    plt.savefig(SS,format='pdf')
    plt.cla() 
    plt.clf()
    plt.close()

def vertex_plotter(x0,x1,dx,data,xlabel,ylabel):
    fig,ax=plt.subplots(figsize=(10,6))
    bins=np.arange(x0,x1,dx)
    vtx_v=np.nan_to_num(data)
    ax.hist(vtx_v, bins=bins,alpha=1.0,label='All Vertex')
    ax.set_xlabel(xlabel,fontweight='bold')
    ax.grid()
    ax.set_ylabel(ylabel,fontweight='bold')
    SS="{}_vs_{}.pdf".format(xlabel,ylabel)
    print "Write {}".format(SS)
    plt.savefig(SS,format='pdf')
    plt.cla()
    plt.clf()
    plt.close()
