import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size']=20
matplotlib.rcParams['font.family']='serif'


def eff_plot(df1,df2,
             Xmin,Xmax,dX,
             query,key,
             Xlabel,Ylabel,
             name):
    
    fig,ax = plt.subplots(figsize=(10,6))
    
    bins = np.arange(Xmin,Xmax+dX,dX)
    
    sig  = plt.hist(df1.query(query)[key].values,bins=bins,color='blue',label='Signal')
    reco = plt.hist(df2.query(query)[key].values,bins=bins,color='red' ,label='Reconstructed')
    
    ax.set_xlabel(Xlabel,fontweight='bold')
    ax.set_ylabel(Ylabel,fontweight='bold')
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.grid()
    SS="{}_hist.pdf".format(name)
    print "Write {}".format(SS)
    plt.savefig("ll_dump/%s" % SS,format='pdf')
    plt.cla() 
    plt.clf()
    plt.close()

    fig,ax = plt.subplots(figsize=(10,6))
    
    reco_sig = np.nan_to_num(reco[0] / sig[0])
    bidx     = np.nonzero(reco_sig)
    signal_v = sig[0]
    param_v  = sig[1][:-1]+float(dX)/2.0
    eff_v    = reco_sig

    res_v    = np.where(eff_v==0)[0]
    eff_vv   = []
    signal_vv= []
    param_vv = []
    eff_v_   = []
    signal_v_= []
    param_v_ = []

    
    for ix in xrange(eff_v.size):
        if ix in res_v:
            eff_vv.append(np.array(eff_v_))
            signal_vv.append(np.array(signal_v_))
            param_vv.append(np.array(param_v_))
            eff_v_    = []
            signal_v_ = []
            param_v_  = []
            continue

        eff_v_.append(eff_v[ix])
        signal_v_.append(signal_v[ix])
        param_v_.append(param_v[ix])

    eff_vv.append(np.array(eff_v_))
    signal_vv.append(np.array(signal_v_))
    param_vv.append(np.array(param_v_))
    
    for param_v_,eff_v_,signal_v_ in zip(param_vv,eff_vv,signal_vv):
        ax.plot(param_v_,eff_v_,'o',color='blue',markersize=8)
        ax.errorbar(param_v_,eff_v_,yerr= np.sqrt( eff_v_ * ( 1 - eff_v_ ) / signal_v_ ),lw=2,color='blue')
        
    ax.set_ylim(0,1.0)
    ax.set_xlabel(Xlabel,fontweight='bold')
    ax.set_ylabel("Reconstruction Efficiency",fontweight='bold')
    plt.grid()
    plt.tight_layout()
    SS="{}_eff.pdf".format(name)
    print "Write {}".format(SS)
    plt.savefig("ll_dump/%s" % SS,format='pdf')
    plt.cla() 
    plt.clf()
    plt.close()
