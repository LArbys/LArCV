import os, sys

if len(sys.argv) != 4:
    print 
    print "sample_name0 = \"nue\""
    print "sample_file0 = str(sys.argv[1])"
    print
    print "sample_name1 = \"cosmic\""
    print "sample_file1 = str(sys.argv[2])"
    print
    print "\"nue_pdfs_{}.root\".format(str(sys.argv[3]))"
    print 
    sys.exit(1)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import ROOT
import root_numpy as rn

from util.fill_df import *

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)

rse    = ['run','subrun','event']
rsev   = ['run','subrun','event','vtxid']
rserv  = ['run','subrun','event','roid','vtxid']

# Vertex data frame
dfs  = {}

# Event data frame
edfs = {}
mdfs = {}

sample_name0 = "nue"
sample_file0 = str(sys.argv[1])

sample_name1 = "cosmic"
sample_file1 = str(sys.argv[2])

for name,file_ in [(sample_name0,sample_file0),
                   (sample_name1,sample_file1)]:
    
    INPUT_FILE  = file_
    comb_df = initialize_df(INPUT_FILE)
    dfs[name] = comb_df.copy()

dfs['nue'] = dfs['nue'].query("scedr<5 and selected1L1P==1")

ts_mdf_m = {}

for name, comb_df in dfs.copy().iteritems():
    print
    print "@ sample",name
    print
    
    ts_mdf = comb_df.copy()
    print "Filling nue assumption..."
    ts_mdf = nue_assumption(ts_mdf)
    print "...done"
    print "Filling parameters..."
    ts_mdf = fill_parameters(ts_mdf)
    print "...done"
    print "Copying..."
    ts_mdf_m[name] = ts_mdf.copy()
    print "...done"



#
# distributions & binning for likelihood
#
import collections
pdf_m = collections.OrderedDict()

#
#
#
xlo= 0.0
xhi= 10.0
dx = 0.2
pdf_m['shr_mean_pixel_dist'] = ((xlo,xhi,dx),"Shower - Mean Distance from 2D PCA [pix]")

#
#
#
xlo= 0.0
xhi= 10.0
dx = 0.2
pdf_m['shr_sigma_pixel_dist'] = ((xlo,xhi,dx),"Shower - Sigma Distance from 2D PCA [pix]")

#
#
#
xlo= 0.0
xhi= 1.0
dx = 0.025
pdf_m['shr_par_pixel_ratio'] = ((xlo,xhi,dx),"Shower - Cluster Size Ratio")

xlo= 0.0
xhi= 1.0
dx = 0.025
pdf_m['trk_par_pixel_ratio'] = ((xlo,xhi,dx),"Track - Cluster Size Ratio")

xlo=-1.0
xhi= 1.0
dx = 0.05
pdf_m['cosangle3d'] = ((xlo,xhi,dx),"Cos 3D Opening Angle")

xlo= 0
xhi= 180
dx = 5
pdf_m['anglediff0'] = ((xlo,xhi,dx),"2D Angle Difference [deg]")

xlo=-1.0
xhi= 1.0
dx = 0.05
pdf_m['shr_trunk_pca_cos_theta_estimate'] = ((xlo,xhi,dx),"Shower - Cos 3D Beam Angle")

xlo=-1.0
xhi= 1.0
dx = 0.05
pdf_m['trk_trunk_pca_cos_theta_estimate'] = ((xlo,xhi,dx),"Track - Cos 3D Beam Angle")

#
# Length
#
xlo= 0
xhi= 500
dx = 10
pdf_m['shr_avg_length'] = ((xlo,xhi,dx),"Shower - Average 2D Length [pix]")

xlo= 0
xhi= 500
dx = 10
pdf_m['trk_avg_length'] = ((xlo,xhi,dx),"Track - Average 2D Length [pix]")

#
# Area
#
xlo= 0
xhi= 1000
dx = 20
pdf_m['shr_avg_area'] = ((xlo,xhi,dx),"Shower - Average 2D Area [pix^2]")

xlo= 0
xhi= 1000
dx = 20
pdf_m['trk_avg_area'] = ((xlo,xhi,dx),"Track - Average 2D Area [pix^2]")

#
# 3D length
#
xlo= 0
xhi= 100
dx = 2
pdf_m['shr_3d_length'] = ((xlo,xhi,dx),"Shower - 3D Length [cm]")

xlo= 0
xhi= 100
dx = 2
pdf_m['trk_3d_length'] = ((xlo,xhi,dx),"Track - 3D Length [cm]")

#
# Width
#

xlo= 0
xhi= 50
dx = 1
pdf_m['shr_avg_width'] = ((xlo,xhi,dx),"Shower - Average 2D Width [px]")

xlo= 0
xhi= 50
dx = 1
pdf_m['trk_avg_width'] = ((xlo,xhi,dx),"Track - Average 2D Width [px]")

#
# Qaverage/L
#

xlo= 0
xhi= 5000
dx = 50
pdf_m['shr_3d_QavgL'] = ((xlo,xhi,dx),"Shower - Average Charge / 3D Length [pix/cm]")

xlo= 0
xhi= 5000
dx = 50
pdf_m['trk_3d_QavgL'] = ((xlo,xhi,dx),"Track - Average Charge / 3D Length [pix/cm]")

xlo= 0
xhi= 1
dx = 0.025
pdf_m['dqds_ratio_01'] = ((xlo,xhi,dx),"dQ/dX Ratio")

xlo= 0
xhi= 500
dx = 10
pdf_m['dqds_diff_01'] = ((xlo,xhi,dx), "dQ/dX Difference [pix/cm]" )

xlo= 0.5
xhi= 1
dx = 0.01
pdf_m['trk_frac'] = ((xlo,xhi,dx),"Track Frac")

xlo= 0.5
xhi= 1
dx = 0.01
pdf_m['shr_frac'] = ((xlo,xhi,dx), "Shower Frac" )

#
# qsum
#
xlo= 0
xhi= 100000
dx = 1000
pdf_m['shr_qsum_max'] = ((xlo,xhi,dx),"shr_qsum_max")

xlo= 0
xhi= 100000
dx = 1000
pdf_m['shr_qsum_min'] = ((xlo,xhi,dx),"shr_qsum_min")

#
#
#
sig_spectrum_m = {}
bkg_spectrum_m = {}

DRAW=True

for key,item in pdf_m.items():
    xlo,xhi,dx = item[0]
    name       = item[1]
    
    ts_mdf0 = ts_mdf_m['nue'].copy()
    ts_mdf1 = ts_mdf_m['cosmic'].copy()
    
    data0 = ts_mdf0[key].values
    data0 = data0[data0 >= xlo]
    data0 = data0[data0 <= xhi]
    
    data1 = ts_mdf1[key].values
    data1 = data1[data1 >= xlo]
    data1 = data1[data1 <= xhi]
    
    bkg_h = np.histogram(data1,bins=np.arange(xlo,xhi+dx,dx))
    sig_h = np.histogram(data0,bins=np.arange(xlo,xhi+dx,dx))
       
    bkg = bkg_h[0]
    sig = sig_h[0]
    
    bkg = np.where(bkg==0,1,bkg)
    sig = np.where(sig==0,1,sig)
    
    centers=bkg_h[1] + (bkg_h[1][1] - bkg_h[1][0]) / 2.0
    centers = centers[:-1]
    
    bkg_norm = bkg / float(bkg.sum())
    sig_norm = sig / float(sig.sum())
   
    bkg_err = np.sqrt(bkg)
    sig_err = np.sqrt(sig)

    bkg_err_norm = bkg_err /float(bkg.sum())
    sig_err_norm = sig_err /float(sig.sum())
    
    bkg_spectrum_m[key] = (centers,bkg_norm)
    sig_spectrum_m[key] = (centers,sig_norm)
    
    if DRAW:
        fig,ax=plt.subplots(figsize=(10,6))
        data = bkg_h[1][:-1]
        bins = bkg_h[1]
        centers = data + (data[1] - data[0])/2.0
        
        ax.hist(data,bins=bins,weights=bkg_norm,histtype='stepfilled',color='red',lw=1,alpha=0.1)
        ax.hist(data,bins=bins,weights=bkg_norm,histtype='step',color='red',lw=2,label='Background')

        ax.hist(data,bins=bins,weights=sig_norm,histtype='stepfilled',color='blue',lw=1,alpha=0.1)
        ax.hist(data,bins=bins,weights=sig_norm,histtype='step',color='blue',lw=2,label='Signal')

        ax.errorbar(centers,bkg_norm,yerr=bkg_err_norm,fmt='o',color='red',markersize=0,lw=2)
        ax.errorbar(centers,sig_norm,yerr=sig_err_norm,fmt='o',color='blue',markersize=0,lw=2)
    
        ax.set_ylabel("Fraction of Vertices",fontweight='bold',fontsize=20)
        ax.set_xlabel(name,fontweight='bold',fontsize=20)
        ax.set_xlim(xlo,xhi)
        ax.legend(loc='best')
        ax.grid()
        SS = os.path.join(BASE_PATH,"ll_dump","pdf_%s.pdf" % key)
        print "Dump --> ll_dump/%s" % os.path.basename(SS)
        plt.savefig(SS)
        plt.show()
        

fout = os.path.join(BASE_PATH,"ll_bin","nue_pdfs_{}.root".format(str(sys.argv[3])))
tf = ROOT.TFile(fout,"RECREATE")
tf.cd()

for key in sig_spectrum_m.keys():
    
    sig_spec = sig_spectrum_m[key]
    bkg_spec = bkg_spectrum_m[key]


    #
    # Write signal
    #
    sig_sz   = sig_spec[0].size
    sig_dx   = (sig_spec[0][1] - sig_spec[0][0]) / 2.0
    sig_bins = list(sig_spec[0] - sig_dx) + [sig_spec[0][-1] + sig_dx]
    sig_bins = np.array(sig_bins,dtype=np.float64)
    sig_data = sig_spec[1].astype(np.float64)

    th = None
    th = ROOT.TH1D("sig_" + key,";;",sig_sz,sig_bins)
    res = None
    res = rn.array2hist(sig_data,th)

    res.Write()

    #
    # Write background
    #
    bkg_sz   = bkg_spec[0].size
    bkg_dx   = (bkg_spec[0][1] - bkg_spec[0][0]) / 2.0
    bkg_bins = list(bkg_spec[0] - bkg_dx) + [bkg_spec[0][-1] + bkg_dx]
    bkg_bins = np.array(bkg_bins,dtype=np.float64)
    bkg_data = bkg_spec[1].astype(np.float64)

    th = None
    th = ROOT.TH1D("bkg_" + key,";;",bkg_sz,bkg_bins)
    res = None
    res = rn.array2hist(bkg_data,th)

    res.Write()


tf.Close()
