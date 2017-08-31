import os, sys
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import ROOT
import root_numpy as rn
from larocv import larocv

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
sample_file0 = "/Users/vgenty/Desktop/intrinsic_nue/testing_aug02/cheater_sz/comb_ana.root"

sample_name1 = "cosmic"
sample_file1 = "/Users/vgenty/Desktop/cosmics/updated_sz/comb_ana.root"

# sample_name2 = "nue_reco"
# sample_file2 = "/Users/vgenty/Desktop/intrinsic_nue/testing_aug02/nominal_sz/comb_ana.root"


for name,file_ in [(sample_name0,sample_file0),
                   (sample_name1,sample_file1)]:
    
    INPUT_FILE  = file_
    
    #
    # Vertex wise Trees
    #
    vertex_df = pd.DataFrame(rn.root2array(INPUT_FILE,treename='VertexTree'))
    angle_df  = pd.DataFrame(rn.root2array(INPUT_FILE,treename='AngleAnalysis'))
    shape_df  = pd.DataFrame(rn.root2array(INPUT_FILE,treename='ShapeAnalysis'))
    gap_df    = pd.DataFrame(rn.root2array(INPUT_FILE,treename="GapAnalysis"))
    match_df  = pd.DataFrame(rn.root2array(INPUT_FILE,treename="MatchAnalysis"))
    dqds_df   = pd.DataFrame(rn.root2array(INPUT_FILE,treename="dQdSAnalysis"))

    #
    # Combine DataFrames
    #
    comb_df = pd.concat([vertex_df.set_index(rserv),
                         angle_df.set_index(rserv),
                         shape_df.set_index(rserv),
                         gap_df.set_index(rserv),
                         angle_df.set_index(rserv),
                         match_df.set_index(rserv),
                         dqds_df.set_index(rserv)],axis=1)

    #
    # Store vertex wise data frame
    #
    comb_df = comb_df.reset_index()
    comb_df = comb_df.loc[:,~comb_df.columns.duplicated()]

    dfs[name] = comb_df.copy()


# dfs['nue_reco']  = dfs['nue_reco'].query("scedr<5").copy()
dfs['nue'] = dfs['nue'].drop_duplicates(subset=rse)

#
# Compute cosing Angle3D
#
print "Computing opening angle..."
for name, comb_df in dfs.iteritems():
    comb_df['cosangle3d']=comb_df.apply(lambda x : larocv.CosOpeningAngle(x['par_trunk_pca_theta_estimate_v'][0],
                                                                          x['par_trunk_pca_phi_estimate_v'][0],
                                                                          x['par_trunk_pca_theta_estimate_v'][1],
                                                                          x['par_trunk_pca_phi_estimate_v'][1]),axis=1)

    comb_df['angle3d']=comb_df.apply(lambda x : np.arccos(x['cosangle3d']),axis=1)


def track_shower_assumption(df):
    df['trkid'] = df.apply(lambda x : 0 if(x['par1_type']==1) else 1,axis=1)
    df['shrid'] = df.apply(lambda x : 1 if(x['par2_type']==2) else 0,axis=1)

for name, comb_df in dfs.iteritems():

    print
    print "@ sample",name
    print
    
    ts_mdf = comb_df.copy()

    print "Asking nue assumption"
    ts_mdf = ts_mdf.query("par1_type != par2_type")
    track_shower_assumption(ts_mdf)

    print "Asking npar==2"
    print "Asking in_fiducial==1"
    print "Asking pathexists2==1"

    ts_mdf = ts_mdf.query("npar==2")
    ts_mdf = ts_mdf.query("in_fiducial==1")
    ts_mdf = ts_mdf.query("pathexists2==1")
    
    ts_mdf['shr_trunk_pca_theta_estimate'] = ts_mdf.apply(lambda x : np.cos(x['par_trunk_pca_theta_estimate_v'][x['shrid']]),axis=1) 
    ts_mdf['trk_trunk_pca_theta_estimate'] = ts_mdf.apply(lambda x : np.cos(x['par_trunk_pca_theta_estimate_v'][x['trkid']]),axis=1) 

    ts_mdf['shr_avg_length'] = ts_mdf.apply(lambda x : x['length_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['trk_avg_length'] = ts_mdf.apply(lambda x : x['length_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)

    ts_mdf['shr_avg_width'] = ts_mdf.apply(lambda x : x['width_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['trk_avg_width'] = ts_mdf.apply(lambda x : x['width_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)

    ts_mdf['shr_avg_perimeter'] = ts_mdf.apply(lambda x : x['perimeter_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['trk_avg_perimeter'] = ts_mdf.apply(lambda x : x['perimeter_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)

    ts_mdf['shr_avg_area'] = ts_mdf.apply(lambda x : x['area_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['trk_avg_area'] = ts_mdf.apply(lambda x : x['area_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)

    ts_mdf['shr_avg_npixel'] = ts_mdf.apply(lambda x : x['npixel_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['trk_avg_npixel'] = ts_mdf.apply(lambda x : x['npixel_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)

    ts_mdf['shr_3d_length'] = ts_mdf.apply(lambda x : x['par_pca_end_len_v'][x['shrid']],axis=1)
    ts_mdf['trk_3d_length'] = ts_mdf.apply(lambda x : x['par_pca_end_len_v'][x['trkid']],axis=1)

    ts_mdf['shr_3d_QavgL'] = ts_mdf.apply(lambda x : x['qsum_v'][x['shrid']] / x['par_pca_end_len_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['trk_3d_QavgL'] = ts_mdf.apply(lambda x : x['qsum_v'][x['trkid']] / x['par_pca_end_len_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)

    ts_mdf['shr_triangle_d_max'] = ts_mdf.apply(lambda x : x['triangle_d_max_v'][x['shrid']],axis=1)
    ts_mdf['trk_triangle_d_max'] = ts_mdf.apply(lambda x : x['triangle_d_max_v'][x['trkid']],axis=1)
    
    ts_mdf['shr_mean_pixel_dist'] = ts_mdf.apply(lambda x : x['mean_pixel_dist_v'][x['shrid']]/x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['trk_mean_pixel_dist'] = ts_mdf.apply(lambda x : x['mean_pixel_dist_v'][x['trkid']]/x['nplanes_v'][x['trkid']],axis=1)
    
    ts_mdf['shr_sigma_pixel_dist'] = ts_mdf.apply(lambda x : x['sigma_pixel_dist_v'][x['shrid']]/x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['trk_sigma_pixel_dist'] = ts_mdf.apply(lambda x : x['sigma_pixel_dist_v'][x['trkid']]/x['nplanes_v'][x['trkid']],axis=1)

    ts_mdf['shr_par_pixel_ratio'] = ts_mdf.apply(lambda x : x['par_pixel_ratio_v'][x['shrid']],axis=1)
    ts_mdf['trk_par_pixel_ratio'] = ts_mdf.apply(lambda x : x['par_pixel_ratio_v'][x['trkid']],axis=1) 
    
    ts_mdf['anglediff0'] = ts_mdf['anglediff'].values[:,0]
    
    ts_mdf_m[name] = ts_mdf.copy()


#
# distributions & binning for likelihood
#
import collections
pdf_m = collections.OrderedDict()

xlo= 0.0
xhi= 40.0
dx = 2
pdf_m['shr_triangle_d_max'] = ((xlo,xhi,dx),"Shower - Max 2D Deflection [pix]")

xlo= 0.0
xhi= 40.0
dx = 2
pdf_m['trk_triangle_d_max'] = ((xlo,xhi,dx),"Track - Max 2D Deflection [pix]")

xlo= 0.0
xhi= 5.0
dx = 0.2
pdf_m['shr_mean_pixel_dist'] = ((xlo,xhi,dx),"Shower - Mean Distance from 2D PCA [pix]")

xlo= 0.0
xhi= 5.0
dx = 0.2
pdf_m['trk_mean_pixel_dist'] = ((xlo,xhi,dx),"Track - Mean Distance from 2D PCA [pix]")

xlo= 0.0
xhi= 5.0
dx = 0.2
pdf_m['shr_sigma_pixel_dist'] = ((xlo,xhi,dx),"Shower - Sigma Distance from 2D PCA [pix]")

xlo= 0.0
xhi= 5.0
dx = 0.2
pdf_m['trk_sigma_pixel_dist'] = ((xlo,xhi,dx),"Track - Sigma Distance from 2D PCA [pix]")


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
pdf_m['shr_trunk_pca_theta_estimate'] = ((xlo,xhi,dx),"Shower - Cos 3D Beam Angle")

xlo=-1.0
xhi= 1.0
dx = 0.05
pdf_m['trk_trunk_pca_theta_estimate'] = ((xlo,xhi,dx),"Track - Cos 3D Beam Angle")

xlo= 0
xhi= 150
dx = 5
pdf_m['shr_avg_length'] = ((xlo,xhi,dx),"Shower - Average 2D Length [pix]")

xlo= 0
xhi= 150
dx = 5
pdf_m['trk_avg_length'] = ((xlo,xhi,dx),"Track - Average 2D Length [pix]")

# xlo= 0
# xhi= 300
# dx = 10
# pdf_m['shr_avg_area'] = ((xlo,xhi,dx),"Shower - Average 2D Area [pix^2]")

# xlo= 0
# xhi= 300
# dx = 10
# pdf_m['trk_avg_area'] = ((xlo,xhi,dx),"Track - Average 2D Area [pix^2]")


xlo= 0
xhi= 60
dx = 2
pdf_m['shr_3d_length'] = ((xlo,xhi,dx),"Shower - 3D Length [cm]")

xlo= 0
xhi= 60
dx = 2
pdf_m['trk_3d_length'] = ((xlo,xhi,dx),"Track - 3D Length [cm]")

xlo= 0
xhi= 50
dx = 2
pdf_m['shr_avg_width'] = ((xlo,xhi,dx),"Shower - Average 2D Width [px]")

xlo= 0
xhi= 50
dx = 2
pdf_m['trk_avg_width'] = ((xlo,xhi,dx),"Track - Average 2D Width [px]")

xlo= 0
xhi= 600
dx = 10
pdf_m['shr_avg_npixel'] = ((xlo,xhi,dx),"Shower - Average Num. Pixel")

xlo= 0
xhi= 600
dx = 10
pdf_m['trk_avg_npixel'] = ((xlo,xhi,dx),"Track - Average Num. Pixel")

xlo= 0
xhi= 1000
dx = 20
pdf_m['shr_3d_QavgL'] = ((xlo,xhi,dx),"Shower - Average Charge / 3D Length [pix/cm]")

xlo= 0
xhi= 1000
dx = 20
pdf_m['trk_3d_QavgL'] = ((xlo,xhi,dx),"Track - Average Charge / 3D Length [pix/cm]")

xlo= 0
xhi= 1
dx = 0.025
pdf_m['dqds_ratio_01'] = ((xlo,xhi,dx),"dQ/dX Ratio")

xlo= 0
xhi= 200
dx = 5
pdf_m['dqds_diff_01'] = ((xlo,xhi,dx), "dQ/dX Difference [pix/cm]" )



# In[32]:

sig_spectrum_m = {}
bkg_spectrum_m = {}


DRAW=False

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
        SS = os.path.join(BASE_PATH,"ll_dump","%s.pdf" % key)
        print "Dump --> ll_dump/%s" % os.path.basename(SS)
        plt.savefig(SS)
        plt.show()
        

fout = os.path.join(BASE_PATH,"ll_bin","nue_pdfs.root")
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
