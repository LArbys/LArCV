import os, sys
import matplotlib
matplotlib.use('Agg')
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
sample_file0 = "comb_ana_nue.root"

sample_name1 = "cosmic"
sample_file1 = "comb_ana_cosmic_no_stopmu.root"

for name,file_ in [(sample_name0,sample_file0),
                   (sample_name1,sample_file1)]:
    
    INPUT_FILE  = file_
    
    print "@FILE=",INPUT_FILE
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

    comb_df = comb_df.reset_index()
    event_vertex_df   = pd.DataFrame(rn.root2array(INPUT_FILE,treename="EventVertexTree"))

    def drop_y(df):
        to_drop = [x for x in df if x.endswith('_y')]
        df.drop(to_drop, axis=1, inplace=True)
        
    comb_df = comb_df.set_index(rse).join(event_vertex_df.set_index(rse),how='outer',lsuffix='',rsuffix='_y').reset_index()
    drop_y(comb_df)
    
    if name == "nue":
        nufilter_df       = pd.DataFrame(rn.root2array(INPUT_FILE,treename="NuFilterTree"))
        mc_df             = pd.DataFrame(rn.root2array(INPUT_FILE,treename="MCTree"))
        
        comb_df = comb_df.set_index(rse).join(nufilter_df.set_index(rse),how='outer',lsuffix='',rsuffix='_y').reset_index()
        drop_y(comb_df)    
        
        comb_df = comb_df.set_index(rse).join(mc_df.set_index(rse),how='outer',lsuffix='',rsuffix='_y').reset_index()
        drop_y(comb_df)

    #
    # Store vertex wise data frame
    #
    comb_df = comb_df.reset_index()
    comb_df = comb_df.loc[:,~comb_df.columns.duplicated()]
    
    comb_df['cvtxid'] = 0.0
    
    def func(group):
        group['cvtxid'] = np.arange(0,group['cvtxid'].size)
        return group

    comb_df = comb_df.groupby(['run','subrun','event']).apply(func)
    
    dfs[name] = comb_df.copy()

dfs['nue'] = dfs['nue'].query("scedr<5")

def track_shower_assumption(df):
    df['trkid'] = df.apply(lambda x : 0 if(x['par1_type']==1) else 1,axis=1)
    df['shrid'] = df.apply(lambda x : 1 if(x['par2_type']==2) else 0,axis=1)
    
    df['trk_frac_avg'] = df.apply(lambda x : x['par1_frac'] if(x['par1_type']==1) else x['par2_frac'],axis=1)
    df['shr_frac_avg'] = df.apply(lambda x : x['par2_frac'] if(x['par2_type']==2) else x['par1_frac'],axis=1)

ts_mdf_m = {}
for name, comb_df in dfs.copy().iteritems():
print
    print "@ sample",name
    print
    
    ts_mdf = comb_df.copy()

    print "Asking nue assumption"
    print "Asking npar==2"
    print "Asking in_fiducial==1"
    print "Asking pathexists2==1"

    ts_mdf = ts_mdf.query("npar==2")
    track_shower_assumption(ts_mdf)
    ts_mdf = ts_mdf.query("par1_type != par2_type")
    ts_mdf = ts_mdf.query("in_fiducial==1")
    ts_mdf = ts_mdf.query("pathexists2==1")
    
    #
    # SSNet Fraction
    #
    ts_mdf['trk_frac'] = ts_mdf.apply(lambda x : x['trk_frac_avg'] / x['nplanes_v'][x['trkid']],axis=1) 
    ts_mdf['shr_frac'] = ts_mdf.apply(lambda x : x['shr_frac_avg'] / x['nplanes_v'][x['shrid']],axis=1) 
    
    #
    # PCA
    #
    
    ts_mdf['cosangle3d']=ts_mdf.apply(lambda x : larocv.CosOpeningAngle(x['par_trunk_pca_theta_estimate_v'][0],
                                                                        x['par_trunk_pca_phi_estimate_v'][0],
                                                                        x['par_trunk_pca_theta_estimate_v'][1],
                                                                        x['par_trunk_pca_phi_estimate_v'][1]),axis=1)
    
    ts_mdf['angle3d'] = ts_mdf.apply(lambda x : np.arccos(x['cosangle3d']),axis=1)
    
    
    ts_mdf['shr_trunk_pca_theta_estimate'] = ts_mdf.apply(lambda x : x['par_trunk_pca_theta_estimate_v'][x['shrid']],axis=1) 
    ts_mdf['trk_trunk_pca_theta_estimate'] = ts_mdf.apply(lambda x : x['par_trunk_pca_theta_estimate_v'][x['trkid']],axis=1) 
    
    ts_mdf['shr_trunk_pca_cos_theta_estimate'] = ts_mdf.apply(lambda x : np.cos(x['par_trunk_pca_theta_estimate_v'][x['shrid']]),axis=1) 
    ts_mdf['trk_trunk_pca_cos_theta_estimate'] = ts_mdf.apply(lambda x : np.cos(x['par_trunk_pca_theta_estimate_v'][x['trkid']]),axis=1) 

    
    #
    # 3D
    #
    ts_mdf['shr_3d_length'] = ts_mdf.apply(lambda x : x['par_pca_end_len_v'][x['shrid']],axis=1)
    ts_mdf['trk_3d_length'] = ts_mdf.apply(lambda x : x['par_pca_end_len_v'][x['trkid']],axis=1)

    ts_mdf['shr_3d_QavgL'] = ts_mdf.apply(lambda x : x['qsum_v'][x['shrid']] / x['par_pca_end_len_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['trk_3d_QavgL'] = ts_mdf.apply(lambda x : x['qsum_v'][x['trkid']] / x['par_pca_end_len_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)

    #
    # Max deflection
    #
    ts_mdf['shr_triangle_d_max'] = ts_mdf.apply(lambda x : x['triangle_d_max_v'][x['shrid']],axis=1)
    ts_mdf['trk_triangle_d_max'] = ts_mdf.apply(lambda x : x['triangle_d_max_v'][x['trkid']],axis=1)
    
    #
    # Mean pixel dist from 2D PCA
    #
    ts_mdf['shr_mean_pixel_dist'] = ts_mdf.apply(lambda x : x['mean_pixel_dist_v'][x['shrid']]/x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['shr_mean_pixel_dist_max'] = ts_mdf.apply(lambda x : x['mean_pixel_dist_max_v'][x['shrid']],axis=1)
    ts_mdf['shr_mean_pixel_dist_min'] = ts_mdf.apply(lambda x : x['mean_pixel_dist_min_v'][x['shrid']],axis=1)
    ts_mdf['shr_mean_pixel_dist_ratio'] = ts_mdf.apply(lambda x : x['mean_pixel_dist_min_v'][x['shrid']] / x['mean_pixel_dist_max_v'][x['shrid']],axis=1)
    
    ts_mdf['trk_mean_pixel_dist'] = ts_mdf.apply(lambda x : x['mean_pixel_dist_v'][x['trkid']]/x['nplanes_v'][x['trkid']],axis=1)
    ts_mdf['trk_mean_pixel_dist_max'] = ts_mdf.apply(lambda x : x['mean_pixel_dist_max_v'][x['trkid']],axis=1)
    ts_mdf['trk_mean_pixel_dist_min'] = ts_mdf.apply(lambda x : x['mean_pixel_dist_min_v'][x['trkid']],axis=1)
    ts_mdf['trk_mean_pixel_dist_ratio'] = ts_mdf.apply(lambda x : x['mean_pixel_dist_min_v'][x['trkid']] / x['mean_pixel_dist_max_v'][x['trkid']],axis=1)     

    #
    # Sigma pixel dist from 2D PCA
    #
    ts_mdf['shr_sigma_pixel_dist']       = ts_mdf.apply(lambda x : x['sigma_pixel_dist_v'][x['shrid']]/x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['shr_sigma_pixel_dist_max']   = ts_mdf.apply(lambda x : x['sigma_pixel_dist_max_v'][x['shrid']],axis=1)
    ts_mdf['shr_sigma_pixel_dist_min']   = ts_mdf.apply(lambda x : x['sigma_pixel_dist_min_v'][x['shrid']],axis=1)
    ts_mdf['shr_sigma_pixel_dist_ratio'] = ts_mdf.apply(lambda x : x['sigma_pixel_dist_min_v'][x['shrid']] / x['sigma_pixel_dist_max_v'][x['shrid']],axis=1)
    
    ts_mdf['trk_sigma_pixel_dist']       = ts_mdf.apply(lambda x : x['sigma_pixel_dist_v'][x['trkid']]/x['nplanes_v'][x['trkid']],axis=1)
    ts_mdf['trk_sigma_pixel_dist_max']   = ts_mdf.apply(lambda x : x['sigma_pixel_dist_max_v'][x['trkid']],axis=1)
    ts_mdf['trk_sigma_pixel_dist_min']   = ts_mdf.apply(lambda x : x['sigma_pixel_dist_min_v'][x['trkid']],axis=1)
    ts_mdf['trk_sigma_pixel_dist_ratio'] = ts_mdf.apply(lambda x : x['sigma_pixel_dist_min_v'][x['trkid']] / x['sigma_pixel_dist_max_v'][x['trkid']],axis=1)    

    #
    # Ratio of # num pixels
    #
    ts_mdf['shr_par_pixel_ratio'] = ts_mdf.apply(lambda x : x['par_pixel_ratio_v'][x['shrid']],axis=1)
    ts_mdf['trk_par_pixel_ratio'] = ts_mdf.apply(lambda x : x['par_pixel_ratio_v'][x['trkid']],axis=1) 

    #
    # 2D angle difference @ vertex
    #
    ts_mdf['anglediff0'] = ts_mdf['anglediff'].values 

    #
    # 2D length
    #
    ts_mdf['shr_avg_length']   = ts_mdf.apply(lambda x : x['length_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['shr_length_min']   = ts_mdf.apply(lambda x : x['length_min_v'][x['shrid']],axis=1)
    ts_mdf['shr_length_max']   = ts_mdf.apply(lambda x : x['length_max_v'][x['shrid']],axis=1)
    ts_mdf['shr_length_ratio'] = ts_mdf.apply(lambda x : x['length_min_v'][x['shrid']] / x['length_max_v'][x['shrid']],axis=1)
    
    ts_mdf['trk_avg_length']   = ts_mdf.apply(lambda x : x['length_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)
    ts_mdf['trk_length_min']   = ts_mdf.apply(lambda x : x['length_min_v'][x['trkid']],axis=1)
    ts_mdf['trk_length_max']   = ts_mdf.apply(lambda x : x['length_max_v'][x['trkid']],axis=1)
    ts_mdf['trk_length_ratio'] = ts_mdf.apply(lambda x : x['length_min_v'][x['trkid']] / x['length_max_v'][x['trkid']],axis=1)
    
    #
    # 2D width
    #
    ts_mdf['shr_avg_width']   = ts_mdf.apply(lambda x : x['width_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['shr_width_min']   = ts_mdf.apply(lambda x : x['width_min_v'][x['shrid']],axis=1)
    ts_mdf['shr_width_max']   = ts_mdf.apply(lambda x : x['width_max_v'][x['shrid']],axis=1)
    ts_mdf['shr_width_ratio'] = ts_mdf.apply(lambda x : x['width_min_v'][x['shrid']] / x['width_max_v'][x['shrid']],axis=1)

    ts_mdf['trk_avg_width']   = ts_mdf.apply(lambda x : x['width_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)
    ts_mdf['trk_width_max']   = ts_mdf.apply(lambda x : x['width_max_v'][x['trkid']],axis=1)
    ts_mdf['trk_width_min']   = ts_mdf.apply(lambda x : x['width_max_v'][x['trkid']],axis=1)
    ts_mdf['trk_width_ratio'] = ts_mdf.apply(lambda x : x['width_min_v'][x['trkid']] / x['width_max_v'][x['trkid']],axis=1)

    #
    # 2D perimeter
    #
    ts_mdf['shr_avg_perimeter'] = ts_mdf.apply(lambda x : x['perimeter_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['shr_perimeter_min'] = ts_mdf.apply(lambda x : x['perimeter_min_v'][x['shrid']],axis=1)
    ts_mdf['shr_perimeter_max'] = ts_mdf.apply(lambda x : x['perimeter_max_v'][x['shrid']],axis=1)
    ts_mdf['shr_perimeter_ratio'] = ts_mdf.apply(lambda x : x['perimeter_min_v'][x['shrid']] / x['perimeter_max_v'][x['shrid']],axis=1)
    
    ts_mdf['trk_avg_perimeter'] = ts_mdf.apply(lambda x : x['perimeter_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)
    ts_mdf['trk_perimeter_min'] = ts_mdf.apply(lambda x : x['perimeter_max_v'][x['trkid']],axis=1)
    ts_mdf['trk_perimeter_max'] = ts_mdf.apply(lambda x : x['perimeter_max_v'][x['trkid']],axis=1)
    ts_mdf['trk_perimeter_ratio'] = ts_mdf.apply(lambda x : x['perimeter_min_v'][x['trkid']] / x['perimeter_max_v'][x['trkid']],axis=1)

    #
    # 2D area
    #
    ts_mdf['shr_avg_area'] = ts_mdf.apply(lambda x : x['area_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['shr_area_min'] = ts_mdf.apply(lambda x : x['area_min_v'][x['shrid']],axis=1)
    ts_mdf['shr_area_max'] = ts_mdf.apply(lambda x : x['area_max_v'][x['shrid']],axis=1)
    ts_mdf['shr_area_ratio'] = ts_mdf.apply(lambda x : x['area_min_v'][x['shrid']] / x['area_max_v'][x['shrid']],axis=1)
    
    ts_mdf['trk_avg_area'] = ts_mdf.apply(lambda x : x['area_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)
    ts_mdf['trk_area_min'] = ts_mdf.apply(lambda x : x['area_max_v'][x['trkid']],axis=1)
    ts_mdf['trk_area_max'] = ts_mdf.apply(lambda x : x['area_max_v'][x['trkid']],axis=1)
    ts_mdf['trk_area_ratio'] = ts_mdf.apply(lambda x : x['area_min_v'][x['trkid']] / x['area_max_v'][x['trkid']],axis=1)

    #
    # N pixel
    #
    ts_mdf['shr_avg_npixel'] = ts_mdf.apply(lambda x : x['npixel_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['shr_npixel_min'] = ts_mdf.apply(lambda x : x['npixel_min_v'][x['shrid']],axis=1)
    ts_mdf['shr_npixel_max'] = ts_mdf.apply(lambda x : x['npixel_max_v'][x['shrid']],axis=1)
    ts_mdf['shr_npixel_ratio'] = ts_mdf.apply(lambda x : x['npixel_min_v'][x['shrid']] / x['npixel_max_v'][x['shrid']],axis=1)
    
    ts_mdf['trk_avg_npixel'] = ts_mdf.apply(lambda x : x['npixel_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)
    ts_mdf['trk_npixel_min'] = ts_mdf.apply(lambda x : x['npixel_max_v'][x['trkid']],axis=1)
    ts_mdf['trk_npixel_max'] = ts_mdf.apply(lambda x : x['npixel_max_v'][x['trkid']],axis=1)
    ts_mdf['trk_npixel_ratio'] = ts_mdf.apply(lambda x : x['npixel_min_v'][x['trkid']] / x['npixel_max_v'][x['trkid']],axis=1)

    #
    # Q sum
    #
    ts_mdf['shr_avg_qsum']   = ts_mdf.apply(lambda x : x['qsum_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    ts_mdf['shr_qsum_min']   = ts_mdf.apply(lambda x : x['qsum_min_v'][x['shrid']],axis=1)
    ts_mdf['shr_qsum_max']   = ts_mdf.apply(lambda x : x['qsum_max_v'][x['shrid']],axis=1)
    ts_mdf['shr_qsum_ratio'] = ts_mdf.apply(lambda x : x['qsum_min_v'][x['shrid']] / x['qsum_max_v'][x['shrid']],axis=1)
    
    ts_mdf['trk_avg_qsum']   = ts_mdf.apply(lambda x : x['qsum_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)
    ts_mdf['trk_qsum_min']   = ts_mdf.apply(lambda x : x['qsum_max_v'][x['trkid']],axis=1)
    ts_mdf['trk_qsum_max']   = ts_mdf.apply(lambda x : x['qsum_max_v'][x['trkid']],axis=1)
    ts_mdf['trk_qsum_ratio'] = ts_mdf.apply(lambda x : x['qsum_min_v'][x['trkid']] / x['qsum_max_v'][x['trkid']],axis=1)

    #
    #
    #
    ts_mdf_m[name] = ts_mdf.copy()



#
# distributions & binning for likelihood
#
import collections
pdf_m = collections.OrderedDict()

# xlo= 0.0
# xhi= 40.0
# dx = 2
# pdf_m['shr_triangle_d_max'] = ((xlo,xhi,dx),"Shower - Max 2D Deflection [pix]")

# xlo= 0.0
# xhi= 40.0
# dx = 2
# pdf_m['trk_triangle_d_max'] = ((xlo,xhi,dx),"Track - Max 2D Deflection [pix]")

#
#
#
xlo= 0.0
xhi= 10.0
dx = 0.2
pdf_m['shr_mean_pixel_dist'] = ((xlo,xhi,dx),"Shower - Mean Distance from 2D PCA [pix]")

# xlo= 0.0
# xhi= 10.0
# dx = 0.2
# pdf_m['trk_mean_pixel_dist'] = ((xlo,xhi,dx),"Track - Mean Distance from 2D PCA [pix]")

# xlo= 0.0
# xhi= 10.0
# dx = 0.2
# pdf_m['shr_mean_pixel_dist_max'] = ((xlo,xhi,dx),"Shower - Max Mean Distance from 2D PCA [pix]")

# xlo= 0.0
# xhi= 10.0
# dx = 0.2
# pdf_m['trk_mean_pixel_dist_max'] = ((xlo,xhi,dx),"Track - Max Mean Distance from 2D PCA [pix]")


#
#
#
xlo= 0.0
xhi= 10.0
dx = 0.2
pdf_m['shr_sigma_pixel_dist'] = ((xlo,xhi,dx),"Shower - Sigma Distance from 2D PCA [pix]")

# xlo= 0.0
# xhi= 10.0
# dx = 0.2
# pdf_m['trk_sigma_pixel_dist'] = ((xlo,xhi,dx),"Track - Sigma Distance from 2D PCA [pix]")

# xlo= 0.0
# xhi= 10.0
# dx = 0.2
# pdf_m['shr_sigma_pixel_dist_max'] = ((xlo,xhi,dx),"Shower - Max Sigma Distance from 2D PCA [pix]")

# xlo= 0.0
# xhi= 10.0
# dx = 0.2
# pdf_m['trk_sigma_pixel_dist_max'] = ((xlo,xhi,dx),"Track - Max Sigma Distance from 2D PCA [pix]")


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

#xlo= 0
#xhi= 3.14159
#dx = 3.14159/40.0
#pdf_m['angle3d'] = ((xlo,xhi,dx),"3D Opening Angle")

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

#xlo= 0
#xhi= 3.14159
#dx = 3.14159/40.0
#pdf_m['trk_trunk_pca_theta_estimate'] = ((xlo,xhi,dx),"Track - 3D Beam Angle")

#xlo= 0
#xhi= 3.14159
#dx = 3.14159/40.0
#pdf_m['shr_trunk_pca_theta_estimate'] = ((xlo,xhi,dx),"Shower - 3D Beam Angle")


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

# xlo= 0
# xhi= 300
# dx = 5
# pdf_m['shr_length_min'] = ((xlo,xhi,dx),"Shower - Min 2D Length [pix]")

# xlo= 0
# xhi= 300
# dx = 5
# pdf_m['trk_length_min'] = ((xlo,xhi,dx),"Track - Min 2D Length [pix]")

# xlo= 0
# xhi= 300
# dx = 5
# pdf_m['shr_length_max'] = ((xlo,xhi,dx),"Shower - Max 2D Length [pix]")

# xlo= 0
# xhi= 300
# dx = 5
# pdf_m['trk_length_max'] = ((xlo,xhi,dx),"Track - Max 2D Length [pix]")


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

# xlo= 0
# xhi= 600
# dx = 10
# pdf_m['shr_area_min'] = ((xlo,xhi,dx),"Shower - Min 2D Area [pix^2]")

# xlo= 0
# xhi= 600
# dx = 10
# pdf_m['trk_area_min'] = ((xlo,xhi,dx),"Track - Min 2D Area [pix^2]")

# xlo= 0
# xhi= 600
# dx = 10
# pdf_m['shr_area_max'] = ((xlo,xhi,dx),"Shower - Max 2D Area [pix^2]")

# xlo= 0
# xhi= 600
# dx = 10
# pdf_m['trk_area_max'] = ((xlo,xhi,dx),"Track - Max 2D Area [pix^2]")

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

# xlo= 0
# xhi= 50
# dx = 2
# pdf_m['shr_width_min'] = ((xlo,xhi,dx),"Shower - Min 2D Width [px]")

# xlo= 0
# xhi= 50
# dx = 2
# pdf_m['trk_width_min'] = ((xlo,xhi,dx),"Track - Min 2D Width [px]")

# xlo= 0
# xhi= 50
# dx = 2
# pdf_m['shr_width_max'] = ((xlo,xhi,dx),"Shower - Max 2D Width [px]")

# xlo= 0
# xhi= 50
# dx = 2
# pdf_m['trk_width_max'] = ((xlo,xhi,dx),"Track - Max 2D Width [px]")


#
# npixel
#
# xlo= 0
# xhi= 1000
# dx = 20
# pdf_m['shr_avg_npixel'] = ((xlo,xhi,dx),"Shower - Average Num. Pixel")

# xlo= 0
# xhi= 1000
# dx = 20
# pdf_m['trk_avg_npixel'] = ((xlo,xhi,dx),"Track - Average Num. Pixel")

# xlo= 0
# xhi= 600
# dx = 10
# pdf_m['shr_npixel_min'] = ((xlo,xhi,dx),"Shower - Min Num. Pixel")

# xlo= 0
# xhi= 600
# dx = 10
# pdf_m['trk_npixel_min'] = ((xlo,xhi,dx),"Track - Min Num. Pixel")

# xlo= 0
# xhi= 600
# dx = 10
# pdf_m['shr_npixel_max'] = ((xlo,xhi,dx),"Shower - Max Num. Pixel")

# xlo= 0
# xhi= 600
# dx = 10
# pdf_m['trk_npixel_max'] = ((xlo,xhi,dx),"Track - Max Num. Pixel")

#
# Perimeter
#
#xlo= 0
#xhi= 300
#dx = 5
#pdf_m['shr_avg_perimeter'] = ((xlo,xhi,dx),"Shower - Average 2D Perimeter [pix]")

#xlo= 0
#xhi= 300
#dx = 5
#pdf_m['trk_avg_perimeter'] = ((xlo,xhi,dx),"Track - Average 2D Perimeter [pix]")

# xlo= 0
# xhi= 300
# dx = 5
# pdf_m['shr_perimeter_min'] = ((xlo,xhi,dx),"Shower - Min 2D Perimeter [pix]")

# xlo= 0
# xhi= 300
# dx = 5
# pdf_m['trk_perimeter_min'] = ((xlo,xhi,dx),"Track - Min 2D Perimeter [pix]")

# xlo= 0
# xhi= 300
# dx = 5
# pdf_m['shr_perimeter_min'] = ((xlo,xhi,dx),"Shower - Max 2D Perimeter [pix]")

# xlo= 0
# xhi= 300
# dx = 5
# pdf_m['trk_perimeter_min'] = ((xlo,xhi,dx),"Track - Max 2D Perimeter [pix]")

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


# #Length

# xlo= 0
# xhi= 1.0
# dx = 0.025
# pdf_m['shr_length_ratio'] = ((xlo,xhi,dx),"shr_length_ratio")

# xlo= 0
# xhi= 1.0
# dx = 0.025
# pdf_m['trk_length_ratio'] = ((xlo,xhi,dx),"trk_length_ratio")


# #Width
# xlo= 0
# xhi= 1.0
# dx = 0.025
# pdf_m['shr_width_ratio'] = ((xlo,xhi,dx),"shr_width_ratio")

# xlo= 0
# xhi= 1.0
# dx = 0.025
# pdf_m['trk_width_ratio'] = ((xlo,xhi,dx),"trk_width_ratio")


# #Area
# xlo= 0
# xhi= 1.0
# dx = 0.025
# pdf_m['shr_area_ratio'] = ((xlo,xhi,dx),"shr_area_ratio")

# xlo= 0
# xhi= 1.0
# dx = 0.025
# pdf_m['trk_area_ratio'] = ((xlo,xhi,dx),"trk_area_ratio")

#qsum
# xlo= 0
# xhi= 100000
# dx = 1000
# pdf_m['shr_qsum_max'] = ((xlo,xhi,dx),"shr_qsum_max")

# xlo= 0
# xhi= 100000
# dx = 1000
# pdf_m['trk_qsum_max'] = ((xlo,xhi,dx),"trk_qsum_max")

# #area
# xlo= 0
# xhi= 1.0
# dx = 0.025
# pdf_m['shr_perimeter_ratio'] = ((xlo,xhi,dx),"shr_perimeter_ratio")

# xlo= 0
# xhi= 1.0
# dx = 0.025
# pdf_m['trk_perimeter_ratio'] = ((xlo,xhi,dx),"trk_perimeter_ratio")

# #area
# xlo= 0
# xhi= 1.0
# dx = 0.025
# pdf_m['shr_npixel_ratio'] = ((xlo,xhi,dx),"shr_npixel_ratio")

# xlo= 0
# xhi= 1.0
# dx = 0.025
# pdf_m['trk_npixel_ratio'] = ((xlo,xhi,dx),"trk_npixel_ratio")

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
