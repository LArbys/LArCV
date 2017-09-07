import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ROOT
import numpy as np
import pandas as pd

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

name        = sys.argv[1]
INPUT_FILE  = sys.argv[2]

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
    
if name == "nue" or name == "ncpizero":
    nufilter_df = pd.DataFrame(rn.root2array(INPUT_FILE,treename="NuFilterTree"))
    mc_df       = pd.DataFrame(rn.root2array(INPUT_FILE,treename="MCTree"))
        
    comb_df = comb_df.set_index(rse).join(nufilter_df.set_index(rse),how='outer',lsuffix='',rsuffix='_y').reset_index()
    drop_y(comb_df)    
        
    comb_df = comb_df.set_index(rse).join(mc_df.set_index(rse),how='outer',lsuffix='',rsuffix='_y').reset_index()
    drop_y(comb_df)

comb_df = comb_df.reset_index()
comb_df = comb_df.loc[:,~comb_df.columns.duplicated()]

comb_df['cvtxid'] = 0.0

def func(group):
    group['cvtxid'] = np.arange(0,group['cvtxid'].size)
    return group

comb_df = comb_df.groupby(['run','subrun','event']).apply(func)
comb_cut_df = comb_df.copy()

def track_shower_assumption(df):
    df['trkid'] = df.apply(lambda x : 0 if(x['par1_type']==1) else 1,axis=1)
    df['shrid'] = df.apply(lambda x : 1 if(x['par2_type']==2) else 0,axis=1)
    
    df['trk_frac_avg'] = df.apply(lambda x : x['par1_frac'] if(x['par1_type']==1) else x['par2_frac'],axis=1)
    df['shr_frac_avg'] = df.apply(lambda x : x['par2_frac'] if(x['par2_type']==2) else x['par1_frac'],axis=1)

print
print "@ sample",name
print

comb_cut_df = comb_df.copy()

print "Asking nue assumption"
print "Asking npar==2"
print "Asking in_fiducial==1"
print "Asking pathexists2==1"

comb_cut_df = comb_cut_df.query("npar==2")
track_shower_assumption(comb_cut_df)
comb_cut_df = comb_cut_df.query("par1_type != par2_type")
comb_cut_df = comb_cut_df.query("in_fiducial==1")
comb_cut_df = comb_cut_df.query("pathexists2==1")

comb_cut_df['cosangle3d']=comb_cut_df.apply(lambda x : larocv.CosOpeningAngle(x['par_trunk_pca_theta_estimate_v'][0],
                                                                              x['par_trunk_pca_phi_estimate_v'][0],
                                                                              x['par_trunk_pca_theta_estimate_v'][1],
                                                                              x['par_trunk_pca_phi_estimate_v'][1]),axis=1)

comb_cut_df['angle3d'] = comb_cut_df.apply(lambda x : np.arccos(x['cosangle3d']),axis=1)


comb_cut_df['trk_frac'] = comb_cut_df.apply(lambda x : x['trk_frac_avg'] / x['nplanes_v'][x['trkid']],axis=1) 
comb_cut_df['shr_frac'] = comb_cut_df.apply(lambda x : x['shr_frac_avg'] / x['nplanes_v'][x['shrid']],axis=1) 

comb_cut_df['shr_trunk_pca_theta_estimate'] = comb_cut_df.apply(lambda x : x['par_trunk_pca_theta_estimate_v'][x['shrid']],axis=1) 
comb_cut_df['trk_trunk_pca_theta_estimate'] = comb_cut_df.apply(lambda x : x['par_trunk_pca_theta_estimate_v'][x['trkid']],axis=1) 

comb_cut_df['shr_trunk_pca_cos_theta_estimate'] = comb_cut_df.apply(lambda x : np.cos(x['par_trunk_pca_theta_estimate_v'][x['shrid']]),axis=1) 
comb_cut_df['trk_trunk_pca_cos_theta_estimate'] = comb_cut_df.apply(lambda x : np.cos(x['par_trunk_pca_theta_estimate_v'][x['trkid']]),axis=1) 

comb_cut_df['shr_avg_length'] = comb_cut_df.apply(lambda x : x['length_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
comb_cut_df['trk_avg_length'] = comb_cut_df.apply(lambda x : x['length_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)

comb_cut_df['shr_avg_width'] = comb_cut_df.apply(lambda x : x['width_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
comb_cut_df['trk_avg_width'] = comb_cut_df.apply(lambda x : x['width_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)

comb_cut_df['shr_avg_perimeter'] = comb_cut_df.apply(lambda x : x['perimeter_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
comb_cut_df['trk_avg_perimeter'] = comb_cut_df.apply(lambda x : x['perimeter_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)

comb_cut_df['shr_avg_area'] = comb_cut_df.apply(lambda x : x['area_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
comb_cut_df['trk_avg_area'] = comb_cut_df.apply(lambda x : x['area_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)

comb_cut_df['shr_avg_npixel'] = comb_cut_df.apply(lambda x : x['npixel_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
comb_cut_df['trk_avg_npixel'] = comb_cut_df.apply(lambda x : x['npixel_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)

comb_cut_df['shr_3d_length'] = comb_cut_df.apply(lambda x : x['par_pca_end_len_v'][x['shrid']],axis=1)
comb_cut_df['trk_3d_length'] = comb_cut_df.apply(lambda x : x['par_pca_end_len_v'][x['trkid']],axis=1)

comb_cut_df['shr_3d_QavgL'] = comb_cut_df.apply(lambda x : x['qsum_v'][x['shrid']] / x['par_pca_end_len_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
comb_cut_df['trk_3d_QavgL'] = comb_cut_df.apply(lambda x : x['qsum_v'][x['trkid']] / x['par_pca_end_len_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)

comb_cut_df['shr_triangle_d_max'] = comb_cut_df.apply(lambda x : x['triangle_d_max_v'][x['shrid']],axis=1)
comb_cut_df['trk_triangle_d_max'] = comb_cut_df.apply(lambda x : x['triangle_d_max_v'][x['trkid']],axis=1)

comb_cut_df['shr_mean_pixel_dist'] = comb_cut_df.apply(lambda x : x['mean_pixel_dist_v'][x['shrid']]/x['nplanes_v'][x['shrid']],axis=1)
comb_cut_df['trk_mean_pixel_dist'] = comb_cut_df.apply(lambda x : x['mean_pixel_dist_v'][x['trkid']]/x['nplanes_v'][x['trkid']],axis=1)

comb_cut_df['shr_sigma_pixel_dist'] = comb_cut_df.apply(lambda x : x['sigma_pixel_dist_v'][x['shrid']]/x['nplanes_v'][x['shrid']],axis=1)
comb_cut_df['trk_sigma_pixel_dist'] = comb_cut_df.apply(lambda x : x['sigma_pixel_dist_v'][x['trkid']]/x['nplanes_v'][x['trkid']],axis=1)

comb_cut_df['shr_par_pixel_ratio'] = comb_cut_df.apply(lambda x : x['par_pixel_ratio_v'][x['shrid']],axis=1)
comb_cut_df['trk_par_pixel_ratio'] = comb_cut_df.apply(lambda x : x['par_pixel_ratio_v'][x['trkid']],axis=1) 

comb_cut_df['anglediff0'] = comb_cut_df['anglediff'].values

comb_cut_df['shr_length_min']    = comb_cut_df.apply(lambda x : x['length_min_v'][x['shrid']],axis=1)
comb_cut_df['shr_width_min']     = comb_cut_df.apply(lambda x : x['width_min_v'][x['shrid']],axis=1)
comb_cut_df['shr_perimeter_min'] = comb_cut_df.apply(lambda x : x['perimeter_min_v'][x['shrid']],axis=1)
comb_cut_df['shr_area_min']      = comb_cut_df.apply(lambda x : x['area_min_v'][x['shrid']],axis=1)
comb_cut_df['shr_npixel_min']    = comb_cut_df.apply(lambda x : x['npixel_min_v'][x['shrid']],axis=1)
comb_cut_df['shr_qsum_min']      = comb_cut_df.apply(lambda x : x['qsum_min_v'][x['shrid']],axis=1)

comb_cut_df['trk_length_min']    = comb_cut_df.apply(lambda x : x['length_max_v'][x['trkid']],axis=1)
comb_cut_df['trk_width_min']     = comb_cut_df.apply(lambda x : x['width_max_v'][x['trkid']],axis=1)
comb_cut_df['trk_perimeter_min'] = comb_cut_df.apply(lambda x : x['perimeter_max_v'][x['trkid']],axis=1)
comb_cut_df['trk_area_min']      = comb_cut_df.apply(lambda x : x['area_max_v'][x['trkid']],axis=1)
comb_cut_df['trk_npixel_min']    = comb_cut_df.apply(lambda x : x['npixel_max_v'][x['trkid']],axis=1)
comb_cut_df['trk_qsum_min']      = comb_cut_df.apply(lambda x : x['qsum_max_v'][x['trkid']],axis=1)

comb_cut_df['shr_length_max']    = comb_cut_df.apply(lambda x : x['length_max_v'][x['shrid']],axis=1)
comb_cut_df['shr_width_max']     = comb_cut_df.apply(lambda x : x['width_max_v'][x['shrid']],axis=1)
comb_cut_df['shr_perimeter_max'] = comb_cut_df.apply(lambda x : x['perimeter_max_v'][x['shrid']],axis=1)
comb_cut_df['shr_area_max']      = comb_cut_df.apply(lambda x : x['area_max_v'][x['shrid']],axis=1)
comb_cut_df['shr_npixel_max']    = comb_cut_df.apply(lambda x : x['npixel_max_v'][x['shrid']],axis=1)
comb_cut_df['shr_qsum_max']      = comb_cut_df.apply(lambda x : x['qsum_max_v'][x['shrid']],axis=1)

comb_cut_df['trk_length_max']    = comb_cut_df.apply(lambda x : x['length_max_v'][x['trkid']],axis=1)
comb_cut_df['trk_width_max']     = comb_cut_df.apply(lambda x : x['width_max_v'][x['trkid']],axis=1)
comb_cut_df['trk_perimeter_max'] = comb_cut_df.apply(lambda x : x['perimeter_max_v'][x['trkid']],axis=1)
comb_cut_df['trk_area_max']      = comb_cut_df.apply(lambda x : x['area_max_v'][x['trkid']],axis=1)
comb_cut_df['trk_npixel_max']    = comb_cut_df.apply(lambda x : x['npixel_max_v'][x['trkid']],axis=1)
comb_cut_df['trk_qsum_max']      = comb_cut_df.apply(lambda x : x['qsum_max_v'][x['trkid']],axis=1)

    
#
# Pull distributions & binning for PDFs from ROOT file @ ll_bin/"
#

fin = os.path.join(BASE_PATH,"ll_bin","nue_pdfs.root")
tf_in = ROOT.TFile(fin,"READ")
tf_in.cd()

keys_v = [key.GetName() for key in tf_in.GetListOfKeys()]

sig_spectrum_m = {}
bkg_spectrum_m = {}

for key in keys_v:
    # print "@key=",key
    hist = tf_in.Get(key)
    arr = rn.hist2array(hist,return_edges=True)

    data = arr[0]
    bins = arr[1][0]
    
    assert data.sum() > 0.99999
    dx   = (bins[1] - bins[0]) / 2.0

    centers = (bins + dx)[:-1]

    type_ = key.split("_")[0]
    
    param = None
    if type_ == "sig":
        param = "_".join(key.split("_")[1:])
        sig_spectrum_m[param] = (centers,data)
    elif  type_ == "bkg":
        param = "_".join(key.split("_")[1:])
        bkg_spectrum_m[param] = (centers,data)
    else:
        raise Exception

tf_in.Close()

#
# Assert sig. and bkg. read from file correctly
#
for key in sig_spectrum_m.keys():
    assert key in sig_spectrum_m.keys(), "key=%s missing from sig"%key
    assert key in bkg_spectrum_m.keys(), "key=%s missing from bkg"%key
    assert sig_spectrum_m[key][0].size == bkg_spectrum_m[key][0].size
    assert sig_spectrum_m[key][1].size == bkg_spectrum_m[key][1].size

for key in bkg_spectrum_m.keys():
    assert key in sig_spectrum_m.keys(), "key=%s missing from sig"%key
    assert key in bkg_spectrum_m.keys(), "key=%s missing from bkg"%key
    assert sig_spectrum_m[key][0].size == bkg_spectrum_m[key][0].size
    assert sig_spectrum_m[key][1].size == bkg_spectrum_m[key][1].size

#
# Define, given value, how to get prob. from PDF
#
def nearest_id(spectrum,value):
    return np.argmin(np.abs(spectrum - value))

def nearest_id_v(spectrum_v,value_v):
    return np.array([np.argmin(np.abs(spectrum[0] - value)) for spectrum, value in zip(spectrum_v,value_v)])
    
#
# Define the LL
#
def LL(row):
    cols = row[sig_spectrum_m.keys()]
    sig_res = nearest_id_v(sig_spectrum_m.values(),cols.values)
    bkg_res = nearest_id_v(bkg_spectrum_m.values(),cols.values)
    
    sig_res = np.array([spectrum[1][v] for spectrum,v in zip(sig_spectrum_m.values(),sig_res)])
    bkg_res = np.array([spectrum[1][v] for spectrum,v in zip(bkg_spectrum_m.values(),bkg_res)])
    
    LL = np.log( sig_res / (sig_res + bkg_res) )

    return LL.sum()

#
# Apply the LL
#
print "Applying LL"
k0 = comb_cut_df.apply(LL,axis=1)
comb_cut_df['LL']=k0

#
# Choose the vertex @ event with the highest LL
#
print "Choosing vertex with max LL"
passed_df = comb_cut_df.copy()
passed_df = passed_df.sort_values(["LL"],ascending=False).groupby(rse).head(1)

OUT=os.path.join("ll_bin","%s_all.pkl" % name)
comb_df.to_pickle(OUT)
print "Store",OUT
print
OUT=os.path.join("ll_bin","%s_post_nue.pkl" % name)
comb_cut_df.to_pickle(OUT)
print "Store",OUT
print
OUT=os.path.join("ll_bin","%s_post_LL.pkl" % name)
passed_df.to_pickle(OUT)
print "Store",OUT
print
print "Done"
