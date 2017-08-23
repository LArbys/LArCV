import os, sys
import matplotlib
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
dfs[name] = comb_df.copy()

#
# Event wise Trees
#
event_vertex_df   = pd.DataFrame(rn.root2array(INPUT_FILE,treename="EventVertexTree"))
# mc_df             = pd.DataFrame(rn.root2array(INPUT_FILE,treename="MCTree"))

edfs[name] = event_vertex_df.copy()
# mdfs[name] = mc_df.copy()

print
print "@ Sample:",name,"& # good croi is:",event_vertex_df.query("good_croi_ctr>0").index.size
print "total events: ", event_vertex_df.index.size
print 
    
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


# In[20]:

ts_mdf_m = {}

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


print "Initial stats..."
for name, comb_df in dfs.iteritems():
    print "@ sample",name

    print
    print "N input vertices....",comb_df.index.size
    print "N input events......",len(comb_df.groupby(rse))
    print "N pre-cut vertices..",ts_mdf_m[name].index.size
    print "N pre-cut events....",len(ts_mdf_m[name].groupby(rse))
    print

    scedr = 5
    print "Good vertex scedr < 5"
    print "N input vertices....",comb_df.query("scedr<@scedr").index.size
    print "N input events......",len(comb_df.query("scedr<@scedr").groupby(rse))
    print "N pre-cut vertices..",ts_mdf_m[name].query("scedr<@scedr").index.size
    print "N pre-cut events....",len(ts_mdf_m[name].query("scedr<@scedr").groupby(rse))
    print

    print "Bad vertex scedr > 5"
    print "N input vertices....",comb_df.query("scedr>@scedr").index.size
    print "N input events......",len(comb_df.query("scedr>@scedr").groupby(rse))
    print "N pre-cut vertices..",ts_mdf_m[name].query("scedr>@scedr").index.size
    print "N pre-cut events....",len(ts_mdf_m[name].query("scedr>@scedr").groupby(rse))
    print


#
# distributions & binning for likelihood
#

fin = os.path.join(BASE_PATH,"ll_bin","nue_pdfs.root")
tf = ROOT.TFile(fin,"READ")
tf.cd()

keys_v = [key.GetName() for key in tf.GetListOfKeys()]

sig_spectrum_m = {}
bkg_spectrum_m = {}

for key in keys_v:
    # print "@key=",key
    hist = tf.Get(key)
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

tf.Close()
        
def nearest_id(spectrum,value):
    return np.argmin(np.abs(spectrum - value))

def nearest_id_v(spectrum_v,value_v):
    return np.array([np.argmin(np.abs(spectrum[0] - value)) for spectrum, value in zip(spectrum_v,value_v)])

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
    
    
def LL(row):
    cols = row[sig_spectrum_m.keys()]
    sig_res = nearest_id_v(sig_spectrum_m.values(),cols.values)
    bkg_res = nearest_id_v(bkg_spectrum_m.values(),cols.values)
    
    sig_res = np.array([spectrum[1][v] for spectrum,v in zip(sig_spectrum_m.values(),sig_res)])
    bkg_res = np.array([spectrum[1][v] for spectrum,v in zip(bkg_spectrum_m.values(),bkg_res)])
    
    LL = np.log( sig_res / (sig_res + bkg_res) )

    return LL.sum()
    
k0 = ts_mdf_m[name].apply(LL,axis=1)
ts_mdf_m[name]['LL']=k0


passed = ts_mdf_m[name].query("LL>-16.25")

print "Final stats..."
for name, comb_df in dfs.iteritems():
    print "@ sample",name
    
    print
    print "N post LL vertices..",passed.index.size
    print "N post LL events....",len(passed.groupby(rse))
    print
    
    scedr = 5
    print
    print "Good vertex scedr < 5"
    print "N post LL vertices..",passed.query("scedr<@scedr").index.size
    print "N post LL events....",len(passed.query("scedr<@scedr").groupby(rse))
    print

    print
    print "Bad vertex scedr > 5"
    print "N post LL vertices..",passed.query("scedr>@scedr").index.size
    print "N post LL events....",len(passed.query("scedr>@scedr").groupby(rse))
    print
