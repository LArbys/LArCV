import numpy as np
import pandas as pd
from larocv import larocv
import ROOT
import root_numpy as rn
from common import *


def initialize_rst(VTX_DF,ST_DF,TRUE_DF):

    comb_df = pd.DataFrame()
    
    ana_locv_df = pd.read_pickle(VTX_DF)
    ana_locv_df.query("num_vertex>0",inplace=True)

    ana_locv_df.drop(['vtxid'],axis=1,inplace=True)
    ana_locv_df.rename(columns={'cvtxid' : 'vtxid'},inplace=True)
    ana_locv_df.set_index(RSEV,inplace=True)
    ana_locv_df = ana_locv_df.add_prefix('locv_')

    ana_st_df = pd.read_pickle(ST_DF)
    ana_st_df.set_index(RSEV,inplace=True)

    print "ana_locv_df.index.size=",ana_locv_df.index.size
    print "ana_st_df.index.size=",ana_st_df.index.size

    df_v    = [ana_locv_df,ana_st_df]
    comb_df = pd.concat(df_v,axis=1,join_axes=[df_v[0].index])

    comb_df.reset_index(inplace=True)

    print "comb_df.index.size=",comb_df.index.size
    
    ana_true_df = pd.read_pickle(TRUE_DF)
    
    print "ana_true_df.index.size=",ana_true_df.index.size

    ana_true_df.set_index(RSE,inplace=True)
    comb_df.set_index(RSE,inplace=True)
    
    comb_df = comb_df.join(ana_true_df,how='outer',lsuffix='',rsuffix='_y')
    drop_y(comb_df)

    comb_df.reset_index(inplace=True)

    print "now comb_df.index.size=",comb_df.index.size

    assert len(comb_df.groupby(RSE)) == int(ana_true_df.index.size)

    return comb_df


def initialize_st(SHR_ANA1,
                  SHR_TRUTH,
                  TRK_ANA1,
                  TRK_ANA2,
                  TRK_TRUTH,
                  TRK_PGRPH):

    comb_df = pd.DataFrame()
    
    match_shr_df = None
    match_trk_df = None

    isdata = False
    try:
        match_shr_df  = pd.DataFrame(rn.root2array(SHR_TRUTH,treename="ShowerTruthMatch"))
        match_trk_df  = pd.DataFrame(rn.root2array(TRK_TRUTH,treename="TrackTruthMatch"))

        match_shr_df.set_index(RSEV,inplace=True)
        match_trk_df.set_index(RSEV,inplace=True)

        match_shr_df  = match_shr_df.add_prefix("mchshr_")
        match_trk_df  = match_trk_df.add_prefix("mchtrk_")

        print "match_shr_df.index.size=",match_shr_df.index.size
        print "match_trk_df.index.size=",match_trk_df.index.size

    except IOError:
        isdata = True

    pgraph_trk_df = pd.DataFrame(rn.root2array(TRK_PGRPH,treename="TrackPGraphMatch"))
    ana_shr_df    = pd.DataFrame(rn.root2array(SHR_ANA1))
    ana_trk1_df   = pd.DataFrame(rn.root2array(TRK_ANA1))
    ana_trk2_df   = pd.DataFrame(rn.root2array(TRK_ANA2))

    ana_shr_df.rename(columns={'vtx_id': 'vtxid'}, inplace=True)
    ana_trk2_df.rename(columns={'vtx_id': 'vtxid'}, inplace=True)

    pgraph_trk_df.set_index(RSEV,inplace=True)
    ana_shr_df.set_index(RSEV,inplace=True)
    ana_trk1_df.set_index(RSEV,inplace=True)
    ana_trk2_df.set_index(RSEV,inplace=True)
    
    pgraph_trk_df = pgraph_trk_df.add_prefix("pgtrk_")
    ana_shr_df    = ana_shr_df.add_prefix("anashr_")
    ana_trk1_df   = ana_trk1_df.add_prefix("anatrk1_")
    ana_trk2_df   = ana_trk2_df.add_prefix("anatrk2_")

    print "pgraph_trk_df.index.size=",pgraph_trk_df.index.size
    print "ana_shr_df.index.size=",ana_shr_df.index.size
    print "ana_trk1_df.index.size=",ana_trk1_df.index.size
    print "ana_trk2_df.index.size=",ana_trk2_df.index.size

    df_v = []

    if isdata==False:
        df_v = [ana_shr_df,ana_trk1_df,ana_trk2_df,pgraph_trk_df,match_shr_df,match_trk_df]
    else:
        df_v = [ana_shr_df,ana_trk1_df,ana_trk2_df,pgraph_trk_df]

    comb_df = pd.concat(df_v,axis=1,join_axes=[df_v[0].index])
    comb_df.reset_index(inplace=True)
    
    return comb_df


def initialize_truth(input_file,data=False):
    print "Loading event TTrees..."
    nufilter_df     = pd.DataFrame(rn.root2array(input_file,treename="NuFilterTree"))
    mc_df           = pd.DataFrame(rn.root2array(input_file,treename="MCTree"))

    print "Reindex..."
    nufilter_df.set_index(rse,inplace=True)
    mc_df.set_index(rse,inplace=True)
    print "...done"
    
    print "Joining mcdf..."
    nufilter_df = nufilter_df.join(mc_df,how='outer',lsuffix='',rsuffix='_y')
    print "...dropping"
    drop_y(nufilter_df)
    print "...dropped"

    print "Reindex..."
    nufilter_df.reset_index(inplace=True)
    print "...done"

    return nufilter_df

    

def initialize_df(input_file,data=False):

    print "Loading vertex TTrees..."
    vertex_df = pd.DataFrame(rn.root2array(input_file,treename='VertexTree'))
    angle_df  = pd.DataFrame(rn.root2array(input_file,treename='AngleAnalysis'))
    shape_df  = pd.DataFrame(rn.root2array(input_file,treename='ShapeAnalysis'))
    gap_df    = pd.DataFrame(rn.root2array(input_file,treename="GapAnalysis"))
    match_df  = pd.DataFrame(rn.root2array(input_file,treename="MatchAnalysis"))
    dqds_df   = pd.DataFrame(rn.root2array(input_file,treename="dQdSAnalysis"))
    cosmic_df = pd.DataFrame(rn.root2array(input_file,treename="CosmicAnalysis"))


    print "Reindex..."
    vertex_df.set_index(rserv,inplace=True)
    angle_df.set_index(rserv,inplace=True) 
    shape_df.set_index(rserv,inplace=True) 
    gap_df.set_index(rserv,inplace=True)   
    match_df.set_index(rserv,inplace=True) 
    dqds_df.set_index(rserv,inplace=True) 
    cosmic_df.set_index(rserv,inplace=True) 

    #
    # Combine DataFrames
    #
    print "Combining Trees..."
    comb_df = pd.DataFrame()
    if data==False:
        print "... is not data"
        comb_df = pd.concat([vertex_df,
                             angle_df,
                             shape_df,
                             gap_df,
                             angle_df,
                             match_df,
                             dqds_df,
                             cosmic_df],axis=1)
    else:
        print "... is data"
        comb_df = pd.concat([angle_df,
                             shape_df,
                             gap_df,
                             angle_df,
                             match_df,
                             dqds_df,
                             cosmic_df],axis=1)        

    
    print "Dropping duplicate cols..."
    comb_df = comb_df.loc[:,~comb_df.columns.duplicated()]
    print "...dropped"

    print "Reindex..."
    comb_df.reset_index(inplace=True)
    print "...done"

    print "Setting vertex id..."
    comb_df['cvtxid'] = 0.0
    def func(group):
        group['cvtxid'] = np.arange(0,group['cvtxid'].size)
        return group
    comb_df = comb_df.groupby(['run','subrun','event']).apply(func)
    print "...set"

    print "Reindex..."
    comb_df.reset_index(inplace=True)
    print "...done"

    print "Joining with truth..."
    pg_df = pd.DataFrame(rn.root2array(input_file,treename="PGraphTruthMatch"))
    
    pg_df.rename(columns={'vtxid' : 'cvtxid'},inplace=True)

    pg_df.set_index(rsec,inplace=True)
    comb_df.set_index(rsec,inplace=True)

    comb_df = pd.concat([comb_df,pg_df],axis=1,join_axes=[comb_df.index])
    print "...joined"

    print "Reindex..."
    comb_df.reset_index(inplace=True)
    comb_df.set_index(rse,inplace=True)
    print "...done"

    print "Loading event TTrees..."
    event_vertex_df = pd.DataFrame(rn.root2array(input_file,treename="EventVertexTree"))
    nufilter_df     = pd.DataFrame(rn.root2array(input_file,treename="NuFilterTree"))
    mc_df           = pd.DataFrame(rn.root2array(input_file,treename="MCTree"))

    print "Reindex..."
    event_vertex_df.set_index(rse,inplace=True)
    nufilter_df.set_index(rse,inplace=True)
    mc_df.set_index(rse,inplace=True)
    print "...done"
    
    print "Joining nufilter..."
    event_vertex_df = event_vertex_df.join(nufilter_df,how='outer',lsuffix='',rsuffix='_y')
    print "...dropping"
    drop_y(event_vertex_df)
    print "...dropped"
        
    print "Joining mcdf..."
    event_vertex_df = event_vertex_df.join(mc_df,how='outer',lsuffix='',rsuffix='_y')
    print "...dropping"
    drop_y(event_vertex_df)
    print "...dropped"

    print "Joining with vertex..."
    comb_df = comb_df.join(event_vertex_df,how='outer',lsuffix='',rsuffix='_y')
    print "...dropping"
    drop_y(event_vertex_df)
    print "...dropped"

    print "Reindex..."
    comb_df.reset_index(inplace=True)
    print "...done"

    return comb_df

def track_shower_assumption(df):
    df['trkid'] = df.apply(lambda x : 0 if(x['par1_type']==1) else 1,axis=1)
    df['shrid'] = df.apply(lambda x : 1 if(x['par2_type']==2) else 0,axis=1)

    df['trk_frac_avg'] = df.apply(lambda x : x['par1_frac'] if(x['par1_type']==1) else x['par2_frac'],axis=1)
    df['shr_frac_avg'] = df.apply(lambda x : x['par2_frac'] if(x['par2_type']==2) else x['par1_frac'],axis=1)

def nue_assumption(comb_cut_df):
    print "Asking nue assumption"
    print "Asking npar==2"
    print "Asking in_fiducial==1"
    print "Asking pathexists2==1"

    comb_cut_df = comb_cut_df.query("npar==2")
    track_shower_assumption(comb_cut_df)
    comb_cut_df = comb_cut_df.query("par1_type != par2_type")
    comb_cut_df = comb_cut_df.query("in_fiducial==1")
    comb_cut_df = comb_cut_df.query("pathexists2==1")

    return comb_cut_df


def fill_parameters(comb_cut_df,ll_only=False):
    # SSNet Fraction
    #
    comb_cut_df['trk_frac'] = comb_cut_df.apply(lambda x : x['trk_frac_avg'] / x['nplanes_v'][x['trkid']],axis=1) 
    comb_cut_df['shr_frac'] = comb_cut_df.apply(lambda x : x['shr_frac_avg'] / x['nplanes_v'][x['shrid']],axis=1) 
    
    #
    # PCA
    #
    
    comb_cut_df['cosangle3d']=comb_cut_df.apply(lambda x : larocv.CosOpeningAngle(x['par_trunk_pca_theta_estimate_v'][0],
                                                                                  x['par_trunk_pca_phi_estimate_v'][0],
                                                                                  x['par_trunk_pca_theta_estimate_v'][1],
                                                                                  x['par_trunk_pca_phi_estimate_v'][1]),axis=1)
    
    comb_cut_df['angle3d'] = comb_cut_df.apply(lambda x : np.arccos(x['cosangle3d']),axis=1)

    
    comb_cut_df['shr_trunk_pca_theta_estimate'] = comb_cut_df.apply(lambda x : x['par_trunk_pca_theta_estimate_v'][x['shrid']],axis=1) 
    comb_cut_df['trk_trunk_pca_theta_estimate'] = comb_cut_df.apply(lambda x : x['par_trunk_pca_theta_estimate_v'][x['trkid']],axis=1) 
    
    comb_cut_df['shr_trunk_pca_cos_theta_estimate'] = comb_cut_df.apply(lambda x : np.cos(x['par_trunk_pca_theta_estimate_v'][x['shrid']]),axis=1) 
    comb_cut_df['trk_trunk_pca_cos_theta_estimate'] = comb_cut_df.apply(lambda x : np.cos(x['par_trunk_pca_theta_estimate_v'][x['trkid']]),axis=1) 
    
    
    #
    # 3D
    #
    comb_cut_df['shr_3d_length'] = comb_cut_df.apply(lambda x : x['par_pca_end_len_v'][x['shrid']],axis=1)
    comb_cut_df['trk_3d_length'] = comb_cut_df.apply(lambda x : x['par_pca_end_len_v'][x['trkid']],axis=1)
    
    comb_cut_df['shr_3d_QavgL'] = comb_cut_df.apply(lambda x : x['qsum_v'][x['shrid']] / x['par_pca_end_len_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    comb_cut_df['trk_3d_QavgL'] = comb_cut_df.apply(lambda x : x['qsum_v'][x['trkid']] / x['par_pca_end_len_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)
    
    #
    # Max deflection
    #
    comb_cut_df['shr_triangle_d_max'] = comb_cut_df.apply(lambda x : x['triangle_d_max_v'][x['shrid']],axis=1)
    comb_cut_df['trk_triangle_d_max'] = comb_cut_df.apply(lambda x : x['triangle_d_max_v'][x['trkid']],axis=1)
    
    #
    # Mean pixel dist from 2D PCA
    #
    comb_cut_df['shr_mean_pixel_dist'] = comb_cut_df.apply(lambda x : x['mean_pixel_dist_v'][x['shrid']]/x['nplanes_v'][x['shrid']],axis=1)
    comb_cut_df['shr_mean_pixel_dist_max'] = comb_cut_df.apply(lambda x : x['mean_pixel_dist_max_v'][x['shrid']],axis=1)
    comb_cut_df['shr_mean_pixel_dist_min'] = comb_cut_df.apply(lambda x : x['mean_pixel_dist_min_v'][x['shrid']],axis=1)
    comb_cut_df['shr_mean_pixel_dist_ratio'] = comb_cut_df.apply(lambda x : x['mean_pixel_dist_min_v'][x['shrid']] / x['mean_pixel_dist_max_v'][x['shrid']],axis=1)
    
    comb_cut_df['trk_mean_pixel_dist'] = comb_cut_df.apply(lambda x : x['mean_pixel_dist_v'][x['trkid']]/x['nplanes_v'][x['trkid']],axis=1)
    comb_cut_df['trk_mean_pixel_dist_max'] = comb_cut_df.apply(lambda x : x['mean_pixel_dist_max_v'][x['trkid']],axis=1)
    comb_cut_df['trk_mean_pixel_dist_min'] = comb_cut_df.apply(lambda x : x['mean_pixel_dist_min_v'][x['trkid']],axis=1)
    comb_cut_df['trk_mean_pixel_dist_ratio'] = comb_cut_df.apply(lambda x : x['mean_pixel_dist_min_v'][x['trkid']] / x['mean_pixel_dist_max_v'][x['trkid']],axis=1)     
    
    #
    # Sigma pixel dist from 2D PCA
    #
    comb_cut_df['shr_sigma_pixel_dist']       = comb_cut_df.apply(lambda x : x['sigma_pixel_dist_v'][x['shrid']]/x['nplanes_v'][x['shrid']],axis=1)
    comb_cut_df['shr_sigma_pixel_dist_max']   = comb_cut_df.apply(lambda x : x['sigma_pixel_dist_max_v'][x['shrid']],axis=1)
    comb_cut_df['shr_sigma_pixel_dist_min']   = comb_cut_df.apply(lambda x : x['sigma_pixel_dist_min_v'][x['shrid']],axis=1)
    comb_cut_df['shr_sigma_pixel_dist_ratio'] = comb_cut_df.apply(lambda x : x['sigma_pixel_dist_min_v'][x['shrid']] / x['sigma_pixel_dist_max_v'][x['shrid']],axis=1)
    
    comb_cut_df['trk_sigma_pixel_dist']       = comb_cut_df.apply(lambda x : x['sigma_pixel_dist_v'][x['trkid']]/x['nplanes_v'][x['trkid']],axis=1)
    comb_cut_df['trk_sigma_pixel_dist_max']   = comb_cut_df.apply(lambda x : x['sigma_pixel_dist_max_v'][x['trkid']],axis=1)
    comb_cut_df['trk_sigma_pixel_dist_min']   = comb_cut_df.apply(lambda x : x['sigma_pixel_dist_min_v'][x['trkid']],axis=1)
    comb_cut_df['trk_sigma_pixel_dist_ratio'] = comb_cut_df.apply(lambda x : x['sigma_pixel_dist_min_v'][x['trkid']] / x['sigma_pixel_dist_max_v'][x['trkid']],axis=1)    
    
    #
    # Ratio of # num pixels
    #
    comb_cut_df['shr_par_pixel_ratio'] = comb_cut_df.apply(lambda x : x['par_pixel_ratio_v'][x['shrid']],axis=1)
    comb_cut_df['trk_par_pixel_ratio'] = comb_cut_df.apply(lambda x : x['par_pixel_ratio_v'][x['trkid']],axis=1) 
    
    #
    # 2D angle difference @ vertex
    #
    comb_cut_df['anglediff0'] = comb_cut_df['anglediff'].values 
    
    #
    # 2D length
    #
    comb_cut_df['shr_avg_length']   = comb_cut_df.apply(lambda x : x['length_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    comb_cut_df['shr_length_min']   = comb_cut_df.apply(lambda x : x['length_min_v'][x['shrid']],axis=1)
    comb_cut_df['shr_length_max']   = comb_cut_df.apply(lambda x : x['length_max_v'][x['shrid']],axis=1)
    comb_cut_df['shr_length_ratio'] = comb_cut_df.apply(lambda x : x['length_min_v'][x['shrid']] / x['length_max_v'][x['shrid']],axis=1)
    
    comb_cut_df['trk_avg_length']   = comb_cut_df.apply(lambda x : x['length_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)
    comb_cut_df['trk_length_min']   = comb_cut_df.apply(lambda x : x['length_min_v'][x['trkid']],axis=1)
    comb_cut_df['trk_length_max']   = comb_cut_df.apply(lambda x : x['length_max_v'][x['trkid']],axis=1)
    comb_cut_df['trk_length_ratio'] = comb_cut_df.apply(lambda x : x['length_min_v'][x['trkid']] / x['length_max_v'][x['trkid']],axis=1)
    
    #
    # 2D width
    #
    comb_cut_df['shr_avg_width']   = comb_cut_df.apply(lambda x : x['width_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    comb_cut_df['shr_width_min']   = comb_cut_df.apply(lambda x : x['width_min_v'][x['shrid']],axis=1)
    comb_cut_df['shr_width_max']   = comb_cut_df.apply(lambda x : x['width_max_v'][x['shrid']],axis=1)
    comb_cut_df['shr_width_ratio'] = comb_cut_df.apply(lambda x : x['width_min_v'][x['shrid']] / x['width_max_v'][x['shrid']],axis=1)
    
    comb_cut_df['trk_avg_width']   = comb_cut_df.apply(lambda x : x['width_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)
    comb_cut_df['trk_width_max']   = comb_cut_df.apply(lambda x : x['width_max_v'][x['trkid']],axis=1)
    comb_cut_df['trk_width_min']   = comb_cut_df.apply(lambda x : x['width_max_v'][x['trkid']],axis=1)
    comb_cut_df['trk_width_ratio'] = comb_cut_df.apply(lambda x : x['width_min_v'][x['trkid']] / x['width_max_v'][x['trkid']],axis=1)
    
    #
    # 2D perimeter
    #
    comb_cut_df['shr_avg_perimeter'] = comb_cut_df.apply(lambda x : x['perimeter_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    comb_cut_df['shr_perimeter_min'] = comb_cut_df.apply(lambda x : x['perimeter_min_v'][x['shrid']],axis=1)
    comb_cut_df['shr_perimeter_max'] = comb_cut_df.apply(lambda x : x['perimeter_max_v'][x['shrid']],axis=1)
    comb_cut_df['shr_perimeter_ratio'] = comb_cut_df.apply(lambda x : x['perimeter_min_v'][x['shrid']] / x['perimeter_max_v'][x['shrid']],axis=1)
    
    comb_cut_df['trk_avg_perimeter'] = comb_cut_df.apply(lambda x : x['perimeter_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)
    comb_cut_df['trk_perimeter_min'] = comb_cut_df.apply(lambda x : x['perimeter_max_v'][x['trkid']],axis=1)
    comb_cut_df['trk_perimeter_max'] = comb_cut_df.apply(lambda x : x['perimeter_max_v'][x['trkid']],axis=1)
    comb_cut_df['trk_perimeter_ratio'] = comb_cut_df.apply(lambda x : x['perimeter_min_v'][x['trkid']] / x['perimeter_max_v'][x['trkid']],axis=1)
    
    #
    # 2D area
    #
    comb_cut_df['shr_avg_area'] = comb_cut_df.apply(lambda x : x['area_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    comb_cut_df['shr_area_min'] = comb_cut_df.apply(lambda x : x['area_min_v'][x['shrid']],axis=1)
    comb_cut_df['shr_area_max'] = comb_cut_df.apply(lambda x : x['area_max_v'][x['shrid']],axis=1)
    comb_cut_df['shr_area_ratio'] = comb_cut_df.apply(lambda x : x['area_min_v'][x['shrid']] / x['area_max_v'][x['shrid']],axis=1)
    
    comb_cut_df['trk_avg_area'] = comb_cut_df.apply(lambda x : x['area_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)
    comb_cut_df['trk_area_min'] = comb_cut_df.apply(lambda x : x['area_max_v'][x['trkid']],axis=1)
    comb_cut_df['trk_area_max'] = comb_cut_df.apply(lambda x : x['area_max_v'][x['trkid']],axis=1)
    comb_cut_df['trk_area_ratio'] = comb_cut_df.apply(lambda x : x['area_min_v'][x['trkid']] / x['area_max_v'][x['trkid']],axis=1)
    
    #
    # N pixel
    #
    comb_cut_df['shr_avg_npixel'] = comb_cut_df.apply(lambda x : x['npixel_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    comb_cut_df['shr_npixel_min'] = comb_cut_df.apply(lambda x : x['npixel_min_v'][x['shrid']],axis=1)
    comb_cut_df['shr_npixel_max'] = comb_cut_df.apply(lambda x : x['npixel_max_v'][x['shrid']],axis=1)
    comb_cut_df['shr_npixel_ratio'] = comb_cut_df.apply(lambda x : x['npixel_min_v'][x['shrid']] / x['npixel_max_v'][x['shrid']],axis=1)
    
    comb_cut_df['trk_avg_npixel'] = comb_cut_df.apply(lambda x : x['npixel_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)
    comb_cut_df['trk_npixel_min'] = comb_cut_df.apply(lambda x : x['npixel_max_v'][x['trkid']],axis=1)
    comb_cut_df['trk_npixel_max'] = comb_cut_df.apply(lambda x : x['npixel_max_v'][x['trkid']],axis=1)
    comb_cut_df['trk_npixel_ratio'] = comb_cut_df.apply(lambda x : x['npixel_min_v'][x['trkid']] / x['npixel_max_v'][x['trkid']],axis=1)
    
    #
    # Q sum
    #
    comb_cut_df['shr_avg_qsum']   = comb_cut_df.apply(lambda x : x['qsum_v'][x['shrid']] / x['nplanes_v'][x['shrid']],axis=1)
    comb_cut_df['shr_qsum_min']   = comb_cut_df.apply(lambda x : x['qsum_min_v'][x['shrid']],axis=1)
    comb_cut_df['shr_qsum_max']   = comb_cut_df.apply(lambda x : x['qsum_max_v'][x['shrid']],axis=1)
    comb_cut_df['shr_qsum_ratio'] = comb_cut_df.apply(lambda x : x['qsum_min_v'][x['shrid']] / x['qsum_max_v'][x['shrid']],axis=1)
    
    comb_cut_df['trk_avg_qsum']   = comb_cut_df.apply(lambda x : x['qsum_v'][x['trkid']] / x['nplanes_v'][x['trkid']],axis=1)
    comb_cut_df['trk_qsum_min']   = comb_cut_df.apply(lambda x : x['qsum_max_v'][x['trkid']],axis=1)
    comb_cut_df['trk_qsum_max']   = comb_cut_df.apply(lambda x : x['qsum_max_v'][x['trkid']],axis=1)
    comb_cut_df['trk_qsum_ratio'] = comb_cut_df.apply(lambda x : x['qsum_min_v'][x['trkid']] / x['qsum_max_v'][x['trkid']],axis=1)
    
    return comb_cut_df


def apply_ll(comb_cut_df,fin):
    print "--> reading PDFs"
    tf_in = ROOT.TFile(fin,"READ")
    tf_in.cd()
    
    keys_v = [key.GetName() for key in tf_in.GetListOfKeys()]
    
    sig_spectrum_m = {}
    bkg_spectrum_m = {}
    
    for key in keys_v:
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
    # Apply the LL
    #
    print "Applying LL"
    comb_cut_df['LL']= comb_cut_df.apply(LL,args=(sig_spectrum_m,bkg_spectrum_m,),axis=1)
    return comb_cut_df

def maximize_ll(comb_cut_df):
    passed_df = comb_cut_df.sort_values(["LL"],ascending=False).groupby(rse).head(1)
    return passed_df
