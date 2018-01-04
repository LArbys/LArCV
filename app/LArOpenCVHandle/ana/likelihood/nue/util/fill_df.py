import numpy as np
import pandas as pd
import ROOT
import root_numpy as rn
from common import *

def initialize_rst(VTX_DF,ST_DF):

    comb_df = pd.DataFrame()
    
    ana_vtx_df = pd.read_pickle(VTX_DF)

    if 'vtxid' not in ana_vtx_df.columns:
        print "EMPTY DF ENCOUNTERED"
        assert pd.read_pickle(ST_DF).index.size == 0
        ana_vtx_df.set_index(RSE,inplace=True)
        ana_vtx_df = ana_vtx_df.add_prefix('locv_')
        ana_vtx_df.reset_index(inplace=True)
        return ana_vtx_df

    ana_vtx_df.drop(['vtxid'],axis=1,inplace=True)
    ana_vtx_df.rename(columns={'cvtxid' : 'vtxid'},inplace=True)

    ana_locv_df = ana_vtx_df.query("num_vertex>0").copy()
    ana_rest_df = ana_vtx_df.drop(ana_locv_df.index).copy()
     
    assert ((ana_rest_df.index.size + ana_locv_df.index.size) == ana_vtx_df.index.size)

    ana_locv_df.set_index(RSEV,inplace=True)
    ana_locv_df = ana_locv_df.add_prefix('locv_')

    ana_st_df = pd.read_pickle(ST_DF)

    ana_st_df.set_index(RSEV,inplace=True)
    
    print "ana_locv_df.index.size=",ana_locv_df.index.size
    print "ana_st_df.index.size=",ana_st_df.index.size

    df_v    = [ana_st_df,ana_locv_df]
    comb_df = pd.concat(df_v,axis=1,join_axes=[df_v[0].index])

    comb_df.reset_index(inplace=True)

    comb_df.set_index(RSE,inplace=True)
    ana_rest_df.set_index(RSE,inplace=True)

    cols = ana_rest_df.columns[~ana_rest_df.columns.str.contains('vtxid')]
    ana_rest_df.rename(columns = dict(zip(cols, 'locv_' + cols)), inplace=True)
    
    comb_df.reset_index(inplace=True)
    ana_rest_df.reset_index(inplace=True)
    
    comb_df = comb_df.append(ana_rest_df,ignore_index=True)
    
    print "now... comb_df.index.size=",comb_df.index.size

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

    try:
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

    except IOError:
        return pd.DataFrame()


def initialize_truth(input_file,data=False):
    print "Loading event TTrees..."
    nufilter_df     = pd.DataFrame(rn.root2array(input_file,treename="NuFilterTree"))
    mc_df           = pd.DataFrame(rn.root2array(input_file,treename="MCTree"))

    print "Reindex..."
    nufilter_df.set_index(rse,inplace=True)
    mc_df.set_index(rse,inplace=True)
    print "...done"
    
    print "Joining mcdf..."
    nufilter_df = nufilter_df.join(mc_df,how='outer',lsuffix='',rsuffix='_q')
    print "...dropping"
    drop_q(nufilter_df)
    print "...dropped"

    print "Reindex..."
    nufilter_df.reset_index(inplace=True)
    print "...done"

    return nufilter_df

    
def initialize_r(ana_true_df,ana_reco_df):
    
    comb_df = pd.DataFrame()
    
    print "ana_true_df.index.size=",ana_true_df.index.size

    ana_true_df.set_index(RSE,inplace=True)
    ana_reco_df.set_index(RSE,inplace=True)
    
    comb_df = ana_reco_df.join(ana_true_df,how='outer',lsuffix='',rsuffix='_q')
    drop_q(comb_df)

    comb_df.reset_index(inplace=True)

    assert len(comb_df.groupby(RSE)) == int(ana_true_df.index.size)

    ana_true_df.reset_index(inplace=True)
    ana_reco_df.reset_index(inplace=True)

    return comb_df

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

    df_v = []
    
    if data==True:
        df_v = [angle_df,shape_df,gap_df,angle_df,match_df,dqds_df,cosmic_df]
    else:
        df_v = [angle_df,shape_df,gap_df,angle_df,match_df,dqds_df,cosmic_df,vertex_df]

    comb_df = pd.concat(df_v,axis=1)
    
    print "Dropping duplicate cols..."
    comb_df = comb_df.loc[:,~comb_df.columns.duplicated()]
    comb_df.drop(['entry'],axis=1,inplace=True)
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
    pg_df.drop(['entry'],axis=1,inplace=True)
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
    event_vertex_df.drop(['entry'],axis=1,inplace=True)
    print "Reindex..."
    event_vertex_df.set_index(rse,inplace=True)
    print "...done"

    print "Joining with vertex..."
    comb_df = comb_df.join(event_vertex_df,how='outer',lsuffix='',rsuffix='_q')
    print "...dropping"
    drop_q(comb_df)
    print "...dropped"

    print "Reindex..."
    comb_df.reset_index(inplace=True)
    print "...done"
    
    return comb_df


