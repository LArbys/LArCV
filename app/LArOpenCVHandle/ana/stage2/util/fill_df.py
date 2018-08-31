import numpy as np
import pandas as pd
import ROOT
import root_numpy as rn
from common import *

def initialize_rst(VTX_DF,NUE_DF):

    comb_df = pd.DataFrame()
    
    ana_vtx_df = pd.read_pickle(VTX_DF)
    ana_nue_df  = pd.read_pickle(NUE_DF)

    print "ana_vtx.index.size=",ana_vtx_df.index.size
    print "ana_nue_df.index.size=",ana_nue_df.index.size

    if 'vtxid' not in ana_vtx_df.columns:
        print "No vertex dataframe encountered"

        ana_vtx_df['run']     = ana_vtx_df['run'].astype(np.int32)
        ana_vtx_df['subrun']  = ana_vtx_df['subrun'].astype(np.int32)
        ana_vtx_df['event']   = ana_vtx_df['event'].astype(np.int32)

        ana_vtx_df.set_index(RSE,inplace=True)
        ana_vtx_df = ana_vtx_df.add_prefix('locv_')
        ana_vtx_df.reset_index(inplace=True)

        return comb_df


    ana_vtx_df['run']     = ana_vtx_df['run'].astype(np.int32)
    ana_vtx_df['subrun']  = ana_vtx_df['subrun'].astype(np.int32)
    ana_vtx_df['event']   = ana_vtx_df['event'].astype(np.int32)
    
    ana_nue_df['run']    = ana_nue_df['run'].astype(np.int32)
    ana_nue_df['subrun'] = ana_nue_df['subrun'].astype(np.int32)
    ana_nue_df['event']  = ana_nue_df['event'].astype(np.int32)
    ana_nue_df['vtxid']  = ana_nue_df['vtxid'].astype(np.int32)

    ana_vtx_df.drop(['vtxid'],axis=1,inplace=True)
    ana_vtx_df.rename(columns={'cvtxid' : 'vtxid'},inplace=True)

    ana_locv_df = ana_vtx_df.query("num_vertex>0").copy()
    ana_rest_df = ana_vtx_df.drop(ana_locv_df.index).copy()

    ana_locv_df['vtxid'] = ana_locv_df['vtxid'].astype(np.int32)
    ana_rest_df['vtxid'] = int(-1)

    assert ((ana_rest_df.index.size + ana_locv_df.index.size) == ana_vtx_df.index.size)

    ana_locv_df.set_index(RSEV,inplace=True)
    ana_locv_df = ana_locv_df.add_prefix('locv_')

    ana_nue_df.set_index(RSEV,inplace=True)
    
    print "ana_locv_df.index.size=",ana_locv_df.index.size
    print "ana_nue_df.index.size=",ana_nue_df.index.size
    assert (ana_locv_df.index.size == ana_nue_df.index.size)

    df_v    = [ana_nue_df,ana_locv_df]
    comb_df = pd.concat(df_v,axis=1,join_axes=[df_v[0].index])

    comb_df.reset_index(inplace=True)

    ana_rest_df.set_index(RSE,inplace=True)

    print "ana_rest_df.index.size=",ana_rest_df.index.size

    cols = ana_rest_df.columns[~ana_rest_df.columns.str.contains('vtxid')]
    ana_rest_df.rename(columns = dict(zip(cols, 'locv_' + cols)), inplace=True)

    ana_rest_df = ana_rest_df.loc[:,~ana_rest_df.columns.duplicated()]

    print "comb_df.index.size=",comb_df.index.size
    print "ana_rest_df.index.size=",ana_rest_df.index.size

    ana_rest_df.reset_index(inplace=True)

    comb_df = comb_df.append(ana_rest_df,ignore_index=True)
    
    print "now... comb_df.index.size=",comb_df.index.size

    return comb_df

def initialize_nueid(SHR_ANA1,
                     SHR_TRUTH,
                     PID_ANA1,
                     PID_ANA2,
                     NUEID_ANA,
                     FLASH_ANA):
    

    comb_df = pd.DataFrame()
    
    match_shr_df = None
    match_trk_df = None

    # 
    # exception @ is data
    #

    isdata = False
    try:
        match_shr_df  = pd.DataFrame(rn.root2array(SHR_TRUTH,treename="ShowerTruthMatch"))
        match_shr_df.set_index(RSEV,inplace=True)
        match_shr_df  = match_shr_df.add_prefix("mchshr_")
        print "match_shr_df.index.size=",match_shr_df.index.size

    except IOError:
        isdata = True

    #
    # exception @ no vertex found
    #

    try:
        print "LOADING TTREES"
        
        print "@SHR_ANA1",SHR_ANA1
        ana_shr1_df   = pd.DataFrame(rn.root2array(SHR_ANA1,treename="ShowerQuality_DL"))

        print "@PID_ANA1",PID_ANA1
        ana_pid1_df   = pd.DataFrame(rn.root2array(PID_ANA1,treename="multipid_tree"))

	print "@PID_ANA2",PID_ANA2
        ana_pid2_df   = pd.DataFrame(rn.root2array(PID_ANA2,treename="multiplicity_tree"))
    
        print "@NUEID_ANA",NUEID_ANA
        ana_nueid_df  = pd.DataFrame(rn.root2array(NUEID_ANA,treename="SelNueID"))

        print "@FLASH_ANA",FLASH_ANA
        ana_flash_df  = pd.DataFrame(rn.root2array(FLASH_ANA,treename="ffmatch"))

        ana_shr1_df.set_index(RSEV,inplace=True)
        ana_pid1_df.set_index(RSEV,inplace=True)
        ana_pid2_df.set_index(RSEV,inplace=True)
        ana_nueid_df.set_index(RSEV,inplace=True)
        ana_flash_df.set_index(RSEV,inplace=True)

        ana_shr1_df   = ana_shr1_df.add_prefix("anashr1_")
        ana_pid1_df   = ana_pid1_df.add_prefix("anapid1_")
        ana_pid2_df   = ana_pid2_df.add_prefix("anapid2_")
        ana_nueid_df  = ana_nueid_df.add_prefix("nueid_")
        ana_flash_df  = ana_flash_df.add_prefix("flash_")

        print "ana_shr1_df.index.size=",ana_shr1_df.index.size
        print "ana_pid1_df.index.size=",ana_pid1_df.index.size
        print "ana_pid2_df.index.size=",ana_pid2_df.index.size
        print "ana_nueid_df.index.size=",ana_nueid_df.index.size
        print "ana_flash_df.index.size=",ana_flash_df.index.size

        df_v = []

        if isdata==False:
            df_v = [ana_shr1_df,
                    ana_pid1_df,
                    ana_pid2_df,
                    ana_nueid_df,
                    ana_flash_df,
                    match_shr_df]
                    
        else:
            df_v = [ana_shr1_df,
                    ana_pid1_df,
                    ana_pid2_df,
                    ana_nueid_df,
                    ana_flash_df]

        comb_df = pd.concat(df_v,axis=1,join_axes=[df_v[0].index])
        comb_df.reset_index(inplace=True)

        return comb_df

    except IOError:
        # no vertex found return empty
        return pd.DataFrame()

    raise Exception

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
    shape_df  = pd.DataFrame(rn.root2array(input_file,treename='ShapeAnalysis'))
    match_df  = pd.DataFrame(rn.root2array(input_file,treename='MatchAnalysis'))
    shower_df = pd.DataFrame(rn.root2array(input_file,treename='SecondShowerAnalysis'))

    print "Reindex..."
    vertex_df.set_index(rserv,inplace=True)
    shape_df.set_index(rserv,inplace=True)
    match_df.set_index(rserv,inplace=True)
    shower_df.set_index(rserv,inplace=True)
 
    #
    # Combine DataFrames
    #
    print "Combining Trees..."
    comb_df = pd.DataFrame()

    df_v = []
    
    if data==True:
        df_v = [shape_df,match_df,shower_df]
    else:
        df_v = [shape_df,match_df,shower_df,vertex_df]

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

