from common import *
import pandas as pd
import numpy as np

def combine_vertex_numu(ana_vtx_df,ana_numu_df=pd.DataFrame()):
    print
    print "start @ combine_vertex_numu"
    comb_df = pd.DataFrame()
    
    print "ana_vtx.index.size=",ana_vtx_df.index.size
    print "ana_numu_df.index.size=",ana_numu_df.index.size
    
    if 'vtxid' not in ana_vtx_df.columns:
        print "No vertex dataframe encountered"
        assert ana_numu_df.empty == True

        ana_vtx_df['run']    = ana_vtx_df['run'].astype(np.int32)
        ana_vtx_df['subrun'] = ana_vtx_df['subrun'].astype(np.int32)
        ana_vtx_df['event']  = ana_vtx_df['event'].astype(np.int32)
        
        ana_vtx_df.set_index(RSE,inplace=True)
        ana_vtx_df = ana_vtx_df.add_prefix('locv_')
        ana_vtx_df.reset_index(inplace=True)
        
        print "ana_vtx_df.shape=",ana_vtx_df.shape
        
        return ana_vtx_df 

    ana_numu_df['run']    = ana_numu_df['run'].astype(np.int32)
    ana_numu_df['subrun'] = ana_numu_df['subrun'].astype(np.int32)
    ana_numu_df['event']  = ana_numu_df['event'].astype(np.int32)
    ana_numu_df['vtxid']  = ana_numu_df['vtxid'].astype(np.int32)
    
    ana_vtx_df.drop(['vtxid'],axis=1,inplace=True)
    ana_vtx_df.rename(columns={'cvtxid' : 'vtxid'},inplace=True)
    
    ana_locv_df = ana_vtx_df.query("num_vertex>0").copy()
    ana_rest_df = ana_vtx_df.drop(ana_locv_df.index).copy()

    ana_locv_df['vtxid'] = ana_locv_df['vtxid'].astype(np.int32)
    ana_rest_df['vtxid'] = int(-1)

    assert ((ana_rest_df.index.size + ana_locv_df.index.size) == ana_vtx_df.index.size)

    ana_locv_df.set_index(RSEV,inplace=True)
    ana_locv_df = ana_locv_df.add_prefix('locv_')

    ana_numu_df.set_index(RSEV,inplace=True)
    
    print "ana_locv_df.index.size=",ana_locv_df.index.size
    print "ana_numu_df.index.size=",ana_numu_df.index.size
    assert (ana_locv_df.index.size == ana_numu_df.index.size)

    df_v    = [ana_numu_df,ana_locv_df]
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
    
    print "end @ combine_vertex_numu"
    print

    return comb_df


def bless_numu_scedr(row):

    dr = float(-1.0)

    dx = row['mcinfo_parentSCEX'] - row['numu_Xreco']
    dy = row['mcinfo_parentSCEY'] - row['numu_Yreco']
    dz = row['mcinfo_parentSCEZ'] - row['numu_Zreco']

    dr = dx*dx+dy*dy+dz*dz
    dr = np.sqrt(dr)

    return dr

def bless_nue_scedr(row):

    dr = float(-1.0)

    dx = row['mcinfo_parentSCEX'] - row['nueid_vertex_x']
    dy = row['mcinfo_parentSCEY'] - row['nueid_vertex_y']
    dz = row['mcinfo_parentSCEZ'] - row['nueid_vertex_z']

    dr = dx*dx+dy*dy+dz*dz
    dr = np.sqrt(dr)

    return dr
