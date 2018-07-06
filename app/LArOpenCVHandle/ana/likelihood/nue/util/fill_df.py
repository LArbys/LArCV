from larlitecv import larlitecv

import numpy as np
import pandas as pd
import ROOT
import root_numpy as rn
from common import *

def initialize_rst(VTX_DF,ST_DF):

    comb_df = pd.DataFrame()
    
    ana_vtx_df = pd.read_pickle(VTX_DF)
    ana_st_df  = pd.read_pickle(ST_DF)

    print "ana_vtx.index.size=",ana_vtx_df.index.size
    print "ana_st_df.index.size=",ana_st_df.index.size

    if 'vtxid' not in ana_vtx_df.columns:
        print "No vertex dataframe encountered"
        
        ana_vtx_df.set_index(RSE,inplace=True)
        ana_vtx_df = ana_vtx_df.add_prefix('locv_')
        ana_vtx_df.reset_index(inplace=True)

        # no mcinfo filled
        if ana_st_df.index.size == 0:
            return ana_vtx_df

        # event mcinfo filled
        ana_st_df.set_index(RSE,inplace=True)

        assert ana_st_df.index.size == ana_vtx_df.index.size

        comb_df = ana_vtx_df.join(ana_st_df,how='outer',lsuffix='',rsuffix='_q')
        drop_q(comb_df)
        comb_df.reset_index(inplace=True)
        return comb_df

    ana_vtx_df.drop(['vtxid'],axis=1,inplace=True)
    ana_vtx_df.rename(columns={'cvtxid' : 'vtxid'},inplace=True)

    ana_locv_df = ana_vtx_df.query("num_vertex>0").copy()
    ana_rest_df = ana_vtx_df.drop(ana_locv_df.index).copy()

    ana_ll_df   = ana_st_df.query("vtxid>=0").copy()
    ana_lest_df = ana_st_df.drop(ana_ll_df.index).copy()

    assert ((ana_rest_df.index.size + ana_locv_df.index.size) == ana_vtx_df.index.size)
    assert ((ana_lest_df.index.size + ana_ll_df.index.size) == ana_st_df.index.size)

    ana_locv_df.set_index(RSEV,inplace=True)
    ana_locv_df = ana_locv_df.add_prefix('locv_')

    ana_ll_df.set_index(RSEV,inplace=True)
    
    print "ana_locv_df.index.size=",ana_locv_df.index.size
    print "ana_ll_df.index.size=",ana_ll_df.index.size

    df_v    = [ana_ll_df,ana_locv_df]
    comb_df = pd.concat(df_v,axis=1,join_axes=[df_v[0].index])

    comb_df.reset_index(inplace=True)

    comb_df.set_index(RSE,inplace=True)

    ana_rest_df.set_index(RSE,inplace=True)
    ana_lest_df.set_index(RSE,inplace=True)

    print "ana_rest_df.index.size=",ana_rest_df.index.size
    print "ana_lest_df.index.size=",ana_lest_df.index.size

    cols = ana_rest_df.columns[~ana_rest_df.columns.str.contains('vtxid')]
    ana_rest_df.rename(columns = dict(zip(cols, 'locv_' + cols)), inplace=True)

    ana_rest_lest_df = pd.concat([ana_rest_df, ana_lest_df],axis=1,join_axes=[ana_rest_df.index])

    ana_rest_lest_df = ana_rest_lest_df.loc[:,~ana_rest_lest_df.columns.duplicated()]
    
    comb_df.reset_index(inplace=True)
    ana_rest_lest_df.reset_index(inplace=True)

    print "comb_df.index.size=",comb_df.index.size
    print "ana_rest_lest_df.index.size=",ana_rest_lest_df.index.size

    comb_df = comb_df.append(ana_rest_lest_df,ignore_index=True)
    
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

    # 
    # exception @ is data
    #

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
        # it's data
        isdata = True

    #
    # exception @ no vertex found
    #

    try:
        pgraph_trk_df = pd.DataFrame(rn.root2array(TRK_PGRPH,treename="TrackPGraphMatch"))
        ana_shr1_df   = pd.DataFrame(rn.root2array(SHR_ANA1,treename="ShowerQuality_DL"))
        ana_shr2_df   = pd.DataFrame(rn.root2array(SHR_ANA1,treename="EventMCINFO_DL"))
        ana_trk1_df   = pd.DataFrame(rn.root2array(TRK_ANA1))
        ana_trk2_df   = pd.DataFrame(rn.root2array(TRK_ANA2))

        ana_shr1_df.rename(columns={'vtx_id': 'vtxid'}, inplace=True)
        ana_trk2_df.rename(columns={'vtx_id': 'vtxid'}, inplace=True)

        pgraph_trk_df.set_index(RSEV,inplace=True)
        ana_shr1_df.set_index(RSEV,inplace=True)
        ana_shr2_df.set_index(RSE,inplace=True)
        ana_trk1_df.set_index(RSEV,inplace=True)
        ana_trk2_df.set_index(RSEV,inplace=True)
    
        pgraph_trk_df = pgraph_trk_df.add_prefix("pgtrk_")
        ana_shr1_df   = ana_shr1_df.add_prefix("anashr1_")
        ana_shr2_df   = ana_shr2_df.add_prefix("anashr2_")
        ana_trk1_df   = ana_trk1_df.add_prefix("anatrk1_")
        ana_trk2_df   = ana_trk2_df.add_prefix("anatrk2_")

        print "pgraph_trk_df.index.size=",pgraph_trk_df.index.size
        print "ana_shr1_df.index.size=",ana_shr1_df.index.size
        print "ana_shr2_df.index.size=",ana_shr2_df.index.size
        print "ana_trk1_df.index.size=",ana_trk1_df.index.size
        print "ana_trk2_df.index.size=",ana_trk2_df.index.size

        df_v = []

        if isdata==False:
            df_v = [ana_shr1_df,
                    ana_trk1_df,
                    ana_trk2_df,
                    pgraph_trk_df,
                    match_shr_df,
                    match_trk_df]
        else:
            df_v = [ana_shr1_df,
                    ana_trk1_df,
                    ana_trk2_df,
                    pgraph_trk_df]

        comb_df = pd.concat(df_v,axis=1,join_axes=[df_v[0].index])
        comb_df.reset_index(inplace=True)
        
        if isdata==False:
            comb_df.set_index(RSE,inplace=True)
            comb_df = comb_df.join(ana_shr2_df,how='outer',lsuffix='',rsuffix='_q')
            drop_q(comb_df)

        comb_df.reset_index(inplace=True)
        return comb_df

    except IOError:
        # no vertex found

        # it's data return empty
        if isdata==True:
            return pd.DataFrame()

        # MC info should still be filled...
        ana_shr2_df = pd.DataFrame(rn.root2array(SHR_ANA1,treename="EventMCINFO_DL"))
        ana_shr2_df.set_index(RSE,inplace=True)
        ana_shr2_df = ana_shr2_df.add_prefix("anashr2_")
        ana_shr2_df.reset_index(inplace=True)
        return ana_shr2_df

    raise Exception

def initialize_stp(SHR_ANA1,
                   SHR_TRUTH,
                   TRK_ANA1,
                   TRK_ANA2,
                   TRK_TRUTH,
                   TRK_PGRPH,
                   PID_ANA,
                   PID_ANA2):

    comb_df = pd.DataFrame()
    
    match_shr_df = None
    match_trk_df = None

    # 
    # exception @ is data
    #

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
        # it's data
        isdata = True

    #
    # exception @ no vertex found
    #

    try:
        pgraph_trk_df = pd.DataFrame(rn.root2array(TRK_PGRPH,treename="TrackPGraphMatch"))
        ana_shr1_df   = pd.DataFrame(rn.root2array(SHR_ANA1,treename="ShowerQuality_DL"))
        ana_shr2_df   = pd.DataFrame(rn.root2array(SHR_ANA1,treename="EventMCINFO_DL"))
        ana_trk1_df   = pd.DataFrame(rn.root2array(TRK_ANA1))
        ana_trk2_df   = pd.DataFrame(rn.root2array(TRK_ANA2))
        ana_pid_df    = pd.DataFrame(rn.root2array(PID_ANA))
        ana_pid2_df   = pd.DataFrame(rn.root2array(PID_ANA2))

        ana_shr1_df.rename(columns={'vtx_id': 'vtxid'}, inplace=True)
        ana_trk2_df.rename(columns={'vtx_id': 'vtxid'}, inplace=True)

        pgraph_trk_df.set_index(RSEV,inplace=True)
        ana_shr1_df.set_index(RSEV,inplace=True)
        ana_shr2_df.set_index(RSE,inplace=True)
        ana_trk1_df.set_index(RSEV,inplace=True)
        ana_trk2_df.set_index(RSEV,inplace=True)
        ana_pid_df.set_index(RSEV,inplace=True)
        ana_pid2_df.set_index(RSEV,inplace=True)
    
        pgraph_trk_df = pgraph_trk_df.add_prefix("pgtrk_")
        ana_shr1_df   = ana_shr1_df.add_prefix("anashr1_")
        ana_shr2_df   = ana_shr2_df.add_prefix("anashr2_")
        ana_trk1_df   = ana_trk1_df.add_prefix("anatrk1_")
        ana_trk2_df   = ana_trk2_df.add_prefix("anatrk2_")
        ana_pid_df    = ana_pid_df.add_prefix("anapid_")
        ana_pid2_df   = ana_pid2_df.add_prefix("anapid2_")

        print "pgraph_trk_df.index.size=",pgraph_trk_df.index.size
        print "ana_shr1_df.index.size=",ana_shr1_df.index.size
        print "ana_trk1_df.index.size=",ana_trk1_df.index.size
        print "ana_trk2_df.index.size=",ana_trk2_df.index.size
        print "ana_pid_df.index.size=",ana_pid_df.index.size
        print "ana_pid2_df.index.size=",ana_pid2_df.index.size

        df_v = []

        if isdata==False:
            df_v = [ana_shr1_df,
                    ana_trk1_df,
                    ana_trk2_df,
                    ana_pid_df,
                    ana_pid2_df,
                    pgraph_trk_df,
                    match_shr_df,
                    match_trk_df]
        else:
            df_v = [ana_shr1_df,
                    ana_trk1_df,
                    ana_trk2_df,
                    ana_pid_df,
                    ana_pid2_df,
                    pgraph_trk_df]

        comb_df = pd.concat(df_v,axis=1,join_axes=[df_v[0].index])
        comb_df.reset_index(inplace=True)
        
        if isdata==False:
            comb_df.set_index(RSE,inplace=True)
            comb_df = comb_df.join(ana_shr2_df,how='outer',lsuffix='',rsuffix='_q')
            drop_q(comb_df)

        comb_df.reset_index(inplace=True)
        return comb_df

    except IOError:
        # no vertex found

        # it's data return empty
        if isdata==True:
            return pd.DataFrame()

        # MC info should still be filled...
        ana_shr2_df = pd.DataFrame(rn.root2array(SHR_ANA1,treename="EventMCINFO_DL"))
        ana_shr2_df.set_index(RSE,inplace=True)
        ana_shr2_df = ana_shr2_df.add_prefix("anashr2_")
        ana_shr2_df.reset_index(inplace=True)
        return ana_shr2_df

    raise Exception

def add_to_rst(RST_PKL,NUMU_LL_ROOT):
    comb_df = pd.DataFrame()

    ana_rst_df  = pd.read_pickle(RST_PKL)
    if 'vtxid' not in ana_rst_df.columns:
        print "No vertex dataframe encountered"
        return ana_rst_df

    ana_numu_df = pd.DataFrame(rn.root2array(NUMU_LL_ROOT,treename="NuMuVertexVariables"))
    
    print "ana_rst_df.index.size=",ana_rst_df.index.size
    print "ana_numu_df.index.size=",ana_numu_df.index.size
    print

    ana_vtx_df = ana_rst_df.query("locv_num_vertex>0").copy()
    ana_rest_df = ana_rst_df.drop(ana_vtx_df.index).copy()

    assert ((ana_rest_df.index.size + ana_vtx_df.index.size) == ana_rst_df.index.size)
    assert ana_vtx_df.index.size == ana_numu_df.index.size

    ana_vtx_df.set_index(RSEV,inplace=True)

    ana_numu_df.rename(columns={ "_run"   : "run",
                                 "_subrun": "subrun",
                                 "_event" : "event",
                                 '_vtxid' : 'vtxid'},inplace=True)


    ana_numu_df.set_index(RSEV,inplace=True)
    ana_numu_df = ana_numu_df.add_prefix('numu_')

    print "ana_vtx_df.index.size=",ana_vtx_df.index.size
    print "ana_numu_df.index.size=",ana_numu_df.index.size
    print

    df_v    = [ana_numu_df,ana_vtx_df]
    comb_df = pd.concat(df_v,axis=1,join_axes=[df_v[0].index])

    comb_df.reset_index(inplace=True)

    print "comb_df.index.size=",comb_df.index.size
    print "ana_rest_df.index.size=",ana_rest_df.index.size

    comb_df = comb_df.append(ana_rest_df,ignore_index=True)
    
    print "now... comb_df.index.size=",comb_df.index.size

    return comb_df

def initialize_stpn(SHR_ANA1,
                    SHR_TRUTH,
                    TRK_ANA1,
                    TRK_ANA2,
                    TRK_TRUTH,
                    TRK_PGRPH,
                    PID_ANA1,
                    PID_ANA2,
                    NUEID_ANA,
                    FLASH_ANA,
                    DEDX_ANA):


    comb_df = pd.DataFrame()
    
    match_shr_df = None
    match_trk_df = None

    # 
    # exception @ is data
    #

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
        # it's data
        isdata = True

    #
    # exception @ no vertex found
    #

    try:
        print "LOADING TTREES"
	print "@TRK_PGRPH=",TRK_PGRPH
        pgraph_trk_df = pd.DataFrame(rn.root2array(TRK_PGRPH,treename="TrackPGraphMatch"))
        
        print "@SHR_ANA1",SHR_ANA1
        ana_shr1_df   = pd.DataFrame(rn.root2array(SHR_ANA1,treename="ShowerQuality_DL"))

        print "@SHR_ANA1",SHR_ANA1
        ana_shr2_df   = pd.DataFrame(rn.root2array(SHR_ANA1,treename="EventMCINFO_DL"))

        print "@TRK_ANA1",TRK_ANA1
        ana_trk1_df   = pd.DataFrame(rn.root2array(TRK_ANA1,treename="TrackRecoAna"))

        print "@TRK_ANA2",TRK_ANA2
        ana_trk2_df   = pd.DataFrame(rn.root2array(TRK_ANA2,treename="_recoTree"))

        print "@PID_ANA1",PID_ANA1
        ana_pid1_df   = pd.DataFrame(rn.root2array(PID_ANA1,treename="multipid_tree"))

	print "@PID_ANA2",PID_ANA2
        ana_pid2_df   = pd.DataFrame(rn.root2array(PID_ANA2,treename="multiplicity_tree"))
    
        print "@NUEID_ANA",NUEID_ANA
        ana_nueid_df  = pd.DataFrame(rn.root2array(NUEID_ANA,treename="SelNueID"))

        print "@FLASH_ANA",FLASH_ANA
        ana_flash_df  = pd.DataFrame(rn.root2array(FLASH_ANA,treename="ffmatch"))

        print "@DEDX_ANA",DEDX_ANA
        ana_dedx_df   = pd.DataFrame(rn.root2array(DEDX_ANA,treename="trackdir"))
        
        ana_shr1_df.rename(columns={'vtx_id': 'vtxid'}, inplace=True)
        ana_trk2_df.rename(columns={'vtx_id': 'vtxid'}, inplace=True)

        pgraph_trk_df.set_index(RSEV,inplace=True)
        ana_shr1_df.set_index(RSEV,inplace=True)
        ana_shr2_df.set_index(RSE,inplace=True)
        ana_trk1_df.set_index(RSEV,inplace=True)
        ana_trk2_df.set_index(RSEV,inplace=True)
        ana_pid1_df.set_index(RSEV,inplace=True)
        ana_pid2_df.set_index(RSEV,inplace=True)
        ana_nueid_df.set_index(RSEV,inplace=True)
        ana_flash_df.set_index(RSEV,inplace=True)
        ana_dedx_df.set_index(RSEV,inplace=True)
    
        pgraph_trk_df = pgraph_trk_df.add_prefix("pgtrk_")
        ana_shr1_df   = ana_shr1_df.add_prefix("anashr1_")
        ana_shr2_df   = ana_shr2_df.add_prefix("anashr2_")
        ana_trk1_df   = ana_trk1_df.add_prefix("anatrk1_")
        ana_trk2_df   = ana_trk2_df.add_prefix("anatrk2_")
        ana_pid1_df   = ana_pid1_df.add_prefix("anapid1_")
        ana_pid2_df   = ana_pid2_df.add_prefix("anapid2_")
        ana_nueid_df  = ana_nueid_df.add_prefix("nueid_")
        ana_flash_df  = ana_flash_df.add_prefix("flash_")
        ana_dedx_df   = ana_dedx_df.add_prefix("dedx_")

        print "pgraph_trk_df.index.size=",pgraph_trk_df.index.size
        print "ana_shr1_df.index.size=",ana_shr1_df.index.size
        print "ana_shr2_df.index.size=",ana_shr2_df.index.size
        print "ana_trk1_df.index.size=",ana_trk1_df.index.size
        print "ana_trk2_df.index.size=",ana_trk2_df.index.size
        print "ana_pid1_df.index.size=",ana_pid1_df.index.size
        print "ana_pid2_df.index.size=",ana_pid2_df.index.size
        print "ana_nueid_df.index.size=",ana_nueid_df.index.size
        print "ana_flash_df.index.size=",ana_flash_df.index.size
        print "ana_dedx_df.index.size=",ana_dedx_df.index.size

        df_v = []

        if isdata==False:
            df_v = [pgraph_trk_df,
                    ana_shr1_df,
                    ana_trk1_df,
                    ana_trk2_df,
                    ana_pid1_df,
                    ana_pid2_df,
                    ana_nueid_df,
                    ana_flash_df,
                    ana_dedx_df,
                    match_shr_df,
                    match_trk_df]

        else:
            df_v = [pgraph_trk_df,
                    ana_shr1_df,
                    ana_trk1_df,
                    ana_trk2_df,
                    ana_pid1_df,
                    ana_pid2_df,
                    ana_nueid_df,
                    ana_flash_df,
                    ana_dedx_df]

        comb_df = pd.concat(df_v,axis=1,join_axes=[df_v[0].index])
        comb_df.reset_index(inplace=True)
        
        if isdata==False:
            comb_df.set_index(RSE,inplace=True)
            comb_df = comb_df.join(ana_shr2_df,how='outer',lsuffix='',rsuffix='_q')
            drop_q(comb_df)

        comb_df.reset_index(inplace=True)
        return comb_df

    except IOError:
        # no vertex found

        # it's data return empty
        if isdata==True:
            return pd.DataFrame()

        # MC info should still be filled...
        ana_shr2_df = pd.DataFrame(rn.root2array(SHR_ANA1,treename="EventMCINFO_DL"))
        ana_shr2_df.set_index(RSE,inplace=True)
        ana_shr2_df = ana_shr2_df.add_prefix("anashr2_")
        ana_shr2_df.reset_index(inplace=True)
        return ana_shr2_df

    raise Exception

def initialize_stpn(SHR_ANA1,
                    SHR_TRUTH,
                    TRK_ANA1,
                    TRK_ANA2,
                    TRK_TRUTH,
                    TRK_PGRPH,
                    PID_ANA,
                    PID_ANA2,
                    NUEID_ANA,
                    DEDX_ANA):


    comb_df = pd.DataFrame()
    
    match_shr_df = None
    match_trk_df = None

    # 
    # exception @ is data
    #

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
        # it's data
        isdata = True

    #
    # exception @ no vertex found
    #

    try:
        pgraph_trk_df = pd.DataFrame(rn.root2array(TRK_PGRPH,treename="TrackPGraphMatch"))
        ana_shr1_df   = pd.DataFrame(rn.root2array(SHR_ANA1,treename="ShowerQuality_DL"))
        ana_shr2_df   = pd.DataFrame(rn.root2array(SHR_ANA1,treename="EventMCINFO_DL"))
        ana_trk1_df   = pd.DataFrame(rn.root2array(TRK_ANA1,treename="TrackRecoAna"))
        ana_trk2_df   = pd.DataFrame(rn.root2array(TRK_ANA2,treename="_recoTree"))
        ana_pid1_df   = pd.DataFrame(rn.root2array(PID_ANA1,treename="multipid_tree"))
        ana_pid2_df   = pd.DataFrame(rn.root2array(PID_ANA2,treename="multiplicity_tree"))
        ana_nueid_df  = pd.DataFrame(rn.root2array(NUEID_ANA,treename="SelNueID"))
        ana_dedx_df   = pd.DataFrame(rn.root2array(DEDX_ANA,treename="trackdir"))
        
        ana_shr1_df.rename(columns={'vtx_id': 'vtxid'}, inplace=True)
        ana_trk2_df.rename(columns={'vtx_id': 'vtxid'}, inplace=True)

        pgraph_trk_df.set_index(RSEV,inplace=True)
        ana_shr1_df.set_index(RSEV,inplace=True)
        ana_shr2_df.set_index(RSE,inplace=True)
        ana_trk1_df.set_index(RSEV,inplace=True)
        ana_trk2_df.set_index(RSEV,inplace=True)
        ana_pid1_df.set_index(RSEV,inplace=True)
        ana_pid2_df.set_index(RSEV,inplace=True)
        ana_nueid_df.set_index(RSEV,inplace=True)
        ana_dedx_df.set_index(RSEV,inplace=True)
    
        pgraph_trk_df = pgraph_trk_df.add_prefix("pgtrk_")
        ana_shr1_df   = ana_shr1_df.add_prefix("anashr1_")
        ana_shr2_df   = ana_shr2_df.add_prefix("anashr2_")
        ana_trk1_df   = ana_trk1_df.add_prefix("anatrk1_")
        ana_trk2_df   = ana_trk2_df.add_prefix("anatrk2_")
        ana_pid1_df   = ana_pid1_df.add_prefix("anapid1_")
        ana_pid2_df   = ana_pid2_df.add_prefix("anapid2_")
        ana_nueid_df  = ana_nueid_df.add_prefix("nueid_")
        ana_dedx_df   = ana_dedx_df.add_prefix("dedx_")

        print "pgraph_trk_df.index.size=",pgraph_trk_df.index.size
        print "ana_shr1_df.index.size=",ana_shr1_df.index.size
        print "ana_shr2_df.index.size=",ana_shr2_df.index.size
        print "ana_trk1_df.index.size=",ana_trk1_df.index.size
        print "ana_trk2_df.index.size=",ana_trk2_df.index.size
        print "ana_pid1_df.index.size=",ana_pid1_df.index.size
        print "ana_pid2_df.index.size=",ana_pid2_df.index.size
        print "ana_nueid_df.index.size=",ana_nueid_df.index.size
        print "ana_dedx_df.index.size=",ana_dedx_df.index.size

        df_v = []

        if isdata==False:
            df_v = [pgraph_trk_df,
                    ana_shr1_df,
                    ana_trk1_df,
                    ana_trk2_df,
                    ana_pid1_df,
                    ana_pid2_df,
                    ana_nueid_df,
                    ana_dedx_df,
                    match_shr_df,
                    match_trk_df]

        else:
            df_v = [pgraph_trk_df,
                    ana_shr1_df,
                    ana_trk1_df,
                    ana_trk2_df,
                    ana_pid1_df,
                    ana_pid2_df,
                    ana_nueid_df,
                    ana_dedx_df]

        comb_df = pd.concat(df_v,axis=1,join_axes=[df_v[0].index])
        comb_df.reset_index(inplace=True)
        
        if isdata==False:
            comb_df.set_index(RSE,inplace=True)
            comb_df = comb_df.join(ana_shr2_df,how='outer',lsuffix='',rsuffix='_q')
            drop_q(comb_df)

        comb_df.reset_index(inplace=True)
        return comb_df

    except IOError:
        # no vertex found

        # it's data return empty
        if isdata==True:
            return pd.DataFrame()

        # MC info should still be filled...
        ana_shr2_df = pd.DataFrame(rn.root2array(SHR_ANA1,treename="EventMCINFO_DL"))
        ana_shr2_df.set_index(RSE,inplace=True)
        ana_shr2_df = ana_shr2_df.add_prefix("anashr2_")
        ana_shr2_df.reset_index(inplace=True)
        return ana_shr2_df

    raise Exception


def add_to_rst(RST_PKL,NUMU_LL_ROOT):
    comb_df = pd.DataFrame()

    ana_rst_df  = pd.read_pickle(RST_PKL)
    if 'vtxid' not in ana_rst_df.columns:
        print "No vertex dataframe encountered"
        return ana_rst_df

    ana_numu_df = pd.DataFrame(rn.root2array(NUMU_LL_ROOT,treename="NuMuVertexVariables"))
    
    print "ana_rst_df.index.size=",ana_rst_df.index.size
    print "ana_numu_df.index.size=",ana_numu_df.index.size
    print

    ana_vtx_df = ana_rst_df.query("locv_num_vertex>0").copy()
    ana_rest_df = ana_rst_df.drop(ana_vtx_df.index).copy()

    assert ((ana_rest_df.index.size + ana_vtx_df.index.size) == ana_rst_df.index.size)
    assert ana_vtx_df.index.size == ana_numu_df.index.size

    ana_vtx_df.set_index(RSEV,inplace=True)

    ana_numu_df.rename(columns={ "_run"   : "run",
                                 "_subrun": "subrun",
                                 "_event" : "event",
                                 '_vtxid' : 'vtxid'},inplace=True)


    ana_numu_df.set_index(RSEV,inplace=True)
    ana_numu_df = ana_numu_df.add_prefix('numu_')

    print "ana_vtx_df.index.size=",ana_vtx_df.index.size
    print "ana_numu_df.index.size=",ana_numu_df.index.size
    print

    df_v    = [ana_numu_df,ana_vtx_df]
    comb_df = pd.concat(df_v,axis=1,join_axes=[df_v[0].index])

    comb_df.reset_index(inplace=True)

    print "comb_df.index.size=",comb_df.index.size
    print "ana_rest_df.index.size=",ana_rest_df.index.size

    comb_df = comb_df.append(ana_rest_df,ignore_index=True)
    
    print "now... comb_df.index.size=",comb_df.index.size

    return comb_df



def initialize_rt(VERTEX_PKL,NUMU_LL_ROOT):
    
    comb_df = pd.DataFrame()

    ana_vtx_df  = pd.read_pickle(VERTEX_PKL)
    ana_numu_df = pd.DataFrame(rn.root2array(NUMU_LL_ROOT,treename="NuMuVertexVariables"))
    
    print "ana_vtx_df.index.size=",ana_vtx_df.index.size
    print "ana_numu_df.index.size=",ana_numu_df.index.size
    print
    if 'vtxid' not in ana_vtx_df.columns:
        print "No vertex dataframe encountered"
        
        ana_vtx_df.set_index(RSE,inplace=True)
        ana_vtx_df = ana_vtx_df.add_prefix('locv_')
        ana_vtx_df.reset_index(inplace=True)

        return ana_vtx_df

    ana_vtx_df.drop(['vtxid'],axis=1,inplace=True)
    ana_vtx_df.rename(columns={'cvtxid' : 'vtxid'},inplace=True)

    ana_locv_df = ana_vtx_df.query("num_vertex>0").copy()
    ana_rest_df = ana_vtx_df.drop(ana_locv_df.index).copy()

    assert ((ana_rest_df.index.size + ana_locv_df.index.size) == ana_vtx_df.index.size)
    assert ana_locv_df.index.size == ana_numu_df.index.size

    ana_locv_df.set_index(RSEV,inplace=True)
    ana_locv_df = ana_locv_df.add_prefix('locv_')

    ana_numu_df.rename(columns={ "_run"   : "run",
                                 "_subrun": "subrun",
                                 "_event" : "event",
                                 '_vtxid' : 'vtxid'},inplace=True)


    ana_numu_df.set_index(RSEV,inplace=True)
    ana_numu_df = ana_numu_df.add_prefix('numu_')

    print "ana_locv_df.index.size=",ana_locv_df.index.size
    print "ana_numu_df.index.size=",ana_numu_df.index.size
    print

    df_v    = [ana_numu_df,ana_locv_df]
    comb_df = pd.concat(df_v,axis=1,join_axes=[df_v[0].index])

    comb_df.reset_index(inplace=True)

    comb_df.set_index(RSE,inplace=True)
    ana_rest_df.set_index(RSE,inplace=True)

    print "ana_rest_df.index.size=",ana_rest_df.index.size
    print

    cols = ana_rest_df.columns[~ana_rest_df.columns.str.contains('vtxid')]
    ana_rest_df.rename(columns = dict(zip(cols, 'locv_' + cols)), inplace=True)
    
    comb_df.reset_index(inplace=True)
    ana_rest_df.reset_index(inplace=True)

    print "comb_df.index.size=",comb_df.index.size
    print "ana_rest_df.index.size=",ana_rest_df.index.size

    comb_df = comb_df.append(ana_rest_df,ignore_index=True)
    
    print "now... comb_df.index.size=",comb_df.index.size

    return comb_df


def initialize_rt_LL(VERTEX_PKL,NUMU_LL_ROOT):
    
    comb_df = pd.DataFrame()

    ana_vtx_df  = pd.read_pickle(VERTEX_PKL)
    ana_numu_df = pd.DataFrame(rn.root2array(NUMU_LL_ROOT,treename="NuMuVertexVariables"))
    
    print "ana_vtx_df.index.size=",ana_vtx_df.index.size
    print "ana_numu_df.index.size=",ana_numu_df.index.size
    print
    if 'vtxid' not in ana_vtx_df.columns:
        print "No vertex dataframe encountered"
        return ana_vtx_df

    ana_locv_df = ana_vtx_df.query("locv_num_vertex>0").copy()
    ana_rest_df = ana_vtx_df.drop(ana_locv_df.index).copy()

    assert ((ana_rest_df.index.size + ana_locv_df.index.size) == ana_vtx_df.index.size)
    assert ana_locv_df.index.size == ana_numu_df.index.size

    ana_locv_df.set_index(RSEV,inplace=True)

    ana_numu_df.rename(columns={ "_run"   : "run",
                                 "_subrun": "subrun",
                                 "_event" : "event",
                                 '_vtxid' : 'vtxid'},inplace=True)


    ana_numu_df.set_index(RSEV,inplace=True)
    ana_numu_df = ana_numu_df.add_prefix('numu_')

    print "ana_locv_df.index.size=",ana_locv_df.index.size
    print "ana_numu_df.index.size=",ana_numu_df.index.size
    print

    df_v    = [ana_numu_df,ana_locv_df]
    comb_df = pd.concat(df_v,axis=1,join_axes=[df_v[0].index])

    comb_df.reset_index(inplace=True)
    ana_rest_df.reset_index(inplace=True)

    print "comb_df.index.size=",comb_df.index.size
    print "ana_rest_df.index.size=",ana_rest_df.index.size

    comb_df = comb_df.append(ana_rest_df,ignore_index=True)
    
    print "now... comb_df.index.size=",comb_df.index.size

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


