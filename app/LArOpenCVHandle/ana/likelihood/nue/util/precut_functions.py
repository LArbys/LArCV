from common import *
import pandas as pd
import root_numpy as rn
from util.ll_functions import prep_two_par_df, prep_LL_vars, LL_reco_parameters

def perform_precuts(INPUT_DF,
                    COSMIC_ROOT,
                    FLASH_ROOT,
                    DEDX_ROOT,
                    PRECUT_TXT,
                    IS_MC):

    print "Reading input"
    input_df = pd.read_pickle(INPUT_DF)

    input_df['precut_passed'] = int(-1)

    print "Read input_df=",INPUT_DF,"sz=",input_df.index.size,"RSE=",len(input_df.groupby(RSE))
    vertex_df  = input_df.query("locv_num_vertex>0").copy()

    event_df = input_df.groupby(RSE).nth(0)
    event_df['precut_passed'] = int(0)

    cosmic_df = pd.DataFrame()
    flash_df  = pd.DataFrame()
    dedx_df   = pd.DataFrame()

    name_v = []
    comb_v = []

    if len(COSMIC_ROOT)>0:

        print "Reading=",COSMIC_ROOT
        cosmic_df= pd.DataFrame(rn.root2array(COSMIC_ROOT))
        
        name_v.append("Cosmic")
        comb_v.append(cosmic_df)
    else: 
        return pd.DataFrame()
    
    if len(FLASH_ROOT)>0:
        print "Reading=",FLASH_ROOT
        flash_df = pd.DataFrame(rn.root2array(FLASH_ROOT))

        name_v.append("Flash")
        comb_v.append(flash_df)
    else:
        return pd.DataFrame()

    if len(DEDX_ROOT)>0:
        print "Reading=",DEDX_ROOT
        dedx_df = pd.DataFrame(rn.root2array(DEDX_ROOT))
        
        name_v.append("dEdx")
        comb_v.append(dedx_df)

    print "Combining..."
    print "vertex_df sz=",vertex_df.index.size,"RSE=",len(vertex_df.groupby(RSE))

    comb_df = vertex_df.copy()
    
    for name,df in zip(name_v,comb_v):
        print "@name=",name,"_df sz=",df.index.size,"RSE=",len(df.groupby(RSE))
        print "comb_df sz=",comb_df.index.size
        
        comb_df.set_index(RSEV,inplace=True)
        df.set_index(RSEV,inplace=True)
        
        comb_df = comb_df.join(df)
    
        comb_df.reset_index(inplace=True)
        df.reset_index(inplace=True)
    
    print "Preparing 2 particle data frame"
    comb_df = prep_two_par_df(comb_df)
    
    print "Preparing LL variables"
    comb_df = prep_LL_vars(comb_df,bool(IS_MC))

    print "Setting particle ID from SSNET"
    comb_df = set_ssnet_particle_reco_id(comb_df)
    
    print "Preparing reco parameters"
    comb_df = LL_reco_parameters(comb_df)

    print "Setting proton & electron track ID"
    comb_df = set_proton_track_id(comb_df)
    comb_df = set_electron_track_id(comb_df)

    print "Preparing precuts"
    comb_df = prepare_precuts(comb_df)
    
    print "Reading precuts=",PRECUT_TXT
    
    cuts = ""
    with open(PRECUT_TXT,'r') as f:
        cuts = f.read().split("\n")
        
    if cuts[-1]=="": 
        cuts = cuts[:-1]

    SS = ""
    for ix,cut in enumerate(cuts): 
        SS+= "(" + cut + ")"
        if ix != len(cuts)-1:
            SS+= " and "
    
    print "SS=",SS

    print "Precutting"
    comb_df.query(SS,inplace=True)
    
    if comb_df.index.size>0:
        comb_df['precut_passed'] = int(1)
        comb_df['valid'] = int(1)
        comb_df.set_index(RSE,inplace=True)
        comb_df = pd.concat((event_df,comb_df))
    else:
        comb_df = event_df.copy()

    comb_df.reset_index(inplace=True)

    return comb_df

def set_ssnet_particle_reco_id(df):
    df['reco_e_id'] = df.apply(lambda x : np.argmax(x['p08_shr_shower_frac']),axis=1)
    df['reco_p_id'] = df.apply(lambda x : np.argmin(x['p08_shr_shower_frac']),axis=1)
    return df

def xing(row):
    default=-0.000001
    
    if type(row['pt_xing_vv']) == float: return default
    
    xing_stack = np.vstack(row['pt_xing_vv'])
    conn_stack = np.vstack(row['connected_vv'])
    
    rows, cols = np.where(xing_stack==2)
    
    nrows = rows.size
    
    if nrows==0: return default
 
    nconn = float(conn_stack[rows,cols].sum())
    nrows = float(nrows)
    
    nc = nconn / nrows
    
    return nc

def xing_plane(row,plane):
    default=0
    
    if type(row['pt_xing_vv']) == float: return default
    if type(row['connected_vv']) == float: return default
    
    xing_stack = np.vstack(row['pt_xing_vv'])[plane,:]
    conn_stack = np.vstack(row['connected_vv'])[plane,:]
    
    id_v = np.where(xing_stack==2)[0]

    if id_v.size==0: return default
 
    nc = float(conn_stack[id_v].sum())

    return nc / float(id_v.size)


def in_fiducial(row):
    
    X = float(row['anatrk1_vtx_x'])
    Y = float(row['anatrk1_vtx_y'])
    Z = float(row['anatrk1_vtx_z'])
    
    XX = 10.0
    YY = 20.0
    ZZ = 10.0
    
    if (Z<ZZ        or Z>(1036-ZZ)): return 0
    if (X<XX        or X>(256-XX)):  return 0
    if (Y<(-111+YY) or Y>(111-YY)):  return 0

    return 1

rrange = np.arange(5,20).astype(np.float32)

def cos(row):
    plane=2
    
    n2_idx = np.where(row['pt_xing_vv'][plane]==2)[0]
    if n2_idx.size==0: return -1.1

    distance_v = row['distance_vv'][plane][n2_idx]
    radius_v = rrange[n2_idx]
    
    cos  = 2*np.power(radius_v,2.0) - np.power(distance_v,2.0)
    cos /= (2*radius_v*radius_v)
    
    return list(cos)

def three_planes_two(row):
    stack = np.vstack(row['pt_xing_vv'])

    res = np.ones(15)*0
    
    slice_ = ((stack[0,:]==2)&(stack[1,:]==2)&(stack[2,:]==2))
    
    s_v = np.where(slice_)[0]
    if s_v.size==0: return list(res)
    
    res[s_v] = 1
    return list(res)
    
def two_planes_two(row):
    stack = np.vstack(row['pt_xing_vv'])

    res = np.ones(15)*0

    slice_  = ((stack[0,:]==2)&(stack[1,:]==2))
    slice_ |= ((stack[0,:]==2)&(stack[2,:]==2))
    slice_ |= ((stack[1,:]==2)&(stack[2,:]==2))

    s_v = np.where(slice_)[0]
    if s_v.size==0: return list(res)
    
    res[s_v]=1
    return list(res)

def cos2(row,plane):

    res = np.ones(15)*-1.0
    n2_idx = np.where(row['pt_xing_vv'][plane]==2)[0]
    if n2_idx.size==0: return list(res)

    distance_v = row['distance_vv'][plane][n2_idx]
    radius_v = rrange[n2_idx]
    
    cos  = 2*np.power(radius_v,2.0) - np.power(distance_v,2.0)
    cos /= (2*radius_v*radius_v)
    
    res[n2_idx] = np.arccos(cos)*180.0/np.pi
    
    return list(res)

def get_1e1p_chi2(row):
    tid = int(row['reco_proton_trackid'])
    eid = int(row['reco_e_id'])

    ret = float(-1.0)
    
    if type(row['proton_shower_pair_vv']) is float: return ret
    
    if row['proton_shower_pair_vv'].size==0: return ret
    
    stack = np.vstack(row['proton_shower_pair_vv'])
    
    slice_ = ((stack[:,0]==tid) & (stack[:,1]==eid))
    idx_v = np.where(slice_)[0]
    
    if idx_v.size==0: return ret
    
    idx = idx_v[0]
    
    ret = float(row['proton_shower_chi2_1e1p_v'][idx])
    
    return ret


def get_1e1p_hypope(row):
    tid = int(row['reco_proton_trackid'])
    eid = int(row['reco_e_id'])

    ret_v = [-1.0]*32
    
    stack = np.vstack(row['proton_shower_pair_vv'])
    
    slice_ = ((stack[:,0]==tid) & (stack[:,1]==eid))
    idx_v = np.where(slice_)[0]
    
    if idx_v.size==0: return ret_v
    
    idx = idx_v[0]
    
    ret_v = list(row['proton_shower_hypo_pe_vv'][idx])
    
    return ret_v

def get_1e1p_datape(row):
    tid = int(row['reco_proton_trackid'])
    eid = int(row['reco_e_id'])

    ret_v = [-1.0]*32
    
    stack = np.vstack(row['proton_shower_pair_vv'])
    
    slice_ = ((stack[:,0]==tid) & (stack[:,1]==eid))
    idx_v = np.where(slice_)[0]
    
    if idx_v.size==0: return ret_v
    
    idx = idx_v[0]
    
    flash_id = int(row['proton_shower_best_data_flash_v'][idx])
    
    ret_v = list(row['data_pe_vv'][flash_id])
    
    return ret_v


def prepare_precuts(df):
    df['fiducial'] = df.apply(in_fiducial,axis=1)
    df['two_pt']   = df.apply(xing,axis=1)
    df['cos']      = df.apply(cos,axis=1)
    df['cos2_p0']  = df.apply(cos2,args=(0,),axis=1)
    df['cos2_p1']  = df.apply(cos2,args=(1,),axis=1)
    df['cos2_p2']  = df.apply(cos2,args=(2,),axis=1)
    df['three_planes_two'] = df.apply(three_planes_two,axis=1)
    df['two_planes_two']   = df.apply(two_planes_two,axis=1)
    df['two_pt_p0']   = df.apply(xing_plane,args=(0,),axis=1)
    df['two_pt_p1']   = df.apply(xing_plane,args=(1,),axis=1)
    df['two_pt_p2']   = df.apply(xing_plane,args=(2,),axis=1)
    df['chi2']        = df.apply(get_1e1p_chi2,axis=1)
    return df


def set_proton_track_id(df):
    df['reco_proton_trackid'] = df.apply(reco_proton_track_id,axis=1)
    return df

def set_electron_track_id(df):
    df['reco_electron_trackid'] = df.apply(reco_electron_track_id,axis=1)
    return df

def reco_proton_track_id(row):
    pgtrk_v   = row['pgtrk_trk_type_v']
    pgtrk_vv  = row['pgtrk_trk_type_vv']
    protonid  = row['reco_p_id']
    trkid_v = np.where(pgtrk_v==protonid)[0]

    if trkid_v.size == 0: 
        return int(-1)
    
    elif trkid_v.size == 1 : 
        if pgtrk_vv[trkid_v[0]][protonid] == 0: 
            return int(-1)
        else:
            return int(trkid_v[0])
    else:
        stacked   = np.vstack(pgtrk_vv[trkid_v])[:,protonid]
        maxid_trk = np.argmax(stacked)
        if stacked[maxid_trk]==0: 
            return int(-1)
        else: 
            return int(maxid_trk)

def reco_electron_track_id(row):
    pgtrk_v   = row['pgtrk_trk_type_v']
    pgtrk_vv  = row['pgtrk_trk_type_vv']
    eleid  = row['reco_e_id']
    trkid_v = np.where(pgtrk_v==eleid)[0]

    if trkid_v.size == 0: 
        return int(-1)
    
    elif trkid_v.size == 1 : 
        if pgtrk_vv[trkid_v[0]][eleid] == 0: 
            return int(-1)
        else:
            return int(trkid_v[0])
    else:
        stacked   = np.vstack(pgtrk_vv[trkid_v])[:,eleid]
        maxid_trk = np.argmax(stacked)
        if stacked[maxid_trk]==0: 
            return int(-1)
        else: 
            return int(maxid_trk)
        
def reco_proton_track_param(row,param):
    res = -1.0
    if row['reco_proton_trackid']<0: return res
    return row[param][row['reco_proton_trackid']]

def reco_electron_track_param(row,param):
    res = -1.0
    if row['reco_electron_trackid']<0: return res
    return row[param][row['reco_electron_trackid']]
    
