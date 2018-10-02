from cut_functions import *
import pandas as pd
from collections import OrderedDict

# 
# Cut application
#

def apply_cuts(cut_df,CUT_CFG):
    print "Applying cuts..."
    
    out_df = cut_df.copy()

    out_df['passed_cuts'] = 0
    out_df['selected'] = 0

    if 'nueid_vertex_x' not in out_df.columns:
        return out_df
        
    cut_ss = read_config(CUT_CFG)

    passed_index = out_df.query(cut_ss).index
    out_df.loc[passed_index, 'passed_cuts'] = 1
    
    out_df['flash_chi2'] = out_df.apply(determine_flash,axis=1)

    passed_df = out_df.query("passed_cuts==1")

    if passed_df.empty == True:
        return out_df

    flash_index = passed_df.sort_values(['flash_chi2'],ascending=True).groupby(RSE).head(1).index

    out_df.loc[flash_index, 'selected'] = 1

    print "...done!"    
    
    return out_df

def parse_cuts(COMB_DF,drop_list=None):
    print "Building cuts..."
    
    comb_df = pd.read_pickle(COMB_DF)
    
    out_df = comb_df.copy()

    # fill default values
    out_df['pid']   = int(-1)
    out_df['eid']   = int(-1)
    out_df['pfrac'] = float(-1)
    out_df['efrac'] = float(-1)
    out_df['reco_proton_energy']   = float(-1)
    out_df['reco_electron_energy'] = float(-1)
    out_df['reco_energy']          = float(-1)
    out_df['reco_ccqe_p']          = float(-1)

    # handle no vertex in data frame (empty data frame)
    if 'nueid_vertex_x' not in out_df.columns:
        out_df = set_cuts(out_df,doit=False)
        return out_df

    # clean data frame
    out_df = drop_columns(out_df,drop_list)
    
    # set electron v. proton
    out_df = set_particle_id(out_df)
    
    # set particle
    out_df = set_particle_energy(out_df)

    # set cuts
    out_df = set_cuts(out_df,doit=True)

    print "...done"

    return out_df

#
# Set particles
#
def set_particle_id(df):
    odf = df.copy()

    odf['rpid']  = odf.apply(define_ep,axis=1)
    odf['pid']   = odf['rpid'].str[0].values
    odf['eid']   = odf['rpid'].str[1].values
    odf['pfrac'] = odf['rpid'].str[2].values
    odf['efrac'] = odf['rpid'].str[3].values
    
    odf.drop(labels=['rpid'],
             axis=1,
             inplace=True,
             errors='ignore')    
    
    return odf

def set_particle_energy(df):
    odf = df.copy()
    
    odf['reco_proton_energy']   = odf.apply(reco_proton_energy,axis=1)
    odf['reco_electron_energy'] = odf.apply(reco_electron_energy,axis=1)
    odf['reco_energy']          = odf.apply(reco_energy,axis=1)
    odf['reco_ccqe_p']          = odf.apply(reco_ccqe_energy,axis=1)    
    
    return odf

#
# drop unused columns
#

def drop_columns(df,drop_list=None):
    odf = df.copy()
    
    if drop_list == None:
        return odf

    drop_list_v = None
    with open(drop_list,"r") as f:
        drop_list_v = f.read()

    drop_list_v = drop_list_v.split("\n")
    drop_list_v = [d for d in drop_list_v if d!='']
    
    odf.drop(labels=drop_list_v,
             axis=1,
             inplace=True,
             errors='ignore')

    return odf

#
# Set Cuts
#
def set_cuts(df,doit=True):
    odf = df.copy()

    cut_v = OrderedDict()
    cut_v["c01"] = c01
    cut_v["c02"] = c02
    cut_v["c22"] = c22
    cut_v["c23"] = c23
    cut_v["c24"] = c24
    cut_v["c25"] = c25
    cut_v["c26"] = c26
    cut_v["c27"] = c27
    cut_v["c28"] = c28
    cut_v["c29"] = c29
    cut_v["c30"] = c30
    cut_v["c31"] = c31
    cut_v["c32"] = c32
    cut_v["c33"] = c33
    cut_v["c34"] = c34
    cut_v["c35"] = c35
    cut_v["c36"] = c36
    cut_v["c37"] = c37
    cut_v["c38"] = c38
    cut_v["c39"] = c39
    cut_v["c40"] = c40
    cut_v["c41"] = c41
    cut_v["c42"] = c42
    cut_v["c43"] = c43
    cut_v["c44"] = c44
    cut_v["c45"] = c45

    if doit == True:
        for cname in cut_v.keys():
            odf[cname] = odf.apply(cut_v[cname],axis=1)
    else:
        for cname in cut_v.keys():
            odf[cname] = int(0)

    return odf

#
# Boxed cuts
#
def c01(row):
    ret = 0

    two = (row['nueid_vtx_xing_U']==2 and 
           row['nueid_vtx_xing_V']==2 and 
           row['nueid_vtx_xing_Y']==2)

    if ((row['nueid_n_par']==2) or
        (two == True and row['nueid_n_par']==3)):
        ret = 1

    return ret

def c02(row):
    ret = 0

    cut = float(np.min(row['nueid_edge_cosmic_vtx_dist_v']))
    
    if (cut > 5):
        ret = 1
        
    return ret

def c19(row):
    ret = 0
    
    if row["pid"]<0: return 0
    if row["eid"]<0: return 0
    
    data2u = angle(row,"U")
    data2v = angle(row,"V")
    data2y = angle(row,"Y")
    
    cut_v = np.array([data2u,data2v,data2y])
    cut = np.max(cut_v)
    
    if (cut < 170):
        ret = 1
        
    return ret
  

def c22(row):
    ret = 0


    if int(row['eid']) < 0: 
        return ret
    
    ev1 = pparam_plane(row,"eid","nueid_par1_showerfrac_U",0)
    ev2 = pparam_plane(row,"eid","nueid_par1_showerfrac_V",1)
    ev3 = pparam_plane(row,"eid","nueid_par1_showerfrac_Y",2)
    
    ev_v = np.array([ev1,ev2,ev3])

    ev_idx_v = np.argsort(ev_v)
    ev_id1 = ev_idx_v[-1]
    ev_id2 = ev_idx_v[-2]
    
    cut1 = ev_v[ev_id1]
    cut2 = ev_v[ev_id2]
    
    if (cut1 > 0.5 and 
        cut2 > 0.5):
        ret = 1
    
    return ret

def c23(row):
    ret = 0

    if int(row['pid']) < 0: 
        return ret
    
    ev1 = pparam_plane(row,"pid","nueid_par1_showerfrac_U",0)
    ev2 = pparam_plane(row,"pid","nueid_par1_showerfrac_V",1)
    ev3 = pparam_plane(row,"pid","nueid_par1_showerfrac_Y",2)
    
    ev_v = np.array([ev1,ev2,ev3])
    ev_idx_v = np.argsort(ev_v)
    ev_id1 = ev_idx_v[-1]
    ev_id2 = ev_idx_v[-2]
    
    cut1 = ev_v[ev_id1]
    cut2 = ev_v[ev_id2]
    
    if (cut1 < 0.5 and 
        cut2 < 0.5):
        ret = 1
    
    return ret

def c24(row):
    ret = 0
        
    if int(row['pid']) < 0: 
        return ret

    if int(row['eid']) < 0: 
        return ret
    
    pv = pparam(row,"pid","nueid_par1_dz1")
    ev = pparam(row,"eid","nueid_par1_dz1")
    
    ev_f = zfunc0_test(ev)
    
    if (pv > ev_f):
        ret = 1
        
    return ret

def c25(row):
    ret = 0
    
    if int(row['pid']) < 0: 
        return ret

    if int(row['eid']) < 0: 
        return ret
    
    e_brem = max_pparam(row,"eid","nueid_par1_triangle_brem_U")
    p_brem = no_brem(row,"pid")

    if (e_brem  > 0 and 
        p_brem == 1):
        ret = 1
    
    return ret


def c26(row):
    ret = 0
    
    if int(row['pid']) < 0: 
        return ret

    if int(row['eid']) < 0: 
        return ret

    pmax = max_pparam(row,"pid","nueid_par1_linefrac_U")
    pmin = min_pparam(row,"pid","nueid_par1_linefrac_U")
    
    pbr = multi_pparam_v(row,"pid","nueid_par1_polybranches_U_v")
    pdf = multi_pparam_v(row,"pid","nueid_par1_numberdefects_U_v")
    
    if ((pmax > 0.99 and pmin > 0.99) and 
        (pbr[0] == 0 and pbr[1] == 0) and 
        (pdf[0] == 0 and pdf[1] == 0)):
        ret = 1;
    
    return ret

def c27(row):

    ret = 0
        
    if int(row['pid']) < 0: 
        return ret

    if int(row['eid']) < 0: 
        return ret

    cut_v = multi_pparam_v(row,"eid","nueid_par1_polyedges_U_v")
    
    if (cut_v[0] > 1 and 
        cut_v[1] > 1):
        ret = 1
        
    return ret
    
    
def c28(row):
    ret = 0
    
    if int(row['pid']) < 0: 
        return ret

    if int(row['eid']) < 0: 
        return ret
    
    p_E_cal = row['reco_proton_energy']
    e_E_cal = row['reco_electron_energy']

    e_p_E_cal = p_E_cal + e_E_cal
    
    p_angle = pparam(row,"pid","nueid_par1_dz1")
    
    p_v = [p_angle, p_E_cal]
    
    CCQE_e = CCQE_p(p_v)

    E_vis  = float(e_p_E_cal+0.511+40)
    
    dE = (CCQE_e - E_vis) / E_vis

    if (dE > -0.7 and 
        dE <  2.0):
        ret = 1
    
    return ret

def c29(row):
    ret = 0
    
    if int(row['pid']) < 0: 
        return ret

    if int(row['eid']) < 0: 
        return ret

    ret = 1

    return ret

def c30(row):
    
    ret = 0

    if int(row['pid']) < 0: 
        return ret
    
    dqdx_v = dqdx_plane(row,'pid',2)
    
    if dqdx_v[0] <= 0: return 1
    
    dqdx_med = float(np.median(dqdx_v[1]))
    
    if dqdx_med > 350:
        ret = 1
    
    return ret

def c31(row):
    ret = 0
    
    if int(row['pid']) < 0: 
        return ret

    p_brem = no_brem(row,"pid")

    if p_brem == 1:
        ret = 1;
        
    return ret

def c32(row):
    ret = 0
    
    if int(row['eid']) < 0: 
        return ret

    if int(row['pid']) < 0: 
        return ret
    
    par1_dist_v = row['nueid_par1_cosmic_dist_end_v']
    par2_dist_v = row['nueid_par2_cosmic_dist_end_v']
    
    if par1_dist_v.size == 0: 
        return ret

    if par2_dist_v.size == 0: 
        return ret
    
    par1_dist_v = par1_dist_v[par1_dist_v > 0]
    par2_dist_v = par2_dist_v[par2_dist_v > 0]
    
    if par1_dist_v.size == 0: 
        return ret
        
    if par2_dist_v.size == 0: 
        return ret
    
    if (np.min(par1_dist_v) > 5 and 
        np.min(par2_dist_v) > 5):
        ret = 1

    return ret


def c33(row):
    ret = 0
    
    if int(row['eid']) < 0: 
        return ret

    emax = max_pparam(row,"eid","nueid_par1_linefrac_U")
    emin = min_pparam(row,"eid","nueid_par1_linefrac_U")
    
    if (emax < 0.99 and 
        emin < 0.99):
        ret = 1
    
    return ret

def c34(row):
    ret = 0
    
    if int(row['eid']) < 0: 
        return ret
    
    emax = max_pparam(row,"eid","nueid_par1_line_vtx_density_U")
    emin = min_pparam(row,"eid","nueid_par1_line_vtx_density_U")
    
    if (emax < 0.9 and 
        emin < 0.9):
        ret = 1
    
    return ret
    
def c35(row):
    
    ret = 0
    
    if int(row['eid']) < 0: 
        return ret
    
    emax = max_pparam(row,"eid","nueid_par1_linelength_U")
    emin = min_pparam(row,"eid","nueid_par1_linelength_U")
    
    if (emax < 150 and
        emin < 150):
        ret = 1

    return ret

def c36(row):
    ret = 0

    cut = np.min(row['nueid_edge_cosmic_end_vtx_dist_v'])

    if cut > 20:
        ret = 1
        
    return ret

def c37(row):
    ret = 0
    
    if int(row['eid']) < 0: 
        return ret

    if int(row['pid']) < 0: 
        return ret
    
    par1_dist_v = row['nueid_par1_cosmic_end_dist_end_v']
    par2_dist_v = row['nueid_par2_cosmic_end_dist_end_v']
    
    ret = 1
    
    if par1_dist_v.size == 0: 
        return ret

    if par2_dist_v.size == 0: 
        return ret
    
    par1_dist_v = par1_dist_v[par1_dist_v>0]
    par2_dist_v = par2_dist_v[par2_dist_v>0]
    
    if par1_dist_v.size == 0: 
        return ret

    if par2_dist_v.size == 0: 
        return ret
    
    ret = 0
    
    cut_v = [np.min(par1_dist_v), np.min(par2_dist_v)]
    cut = np.min(cut_v)

    if cut > 20:
        ret = 1

    return ret

def c38(row):
    ret = 0
    
    if int(row['eid']) < 0: 
        return ret
    
    key1 = "nueid_par1_triangle_height_U"
    key2 = "nueid_par1_triangle_baselength_U"

    key_v = multi_pparam_ratio(row,"eid",key1,key2)
    cut = key_v[0]

    if cut < 14:
        ret = 1
    
    return ret

def c39(row):
    ret = 0
   
    if int(row['eid']) < 0: 
        return ret

    if int(row['pid']) < 0: 
        return ret
    
    cut = row['anapid1_eminus_int_score'][2]
    
    if cut > 0.05:
        ret = 1

    return ret

def c40(row):
    ret = 0
    
    if int(row['eid']) < 0: 
        return ret
        
    if int(row['pid']) < 0:
        return ret

    cut = row['anapid1_eminus_int_score'][2]
    
    if cut > 0.1:
        ret = 1

    return ret

def c41(row):
    ret = 0
        
    if int(row['eid']) < 0: 
        return ret
        
    if int(row['pid']) < 0: 
        return ret
    
    cut = row['anapid1_eminus_int_score'][2]
    
    if cut > 0.6:
        ret = 1

    return ret

def c42(row):
    ret = 0

    if int(row['eid']) < 0: 
        return ret
    
    e_brem = max_pparam(row,"eid","nueid_par1_triangle_brem_U")

    if e_brem > 0:
        ret = 1;
    
    return ret


def c43(row):
    ret = 0

    if row['eid'] < 0: return ret

    dmax = 4
    mean_median_v = tdqdx_study(row,'eid',dmax)

    dqdx = float(mean_median_v[1])

    dedx = dqdx / float(72.66)
    
    # note this handles == -1 case 
    
    if dedx < 4: 
        ret = 1
    
    return ret


def c44(row):
    ret = 0

    if int(row["eid"]) < 0: 
        return ret

    if row['reco_electron_energy']>60:
        ret = 1
        
    return ret

def c45(row):
    ret = 0

    if int(row["pid"]) < 0: 
        return ret

    if row['reco_proton_energy']>60:
        ret = 1
    
    return ret
