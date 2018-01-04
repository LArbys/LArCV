import numpy as np
import pandas as pd
import ROOT
import root_numpy as rn
from common import *

#
# Prepare for LL 
#

def prep_working_df(df,copy=True):
    SS = "locv_num_vertex>0 and locv_scedr<5 and locv_selected1L1P==1"
    if copy == True:
        res = pd.DataFrame()
        res = df.copy()
        res.query(SS,inplace=True)
        return res
    else:
        df.query(SS,inplace=True)
        return df

def prep_test_df(df,copy=True):
    SS = "locv_num_vertex>0"
    if copy == True:
        res = pd.DataFrame()
        res = df.copy()
        res.query(SS,inplace=True)
        return res
    else:
        df.query(SS,inplace=True)
        return df

def prep_two_par_df(df,copy=True):
    SS = "locv_nparticles==2"
    if copy == True:
        res = pd.DataFrame()
        res = df.copy()
        res.query(SS,inplace=True)
        return res
    else:
        df.query(SS,inplace=True)
        return df

    
def prep_true_par_id(df,lepton):
    
    SS  = ""
    SS += "((locv_par_id_v.str[0] == 9 and locv_par_id_v.str[1] == @lepton)"
    SS += "or (locv_par_id_v.str[0] == @lepton and locv_par_id_v.str[1] == 9))"
    
    df.query(SS,inplace=True)
    
    df['par_protonid'] = df.apply(lambda x : 0 if(x['locv_par_id_v'][0]==9) else 1,axis=1)
    df['par_leptonid'] = df.apply(lambda x : 0 if(x['locv_par_id_v'][0]==lepton) else 1,axis=1)

    return df

def prep_LL_vars(df,ismc=True):

    lepton=3
    proton=9

    leptonpdg=11
    protonpdg=2212

    # 
    # reco per particle
    #
    df['p0_shr_dedx']  = df.apply(reco_dedx,axis=1)
    df['p1_shr_theta'] = df.apply(shower_theta,axis=1)
    df['p1_shr_phi']   = df.apply(shower_phi,axis=1)
    #df['p2_shr_open']  = df.apply(shower_opening,axis=1)


    #
    # mc stuff
    #
    if ismc == True:
        df['reco_mc_shower_energy'] = df.apply(reco_mc_shower_energy,axis=1)
        df['reco_mc_track_energy']  = df.apply(reco_mc_track_energy,axis=1)
        df['reco_mc_total_energy']  = df.apply(reco_mc_total_energy,axis=1)

        df['true_total_energy']   = df.apply(true_total_energy,axis=1)
        df['yplane_angle_p']      = df.apply(yplane_angle_p,axis=1)
        df['proton_yplane_len']   = df.apply(proton_yplane_len,axis=1)
        df['proton_beam_angle']   = df.apply(proton_beam_angle,axis=1)
        df['lepton_beam_angle']   = df.apply(lepton_beam_angle,axis=1)
        df['proton_planar_angle'] = df.apply(proton_planar_angle,axis=1)
        df['lepton_planar_angle'] = df.apply(lepton_planar_angle,axis=1)
        df['yplane_angle_p0']     = df.apply(yplane_angle_p0,axis=1)
        df['yplane_angle_p1']     = df.apply(yplane_angle_p1,axis=1)
        df['yplane_angle_p2']     = df.apply(yplane_angle_p2,axis=1)


    return df

#
# LL function
#

def LL_true(row,sig_spectrum_m,bkg_spectrum_m,istype):

    par_id_v = np.array([])

    if istype == 0:
        cols = row.values[:-2]
        cols = np.vstack(cols)
        par_id_v = row['par_leptonid']
        cols = cols[np.arange(len(cols)),par_id_v]

    elif istype == 1:
        cols = row.values[:-2]
        cols = np.vstack(cols)
        par_id_v = row['par_protonid']
        cols = cols[np.arange(len(cols)),par_id_v]

    elif istype == -1:
        cols = row.values
        cols = np.vstack(cols)
        cols = cols[:,0]

    sig_res = nearest_id_v(sig_spectrum_m.values(),cols)
    bkg_res = nearest_id_v(bkg_spectrum_m.values(),cols)

    sig_res = np.array([spectrum[1][v] for spectrum,v in zip(sig_spectrum_m.values(),sig_res)])
    bkg_res = np.array([spectrum[1][v] for spectrum,v in zip(bkg_spectrum_m.values(),bkg_res)])

    LL = np.log(sig_res / bkg_res)

    return np.sum(LL)


def LL_reco(row,sig_spectrum_m,bkg_spectrum_m,istype):

    par_id_v = np.array([])

    if istype >= 0:
        cols = row.values
        cols = np.vstack(cols)
        cols = cols[:,istype]

    elif istype == -1:
        cols = row.values[:-2]
        cols = np.vstack(cols)
        par_id_v = row['reco_e_id']
        cols = cols[np.arange(len(cols)),par_id_v]

    elif istype == -2:
        cols = row.values[:-2]
        cols = np.vstack(cols)
        par_id_v = row['reco_p_id']
        cols = cols[np.arange(len(cols)),par_id_v]
    
    sig_res = nearest_id_v(sig_spectrum_m.values(),cols)
    bkg_res = nearest_id_v(bkg_spectrum_m.values(),cols)

    sig_res = np.array([spectrum[1][v] for spectrum,v in zip(sig_spectrum_m.values(),sig_res)])
    bkg_res = np.array([spectrum[1][v] for spectrum,v in zip(bkg_spectrum_m.values(),bkg_res)])

    LL = np.log(sig_res / bkg_res)

    return np.sum(LL)


#
# LL file rw 
#
def read_nue_pdfs(fin):
    print "Reading PDFs @fin=%s"%fin
    tf_in = ROOT.TFile(fin,"READ")
    tf_in.cd()
    
    keys_v = [key.GetName() for key in tf_in.GetListOfKeys()]
    
    lepton_spec_m = {}
    proton_spec_m = {}
    cosmic_spec_m = {}
    
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
        if type_ == "lepton":
            param = "_".join(key.split("_")[1:])
            lepton_spec_m[param] = (centers,data)

        elif type_ == "proton":
            param = "_".join(key.split("_")[1:])
            proton_spec_m[param] = (centers,data)

        elif type_ == "cosmic":
            param = "_".join(key.split("_")[1:])
            cosmic_spec_m[param] = (centers,data)

        else:
            raise Exception

    tf_in.Close()
    print "... read"

    print "Asserting..."
    #
    # Assert sig. and bkg. read from file correctly
    #
    for key in lepton_spec_m.keys():
        assert key in lepton_spec_m.keys(), "key=%s missing from lep"%key
        assert key in proton_spec_m.keys(), "key=%s missing from pro"%key
        assert key in cosmic_spec_m.keys(), "key=%s missing from cos"%key
        assert lepton_spec_m[key][0].size == proton_spec_m[key][0].size
        assert lepton_spec_m[key][1].size == proton_spec_m[key][1].size
        assert lepton_spec_m[key][0].size == cosmic_spec_m[key][0].size
        assert lepton_spec_m[key][1].size == cosmic_spec_m[key][1].size
        assert proton_spec_m[key][0].size == cosmic_spec_m[key][0].size
        assert proton_spec_m[key][1].size == cosmic_spec_m[key][1].size

    for key in proton_spec_m.keys():
        assert key in lepton_spec_m.keys(), "key=%s missing from lep"%key
        assert key in proton_spec_m.keys(), "key=%s missing from pro"%key
        assert key in cosmic_spec_m.keys(), "key=%s missing from cos"%key
        assert lepton_spec_m[key][0].size == proton_spec_m[key][0].size
        assert lepton_spec_m[key][1].size == proton_spec_m[key][1].size
        assert lepton_spec_m[key][0].size == cosmic_spec_m[key][0].size
        assert lepton_spec_m[key][1].size == cosmic_spec_m[key][1].size
        assert proton_spec_m[key][0].size == cosmic_spec_m[key][0].size
        assert proton_spec_m[key][1].size == cosmic_spec_m[key][1].size

    for key in cosmic_spec_m.keys():
        assert key in lepton_spec_m.keys(), "key=%s missing from lep"%key
        assert key in proton_spec_m.keys(), "key=%s missing from pro"%key
        assert key in cosmic_spec_m.keys(), "key=%s missing from cos"%key
        assert lepton_spec_m[key][0].size == proton_spec_m[key][0].size
        assert lepton_spec_m[key][1].size == proton_spec_m[key][1].size
        assert lepton_spec_m[key][0].size == cosmic_spec_m[key][0].size
        assert lepton_spec_m[key][1].size == cosmic_spec_m[key][1].size
        assert proton_spec_m[key][0].size == cosmic_spec_m[key][0].size
        assert proton_spec_m[key][1].size == cosmic_spec_m[key][1].size

    print "... asserted"

    return (lepton_spec_m,proton_spec_m,cosmic_spec_m)

def read_line_file(fin):
    print "Reading line @fin=%s"%fin
    tf_in = ROOT.TFile(fin,"READ")
    tf_in.cd()
    
    line = tf_in.Get("LL_line")
    
    res = [0.0,0.0]
    res[0] = float(line.GetParameter(0))
    res[1] = float(line.GetParameter(1))

    tf_in.Close()

    return res

def write_line_file(line_param,DIR_OUT="."):
    fout = os.path.join(DIR_OUT,"nue_line_file.root")
    tf = ROOT.TFile(fout,"RECREATE")
    tf.cd()

    line = ROOT.TF1("LL_line","[0]*x+[1]",0,1)
    line.SetParameter(0,float(line_param[0]))
    line.SetParameter(1,float(line_param[1]))

    line.Write()

    tf.Close()

def write_nue_pdfs(lepton_spec_m,proton_spec_m,cosmic_spec_m,DIR_OUT="."):

    fout = os.path.join(DIR_OUT,"nue_pdfs.root")
    tf = ROOT.TFile(fout,"RECREATE")
    tf.cd()
    
    assert len(lepton_spec_m.keys()) == len(proton_spec_m.keys())
    assert len(lepton_spec_m.keys()) == len(cosmic_spec_m.keys())
    assert len(proton_spec_m.keys()) == len(cosmic_spec_m.keys())

    for key in lepton_spec_m.keys():
    
        lep_spec = lepton_spec_m[key]
        pro_spec = proton_spec_m[key]
        cos_spec = cosmic_spec_m[key]
        
        #
        # Write electron
        #
        lep_sz   = lep_spec[0].size
        lep_dx   = (lep_spec[0][1] - lep_spec[0][0]) / 2.0
        lep_bins = list(lep_spec[0] - lep_dx) + [lep_spec[0][-1] + lep_dx]
        lep_bins = np.array(lep_bins,dtype=np.float64)
        lep_data = lep_spec[1].astype(np.float64)
        
        th = None
        th = ROOT.TH1D("lepton_" + key,";;",lep_sz,lep_bins)
        res = None
        res = rn.array2hist(lep_data,th)
        
        res.Write()
        
        #
        # Write proton
        #
        pro_sz   = pro_spec[0].size
        pro_dx   = (pro_spec[0][1] - pro_spec[0][0]) / 2.0
        pro_bins = list(pro_spec[0] - pro_dx) + [pro_spec[0][-1] + pro_dx]
        pro_bins = np.array(pro_bins,dtype=np.float64)
        pro_data = pro_spec[1].astype(np.float64)
        
        th = None
        th = ROOT.TH1D("proton_" + key,";;",pro_sz,pro_bins)
        res = None
        res = rn.array2hist(pro_data,th)
        
        res.Write()
        
        #
        # Write cosmic
        #
        cos_sz   = cos_spec[0].size
        cos_dx   = (cos_spec[0][1] - cos_spec[0][0]) / 2.0
        cos_bins = list(cos_spec[0] - cos_dx) + [cos_spec[0][-1] + cos_dx]
        cos_bins = np.array(cos_bins,dtype=np.float64)
        cos_data = cos_spec[1].astype(np.float64)
        
        th = None
        th = ROOT.TH1D("cosmic_" + key,";;",cos_sz,cos_bins)
        res = None
        res = rn.array2hist(cos_data,th)
        
        res.Write()

    tf.Close()

    return True

#
# LL reco particle+nue selector
#

def LL_par_id(row):
    e_id = np.argmax([row['L_ep_p0'],row['L_ep_p1']])
    p_id = np.argmin([row['L_ep_p0'],row['L_ep_p1']])
    return [e_id,p_id]

def LL_elec(row):
    eLL = np.max([row['L_ep_p0'],row['L_ep_p1']])
    return eLL    

def LL_truth_particle(df,lepton_spec_m,proton_spec_m):
    var_slice   = proton_spectrum_m.keys() + ['par_protonid','par_leptonid']

    L_e_ep = df[var_slice].apply(LL_true,args=(lepton_spec_m,proton_spec_m,0,),axis=1).values
    L_p_ep = df[var_slice].apply(LL_true,args=(lepton_spec_m,proton_spec_m,1,),axis=1).values

    res = None
    res = np.vstack((L_e_ep,L_p_ep)).T
    
    return res

def LL_reco_particle(df,lepton_spec_m,proton_spec_m):

    var_slice = proton_spec_m.keys()

    df['L_ep_p0'] = df[var_slice].apply(LL_reco,args=(lepton_spec_m,proton_spec_m,0,),axis=1).values
    df['L_ep_p1'] = df[var_slice].apply(LL_reco,args=(lepton_spec_m,proton_spec_m,1,),axis=1).values

    res = None
    res = df.apply(LL_par_id,axis=1)

    df['reco_e_id'] = res.str[0]
    df['reco_p_id'] = res.str[1]
    df['LLe'] = df.apply(LL_elec,axis=1)
    
    return df

def LL_reco_nue(df,lepton_spec_m,proton_spec_m,cosmic_spec_m):
    
    var_slice = proton_spec_m.keys() + ['reco_e_id','reco_p_id']

    df['L_ec_e'] = df[var_slice].apply(LL_reco,args=(lepton_spec_m,cosmic_spec_m,-1,),axis=1).values
    df['L_pc_p'] = df[var_slice].apply(LL_reco,args=(proton_spec_m,cosmic_spec_m,-2,),axis=1).values

    return df


def pt_distance(pt,a,b,c):
    ret = float(0.0)
    ret = a*pt[0]+b*pt[1]+c
    ret /= np.sqrt(a*a+b*b)
    return ret

def LL_reco_line(df,line):

    a = -1.0 * line[0]
    b =  1.0
    c = -1.0 * line[1]

    df['LL_dist']  = np.array([pt_distance(pt,a,b,c) for pt in df[['L_ec_e','L_pc_p']].values])
    
    return df
    

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

#
# LL Reco Variables
#
def reco_dedx(row):
    ret = [-1.0]*2

    for i in xrange(len(ret)):
        if row['anashr_reco_dedx_Y_v'][i] > 0:
            ret[i] = row['anashr_reco_dedx_Y_v'][i]
        else:
            ret[i]  = row['anashr_reco_dedx_U_v'][i]
            ret[i] += row['anashr_reco_dedx_V_v'][i]
            ret[i] /= 2.0

    return ret

def shower_theta(row):
    ret = [0.0]*2
        
    for i in xrange(len(ret)):
        yy = row['anashr_reco_dcosy_v'][i]
        zz = row['anashr_reco_dcosz_v'][i]
        
        ret[i] = np.arctan2(yy,zz)

    return ret 

def shower_phi(row):
    ret = [0.0]*2
    
    for i in xrange(len(ret)):
        xx = row['anashr_reco_dcosx_v'][i]
        zz = row['anashr_reco_dcosz_v'][i]
        
        ret[i] = np.arctan2(xx,zz)

    return ret
    
def shower_opening(row):
    ret = 0.0

    xx0 = row['anashr_reco_dcosx_v'][0]
    yy0 = row['anashr_reco_dcosy_v'][0]
    zz0 = row['anashr_reco_dcosz_v'][0]
    
    xx1 = row['anashr_reco_dcosx_v'][1]
    yy1 = row['anashr_reco_dcosy_v'][1]
    zz1 = row['anashr_reco_dcosz_v'][1]
    
    ret = xx0*xx1+yy0*yy1+zz0*zz1
    ret/= np.sqrt(xx0*xx0+yy0*yy0+zz0*zz0)
    ret/= np.sqrt(xx1*xx1+yy1*yy1+zz1*zz1)
        
    ret = np.arccos(ret)

    return ret

def reco_mc_shower_energy(row):
    lepton=3
    shrid_v = np.where(row['mchshr_shr_type_v']==lepton)[0]

    shrE = float(-1.0)
        
    shrid = int(-1)

    if shrid_v.size==1:
        shrid = shrid_v[0]
    elif shrid_v.size>1:
        shrid = np.argmax(row['mchshr_electronfrac_v'])
        
    if shrid != -1:
        eU = row['anashr_reco_energy_U_v'][shrid]
        eV = row['anashr_reco_energy_V_v'][shrid]
        eY = row['anashr_reco_energy_Y_v'][shrid]
        if eY>0: shrE = eY
        else: shrE = (eU + eV) / 2.0
    
    return shrE

def reco_mc_track_energy(row):
    proton=9
    trkid_v = np.where(row['mchtrk_trk_type_v']==proton)[0]

    trkE = float(-1.0)
        
    trkid = int(-1)

    if trkid_v.size==1:
        trkid = trkid_v[0]
    elif trkid_v.size>1:
        trkid = np.argmax(row['mchtrk_protonfrac_v'])
        
    if trkid != -1:
        trkE = row['anatrk2_E_proton_v'][trkid]
    
    return trkE

def reco_mc_total_energy(row):
    totalE = float(-1.0)
    
    if row['reco_mc_track_energy']<0 : return totalE
    if row['reco_mc_shower_energy']<0 : return totalE
    
    totalE = row['reco_mc_track_energy'] + row['reco_mc_shower_energy']
    
    return totalE

def true_total_energy(row):
    totalE = float(-1)
    
    totalE = row['locv_dep_sum_proton'] + row['anashr_mc_energy']
    
    return totalE


#
# LL True Variables
#
def yplane_angle_p(row):
    protonpdg = 2212

    lid = np.where(row['locv_daughterPdg_v']!=protonpdg)[0][0]
    pid = np.where(row['locv_daughterPdg_v']==protonpdg)[0][0]

    p_x = row['locv_daughterPx_v'][pid]
    p_z = row['locv_daughterPz_v'][pid]

    l_x = row['locv_daughterPx_v'][lid]
    l_z = row['locv_daughterPz_v'][lid]

    cos = p_x*l_x + p_z*l_z
    cos /= np.sqrt(p_x*p_x + p_z*p_z)
    cos /= np.sqrt(l_x*l_x + l_z*l_z)
    
    return np.arccos(cos)*180./np.pi



def yplane_angle_p0(row):
    protonpdg = 2212

    lid = int(np.where(row['locv_daughterPdg_v']!=protonpdg)[0][0])
    pid = int(np.where(row['locv_daughterPdg_v']==protonpdg)[0][0])


    p_x =  float(row['locv_daughter2DEndX_vv'][0][pid])
    p_z =  float(row['locv_daughter2DEndY_vv'][0][pid])
    p_x -= float(row['locv_daughter2DStartX_vv'][0][pid])
    p_z -= float(row['locv_daughter2DStartY_vv'][0][pid])

    l_x =  float(row['locv_daughter2DEndX_vv'][0][lid])
    l_z =  float(row['locv_daughter2DEndY_vv'][0][lid])
    l_x -= float(row['locv_daughter2DStartX_vv'][0][lid])
    l_z -= float(row['locv_daughter2DStartY_vv'][0][lid])

    cos = p_x*l_x + p_z*l_z
    cos /= np.sqrt(p_x*p_x + p_z*p_z)
    cos /= np.sqrt(l_x*l_x + l_z*l_z)
    
    return np.arccos(cos)*180./np.pi


def yplane_angle_p1(row):
    protonpdg = 2212

    lid = int(np.where(row['locv_daughterPdg_v']!=protonpdg)[0][0])
    pid = int(np.where(row['locv_daughterPdg_v']==protonpdg)[0][0])


    p_x =  float(row['locv_daughter2DEndX_vv'][1][pid])
    p_z =  float(row['locv_daughter2DEndY_vv'][1][pid])
    p_x -= float(row['locv_daughter2DStartX_vv'][1][pid])
    p_z -= float(row['locv_daughter2DStartY_vv'][1][pid])

    l_x =  float(row['locv_daughter2DEndX_vv'][1][lid])
    l_z =  float(row['locv_daughter2DEndY_vv'][1][lid])
    l_x -= float(row['locv_daughter2DStartX_vv'][1][lid])
    l_z -= float(row['locv_daughter2DStartY_vv'][1][lid])
    

    cos = p_x*l_x + p_z*l_z
    cos /= np.sqrt(p_x*p_x + p_z*p_z)
    cos /= np.sqrt(l_x*l_x + l_z*l_z)
    
    return np.arccos(cos)*180./np.pi


def yplane_angle_p2(row):
    protonpdg = 2212

    lid = int(np.where(row['locv_daughterPdg_v']!=protonpdg)[0][0])
    pid = int(np.where(row['locv_daughterPdg_v']==protonpdg)[0][0])


    p_x =  float(row['locv_daughter2DEndX_vv'][2][pid])
    p_z =  float(row['locv_daughter2DEndY_vv'][2][pid])
    p_x -= float(row['locv_daughter2DStartX_vv'][2][pid])
    p_z -= float(row['locv_daughter2DStartY_vv'][2][pid])

    l_x =  float(row['locv_daughter2DEndX_vv'][2][lid])
    l_z =  float(row['locv_daughter2DEndY_vv'][2][lid])
    l_x -= float(row['locv_daughter2DStartX_vv'][2][lid])
    l_z -= float(row['locv_daughter2DStartY_vv'][2][lid])
    

    cos = p_x*l_x + p_z*l_z
    cos /= np.sqrt(p_x*p_x + p_z*p_z)
    cos /= np.sqrt(l_x*l_x + l_z*l_z)
    
    return np.arccos(cos)*180./np.pi


def lepton_beam_angle(row):
    protonpdg = 2212

    lid = np.where(row['locv_daughterPdg_v']!=protonpdg)[0][0]
    
    l_x = row['locv_daughterPx_v'][lid]
    l_y = row['locv_daughterPy_v'][lid]
    l_z = row['locv_daughterPz_v'][lid]

    cos = l_z / np.sqrt(l_z*l_z + l_y*l_y)
    
    return np.arccos(cos)*180./np.pi

def proton_beam_angle(row):
    protonpdg = 2212

    pid = np.where(row['locv_daughterPdg_v']==protonpdg)[0][0]
    
    p_x = row['locv_daughterPx_v'][pid]
    p_y = row['locv_daughterPy_v'][pid]
    p_z = row['locv_daughterPz_v'][pid]

    cos = p_z / np.sqrt(p_z*p_z + p_y*p_y)
    
    return np.arccos(cos)*180./np.pi

def lepton_planar_angle(row):
    protonpdg = 2212

    lid = np.where(row['locv_daughterPdg_v']!=protonpdg)[0][0]
    
    l_x = row['locv_daughterPx_v'][lid]
    l_y = row['locv_daughterPy_v'][lid]
    l_z = row['locv_daughterPz_v'][lid]

    cos = l_x / np.sqrt(l_z*l_z + l_x*l_x)
    
    return np.arccos(cos)*180./np.pi

def proton_planar_angle(row):
    protonpdg = 2212

    pid = np.where(row['locv_daughterPdg_v']==protonpdg)[0][0]
    
    p_x = row['locv_daughterPx_v'][pid]
    p_y = row['locv_daughterPy_v'][pid]
    p_z = row['locv_daughterPz_v'][pid]

    cos = p_x / np.sqrt(p_z*p_z + p_x*p_x)
    
    return np.arccos(cos)*180./np.pi

def proton_yplane_len(row):
    dx = row['locv_proton_last_pt_x'] - row['locv_proton_1st_pt_x']
    dz = row['locv_proton_last_pt_z'] - row['locv_proton_1st_pt_z']
    return np.sqrt(dx*dx+dz*dz)


def gaus(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


#
# Fill variables
#

