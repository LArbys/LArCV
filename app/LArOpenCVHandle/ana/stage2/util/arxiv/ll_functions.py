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
    SS = "locv_nparticles==2 and anashr1_nshowers==2"
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

def prep_common_vars(df,ismc=True):
    
    res = pd.DataFrame()
    res = df.copy()

    if ismc == True:
        res['true_total_energy']   = res.apply(true_total_energy,axis=1)
        res['yplane_angle_p']      = res.apply(yplane_angle_p,axis=1)
        res['opening_angle']       = res.apply(opening_angle,axis=1)
        res['proton_yplane_len']   = res.apply(proton_yplane_len,axis=1)
        res['proton_beam_angle']   = res.apply(proton_beam_angle,axis=1)
        res['lepton_beam_angle']   = res.apply(lepton_beam_angle,axis=1)
        res['proton_planar_angle'] = res.apply(proton_planar_angle,axis=1)
        res['lepton_planar_angle'] = res.apply(lepton_planar_angle,axis=1)

        res['proton_momentum_X'] = res.apply(proton_momentum_X,axis=1)
        res['proton_momentum_Y'] = res.apply(proton_momentum_Y,axis=1)
        res['proton_momentum_Z'] = res.apply(proton_momentum_Z,axis=1)

        res['lepton_momentum_X'] = res.apply(lepton_momentum_X,axis=1)
        res['lepton_momentum_Y'] = res.apply(lepton_momentum_Y,axis=1)
        res['lepton_momentum_Z'] = res.apply(lepton_momentum_Z,axis=1)


    return res

def prep_LL_vars(df,ismc=True):

    lepton=3
    proton=9

    leptonpdg=11
    protonpdg=2212

    # 
    # reco per particle
    #
    df['p00_shr_dedx']            = df.apply(shower_reco_dedx,axis=1)
    df['p01_shr_theta']           = df.apply(shower_theta,axis=1)
    df['p02_shr_phi']             = df.apply(shower_phi,axis=1)
    df['p03_shr_length']          = df.apply(shower_length,axis=1)
    df['p04_shr_mean_pixel_dist'] = df.apply(shower_mean_pixel_dist,axis=1)
    df['p05_shr_width']           = df.apply(shower_width,axis=1)
    df['p06_shr_area']            = df.apply(shower_area,axis=1)
    df['p07_shr_qsum']            = df.apply(shower_qsum,axis=1)
    df['p08_shr_shower_frac']     = df.apply(shower_shower_frac,axis=1)

    if ismc == True:
        df['reco_mc_shower_energy'] = df.apply(reco_mc_shower_energy,axis=1)
        df['reco_mc_track_energy']  = df.apply(reco_mc_track_energy,axis=1)
        df['reco_mc_total_energy']  = df.apply(reco_mc_total_energy,axis=1)

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

def LL_proton(row):
    pLL = np.min([row['L_ep_p0'],row['L_ep_p1']])
    return pLL    

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
    df['LLp'] = df.apply(LL_proton,axis=1)
    
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

def LL_reco_parameters(df):
    
    df['reco_LL_proton_id']     = df.apply(reco_LL_proton_id,axis=1)
    df['reco_LL_proton_energy'] = df.apply(reco_LL_proton_energy,axis=1)
    df['reco_LL_proton_theta']  = df.apply(reco_LL_proton_theta,axis=1)
    df['reco_LL_proton_phi']    = df.apply(reco_LL_proton_phi,axis=1)
    df['reco_LL_proton_dEdx']   = df.apply(reco_LL_proton_dEdx,axis=1)
    df['reco_LL_proton_length'] = df.apply(reco_LL_proton_length,axis=1)
    df['reco_LL_proton_mean_pixel_dist'] = df.apply(reco_LL_proton_mean_pixel_dist,axis=1)
    df['reco_LL_proton_width']       = df.apply(reco_LL_proton_width,axis=1)
    df['reco_LL_proton_area']        = df.apply(reco_LL_proton_area,axis=1)
    df['reco_LL_proton_qsum']        = df.apply(reco_LL_proton_qsum,axis=1)
    df['reco_LL_proton_shower_frac'] = df.apply(reco_LL_proton_shower_frac,axis=1)
                

    df['reco_LL_electron_id']     = df.apply(reco_LL_electron_id,axis=1)
    df['reco_LL_electron_energy'] = df.apply(reco_LL_electron_energy,axis=1)
    df['reco_LL_electron_theta']  = df.apply(reco_LL_electron_theta,axis=1)
    df['reco_LL_electron_phi']    = df.apply(reco_LL_electron_phi,axis=1)
    df['reco_LL_electron_dEdx']   = df.apply(reco_LL_electron_dEdx,axis=1)
    df['reco_LL_electron_length'] = df.apply(reco_LL_electron_length,axis=1)
    df['reco_LL_electron_mean_pixel_dist'] = df.apply(reco_LL_electron_mean_pixel_dist,axis=1)
    df['reco_LL_electron_width']       = df.apply(reco_LL_electron_width,axis=1)
    df['reco_LL_electron_area']        = df.apply(reco_LL_electron_area,axis=1)
    df['reco_LL_electron_qsum']        = df.apply(reco_LL_electron_qsum,axis=1)
    df['reco_LL_electron_shower_frac'] = df.apply(reco_LL_electron_shower_frac,axis=1)

    df['reco_LL_total_energy']    = df.apply(reco_LL_total_energy,axis=1)

    return df
    

################################################################################
################################################################################

#
# LL Reco Variables
#
def shower_reco_dedx(row):
    ret = [-1.0]*2
    
    for i in xrange(len(ret)):
        if row['anashr1_reco_dedx_Y_v'][i] > 0:
            ret[i] = row['anashr1_reco_dedx_Y_v'][i]
        else:
            ret[i]  = row['anashr1_reco_dedx_U_v'][i]
            ret[i] += row['anashr1_reco_dedx_V_v'][i]
            ret[i] /= 2.0

    return ret

def shower_theta(row):
    ret = [0.0]*2
        
    for i in xrange(len(ret)):
        yy = row['anashr1_reco_dcosy_v'][i]
        zz = row['anashr1_reco_dcosz_v'][i]
        
        ret[i] = np.arctan2(yy,zz)

    return ret 

def shower_phi(row):
    ret = [0.0]*2
    
    for i in xrange(len(ret)):
        xx = row['anashr1_reco_dcosx_v'][i]
        zz = row['anashr1_reco_dcosz_v'][i]
        
        ret[i] = np.arctan2(xx,zz)

    return ret
    
def shower_length(row):
    ret = [0.0]*2
    
    for i in xrange(len(ret)):
        ret[i] = row['anashr1_reco_length_v'][i]
        
    return ret

def shower_mean_pixel_dist(row):
    ret = [0.0]*2
    
    for i in xrange(len(ret)):
        val_Y = row['locv_mean_pixel_dist_Y_v'][i]
        val_V = row['locv_mean_pixel_dist_V_v'][i]
        
        if val_Y>=0:
            ret[i] = val_Y
        elif val_V>=0:
            ret[i] = val_V
        else:
            raise Exception("ll function shower_mean_pixel_dist")
        
    return ret

def shower_width(row):
    ret = [0.0]*2
    
    for i in xrange(len(ret)):
        val_Y = row['locv_width_Y_v'][i]
        val_V = row['locv_width_V_v'][i]
        
        if val_Y>=0:
            ret[i] = val_Y
        elif val_V>=0:
            ret[i] = val_V
        else:
            raise Exception("ll function shower_width")
        
    return ret

def shower_area(row):
    ret = [0.0]*2
    
    for i in xrange(len(ret)):
        val_Y = row['locv_area_Y_v'][i]
        val_V = row['locv_area_V_v'][i]
        
        if val_Y>=0:
            ret[i] = val_Y
        elif val_V>=0:
            ret[i] = val_V
        else:
            raise Exception("ll function shower_area")

    return ret

def shower_qsum(row):
    ret = [0.0]*2
    
    for i in xrange(len(ret)):
        val_Y = row['locv_qsum_Y_v'][i]
        val_V = row['locv_qsum_V_v'][i]
        
        if val_Y>=0:
            ret[i] = val_Y
        elif val_V>=0:
            ret[i] = val_V
        else:
            raise Exception("ll function shower_qsum")

    return ret

def shower_shower_frac(row):
    ret = [0.0]*2
    
    for i in xrange(len(ret)):
        val_Y = row['locv_shower_frac_Y_v'][i]
        val_V = row['locv_shower_frac_V_v'][i]
        
        if val_Y>=0:
            ret[i] = val_Y
        elif val_V>=0:
            ret[i] = val_V
        else:
            raise Exception("ll function shower_shower_frac")

    return ret


def shower_opening(row):
    ret = 0.0

    xx0 = row['anashr1_reco_dcosx_v'][0]
    yy0 = row['anashr1_reco_dcosy_v'][0]
    zz0 = row['anashr1_reco_dcosz_v'][0]
    
    xx1 = row['anashr1_reco_dcosx_v'][1]
    yy1 = row['anashr1_reco_dcosy_v'][1]
    zz1 = row['anashr1_reco_dcosz_v'][1]
    
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
        eU = row['anashr1_reco_energy_U_v'][shrid]
        eV = row['anashr1_reco_energy_V_v'][shrid]
        eY = row['anashr1_reco_energy_Y_v'][shrid]
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
    
    totalE = row['locv_dep_sum_proton'] + row['anashr2_mc_energy']
    
    return totalE


#
# LL True Variables
#
def yplane_angle_p(row):

    protonpdg = 2212

    res = float(-1.0)

    lid = int(-1)
    pid = int(-1)

    lid_v = np.where(row['locv_daughterPdg_v']!=protonpdg)[0]
    if len(lid_v) > 0: lid = lid_v[0]

    pid_v = np.where(row['locv_daughterPdg_v']==protonpdg)[0]
    if len(pid_v) > 0: pid = pid_v[0]
    
    if lid < 0 or pid < 0: return res

    p_x = row['locv_daughterPx_v'][pid]
    p_z = row['locv_daughterPz_v'][pid]

    l_x = row['locv_daughterPx_v'][lid]
    l_z = row['locv_daughterPz_v'][lid]

    cos = p_x*l_x + p_z*l_z
    cos /= np.sqrt(p_x*p_x + p_z*p_z)
    cos /= np.sqrt(l_x*l_x + l_z*l_z)
    
    res = np.arccos(cos)*180./np.pi

    return res


def opening_angle(row):

    protonpdg = 2212

    res = float(-1.0)

    lid = int(-1)
    pid = int(-1)

    lid_v = np.where(row['locv_daughterPdg_v']!=protonpdg)[0]
    if len(lid_v) > 0: lid = lid_v[0]

    pid_v = np.where(row['locv_daughterPdg_v']==protonpdg)[0]
    if len(pid_v) > 0: pid = pid_v[0]
    
    if lid < 0 or pid < 0: return res

    p_x = row['locv_daughterPx_v'][pid]
    p_y = row['locv_daughterPx_v'][pid]
    p_z = row['locv_daughterPz_v'][pid]

    l_x = row['locv_daughterPx_v'][lid]
    l_y = row['locv_daughterPx_v'][lid]
    l_z = row['locv_daughterPz_v'][lid]

    cos = p_x*l_x + p_y*l_y + p_z*l_z
    cos /= np.sqrt(p_x*p_x + p_y*p_y + p_z*p_z)
    cos /= np.sqrt(l_x*l_x + l_y*l_y + l_z*l_z)
    
    res = np.arccos(cos)*180./np.pi

    return res


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

    res = float(-1.0)

    lid = int(-1)

    lid_v = np.where(row['locv_daughterPdg_v']!=protonpdg)[0]
    if len(lid_v) > 0: lid = lid_v[0]    

    if lid < 0 : return res

    l_x = row['locv_daughterPx_v'][lid]
    l_y = row['locv_daughterPy_v'][lid]
    l_z = row['locv_daughterPz_v'][lid]

    cos = l_z / np.sqrt(l_z*l_z + l_y*l_y)
    
    return np.arccos(cos)*180./np.pi

def lepton_momentum_X(row):
    protonpdg = 2212

    res = float(-1.0)

    lid = int(-1)

    lid_v = np.where(row['locv_daughterPdg_v']!=protonpdg)[0]
    if len(lid_v) > 0: lid = lid_v[0]    

    if lid < 0 : return res

    res = row['locv_daughterPx_v'][lid]

    return  res

def lepton_momentum_Y(row):
    protonpdg = 2212

    res = float(-1.0)

    lid = int(-1)

    lid_v = np.where(row['locv_daughterPdg_v']!=protonpdg)[0]
    if len(lid_v) > 0: lid = lid_v[0]    

    if lid < 0 : return res

    res = row['locv_daughterPy_v'][lid]

    return  res


def lepton_momentum_Z(row):
    protonpdg = 2212

    res = float(-1.0)

    lid = int(-1)

    lid_v = np.where(row['locv_daughterPdg_v']!=protonpdg)[0]
    if len(lid_v) > 0: lid = lid_v[0]    

    if lid < 0 : return res

    res = row['locv_daughterPz_v'][lid]

    return  res

def proton_beam_angle(row):
    protonpdg = 2212

    res = float(-1.0)

    pid = int(-1)

    pid_v = np.where(row['locv_daughterPdg_v']==protonpdg)[0]
    if len(pid_v) > 0: pid = pid_v[0]
    
    if pid < 0: return res

    p_x = row['locv_daughterPx_v'][pid]
    p_y = row['locv_daughterPy_v'][pid]
    p_z = row['locv_daughterPz_v'][pid]

    cos = p_z / np.sqrt(p_z*p_z + p_y*p_y)

    res = np.arccos(cos)*180./np.pi
    
    return res

def lepton_planar_angle(row):
    protonpdg = 2212
    
    res = float(-1)
    
    lid = int(-1)

    lid_v = np.where(row['locv_daughterPdg_v']!=protonpdg)[0]
    if len(lid_v) > 0 : lid = lid_v[0]

    if lid<0 : return res
    
    l_x = row['locv_daughterPx_v'][lid]
    l_y = row['locv_daughterPy_v'][lid]
    l_z = row['locv_daughterPz_v'][lid]

    cos = l_x / np.sqrt(l_z*l_z + l_x*l_x)
    
    res = np.arccos(cos)*180./np.pi

    return res

def proton_planar_angle(row):
    protonpdg = 2212

    res = float(-1)

    pid = int(-1)

    pid_v = np.where(row['locv_daughterPdg_v']==protonpdg)[0]
    if len(pid_v) > 0 : pid = pid_v[0]
    
    if pid<0 : return res

    p_x = row['locv_daughterPx_v'][pid]
    p_y = row['locv_daughterPy_v'][pid]
    p_z = row['locv_daughterPz_v'][pid]

    cos = p_x / np.sqrt(p_z*p_z + p_x*p_x)
    
    res = np.arccos(cos)*180./np.pi

    return res

def proton_yplane_len(row):
    dx = row['locv_proton_last_pt_x'] - row['locv_proton_1st_pt_x']
    dz = row['locv_proton_last_pt_z'] - row['locv_proton_1st_pt_z']
    return np.sqrt(dx*dx+dz*dz)

def proton_momentum_X(row):
    protonpdg = 2212

    res = float(-1)

    pid = int(-1)

    pid_v = np.where(row['locv_daughterPdg_v']==protonpdg)[0]
    if len(pid_v) > 0 : pid = pid_v[0]
    
    if pid<0 : return res

    res = row['locv_daughterPx_v'][pid]

    return res

def proton_momentum_Y(row):
    protonpdg = 2212

    res = float(-1)

    pid = int(-1)

    pid_v = np.where(row['locv_daughterPdg_v']==protonpdg)[0]
    if len(pid_v) > 0 : pid = pid_v[0]
    
    if pid<0 : return res

    res = row['locv_daughterPy_v'][pid]

    return res


def proton_momentum_Z(row):
    protonpdg = 2212

    res = float(-1)

    pid = int(-1)

    pid_v = np.where(row['locv_daughterPdg_v']==protonpdg)[0]
    if len(pid_v) > 0 : pid = pid_v[0]
    
    if pid<0 : return res

    res = row['locv_daughterPz_v'][pid]

    return res



def gaus(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


#
# proton
#
def reco_LL_proton_id(row):
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

def reco_LL_proton_energy(row):
    res = float(-1)    
    pid = int(row['reco_LL_proton_id'])
    if pid < 0: return res
    return float(row['anatrk2_E_proton_v'][pid])

def reco_LL_proton_dEdx(row):
    protonid  = int(row['reco_p_id'])
    return float(row['p00_shr_dedx'][protonid])

def reco_LL_proton_theta(row):
    protonid  = int(row['reco_p_id'])
    return float(row['p01_shr_theta'][protonid])

def reco_LL_proton_phi(row):
    protonid  = int(row['reco_p_id'])
    return float(row['p02_shr_phi'][protonid])

def reco_LL_proton_length(row):
    protonid  = int(row['reco_p_id'])
    return float(row['p03_shr_length'][protonid])

def reco_LL_proton_mean_pixel_dist(row):
    protonid  = int(row['reco_p_id'])
    return float(row['p04_shr_mean_pixel_dist'][protonid])

def reco_LL_proton_width(row):
    protonid  = int(row['reco_p_id'])
    return float(row['p05_shr_width'][protonid])

def reco_LL_proton_area(row):
    protonid  = int(row['reco_p_id'])
    return float(row['p06_shr_area'][protonid])

def reco_LL_proton_qsum(row):
    protonid  = int(row['reco_p_id'])
    return float(row['p07_shr_qsum'][protonid])

def reco_LL_proton_shower_frac(row):
    protonid  = int(row['reco_p_id'])
    return float(row['p08_shr_shower_frac'][protonid])

#
# electron
#
def reco_LL_electron_id(row):
    shrid  = int(row['reco_e_id'])
    return shrid

def reco_LL_electron_energy(row):
    shrid  = int(row['reco_LL_electron_id'])

    shrE = float(-1.0)

    eU = row['anashr1_reco_energy_U_v'][shrid]
    eV = row['anashr1_reco_energy_V_v'][shrid]
    eY = row['anashr1_reco_energy_Y_v'][shrid]
    
    if eY>0: shrE = eY
    else: shrE = (eU + eV) / 2.0
    return shrE


def reco_LL_electron_dEdx(row):
    electronid  = int(row['reco_LL_electron_id'])
    return float(row['p00_shr_dedx'][electronid])

def reco_LL_electron_theta(row):
    electronid  = int(row['reco_LL_electron_id'])
    return float(row['p01_shr_theta'][electronid])

def reco_LL_electron_phi(row):
    electronid  = int(row['reco_LL_electron_id'])
    return float(row['p02_shr_phi'][electronid])

def reco_LL_electron_length(row):
    electronid  = int(row['reco_LL_electron_id'])
    return float(row['p03_shr_length'][electronid])

def reco_LL_electron_mean_pixel_dist(row):
    electronid  = int(row['reco_LL_electron_id'])
    return float(row['p04_shr_mean_pixel_dist'][electronid])

def reco_LL_electron_width(row):
    electronid  = int(row['reco_LL_electron_id'])
    return float(row['p05_shr_width'][electronid])

def reco_LL_electron_area(row):
    electronid  = int(row['reco_LL_electron_id'])
    return float(row['p06_shr_area'][electronid])

def reco_LL_electron_qsum(row):
    electronid  = int(row['reco_LL_electron_id'])
    return float(row['p07_shr_qsum'][electronid])

def reco_LL_electron_shower_frac(row):
    electronid  = int(row['reco_LL_electron_id'])
    return float(row['p08_shr_shower_frac'][electronid])



#
# combined
#
def reco_LL_total_energy(row):
    res = float(-1)

    if row['reco_LL_proton_energy']  < 0: return res
    if row['reco_LL_electron_energy']< 0: return res

    res  = row['reco_LL_proton_energy']
    res += row['reco_LL_electron_energy']

    return res
