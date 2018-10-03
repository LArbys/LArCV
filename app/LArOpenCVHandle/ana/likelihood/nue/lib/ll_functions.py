import ROOT
import numpy as np
import root_numpy as rn
import pandas as pd
import sys, os
from cut_functions import *
import collections

def generate_ll_spec(sig_df,bkg_df,pdf_m):

    sig_spec_m = collections.OrderedDict()
    bkg_spec_m = collections.OrderedDict()

    for name, item in pdf_m.items():
        
        bins    = np.array(item[0])
        xlo,xhi = (bins[0],bins[-1])
        
        sig_data = sig_df[name].values.astype(np.float32)
        bkg_data = bkg_df[name].values.astype(np.float32)

        data_v = [sig_data,bkg_data]
        spec_v = [sig_spec_m,bkg_spec_m]

        for data, spec in zip(data_v,spec_v):
            
            data = data[data >= xlo]
            data = data[data <= xhi]

            data_h = np.histogram(data,bins=bins)
            
            dat   = data_h[0]
            edges = data_h[1][:-1]

            dat = np.where(dat==0,1,dat)
            
            dat_norm = dat / float(dat.sum())
            
            spec[name] = (edges.astype(np.float32),dat_norm)

    return (sig_spec_m,bkg_spec_m)


def apply_ll_vars(df,pdf_m,pid):
    
    for name, item in pdf_m.items():
        func = item[1]
        args = item[2]
        
        args[0] = pid
        
        df[name] = df.apply(func,args=tuple(args),axis=1).values
    
    return df

def fill_empty_ll_vars(df,pdf_m):
    
    for name, item in pdf_m.items():
        SS = str(name)
        df[SS] = np.nan
        SS = "LL"+SS
        df[SS] = np.nan

    return df

def define_LLem_vars():
    pdf_m = collections.OrderedDict()

    key0 = "em00"
    key1 = "nueid_par1_triangle_height_U"
    bb_v = np.arange(0,200+5,5)
    func = max_pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em01"
    key1 = "nueid_par1_triangle_height_U"
    bb_v = np.arange(5,150+5,5)
    func = min_pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)

    key0 = "em02"
    key1 = "nueid_par1_triangle_emptyarearatio_U"
    bb_v = np.arange(1,6+0.1,0.1)
    func = max_pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em03"
    key1 = "nueid_par1_triangle_emptyarearatio_U"
    bb_v = np.arange(0.5,4+0.1,0.1)
    func = min_pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em04"
    key1 = "nueid_par1_triangle_baselength_U"
    bb_v = np.arange(2,30+1,1)
    func = max_pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em05"
    key1 = "nueid_par1_triangle_baselength_U"
    bb_v = np.arange(2,20+1,1)
    func = min_pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em06"
    key1 = "nueid_par1_line_second_half_linefrac_U"
    bb_v = list(np.arange(0.4,1.0,0.1))+[0.95,0.99,1.0]
    func = max_pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em07"
    key1 = "nueid_par1_line_second_half_linefrac_U"
    func = min_pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em08"
    key1 = "nueid_par1_showerfrac_U"
    bb_v = np.arange(0,1.0+0.1,0.1)
    func = max_pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em09"
    key1 = "nueid_par1_showerfrac_U"
    bb_v = np.arange(0,1.0+0.1,0.1)
    func = min_pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em10"
    key1 = "nueid_par1_triangle_height_U"
    key2 = "nueid_par1_triangle_baselength_U"
    bb_v = np.arange(0,20+0.5,0.5)
    func = multi_pparam_ratio_min
    args = ["XXX",key1,key2,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em11"
    key1 = "nueid_par1_triangle_height_U"
    key2 = "nueid_par1_triangle_baselength_U"
    bb_v = np.arange(0,20+0.5,0.5)
    func = multi_pparam_ratio_max
    args = ["XXX",key1,key2,]
    pdf_m[key0] = (bb_v,func,args)

    key0 = "em12"
    key1 = "nueid_par1_polybranches_U_v"
    bb_v = np.arange(0,10+1,1)
    func = multi_pparam_max
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em13"
    key1 = "nueid_par1_polybranches_U_v"
    bb_v = np.arange(0,10+1,1)
    func = multi_pparam_min
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em14"
    key1 = "nueid_par1_polyedges_U_v"
    bb_v = np.arange(0,10+1,1)
    func = multi_pparam_max
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em15"
    key1 = "nueid_par1_polyedges_U_v"
    bb_v = np.arange(0,10+1,1)
    func = multi_pparam_min
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)

    key0 = "em16"
    key1 = "nueid_par1_triangle_brem_U"
    bb_v = np.arange(0,5+1,1)
    func = max_pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em17"
    key1 = "nueid_par1_triangle_brem_U"
    bb_v = np.arange(0,5+1,1)
    func = min_pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em18"
    key1 = "nueid_par1_numberdefects_U_v"
    bb_v = np.arange(0,5+1,1)
    func = multi_pparam_max
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em19"
    key1 = "nueid_par1_numberdefects_U_v"
    bb_v = np.arange(0,5+1,1)
    func = multi_pparam_min
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em20"
    key1 = "nueid_par1_largestdefect_U_v"
    bb_v = np.arange(0,30+1,1)
    func = multi_pparam_max
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "em21"
    key1 = "nueid_par1_largestdefect_U_v"
    bb_v = np.arange(0,20+1,1)
    func = multi_pparam_min
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)

    return pdf_m


def define_LLpc_vars():
    pdf_m = collections.OrderedDict()    

    key0 = "pc00"
    key1 = "nueid_par1_linefrac_U"
    bb_v = list(np.arange(0.4,1.0,0.1))+[0.95,0.99,1.0]
    func = max_pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)

    key0 = "pc01"
    key1 = "nueid_par1_linefrac_U"
    bb_v = list(np.arange(0.4,1.0,0.1))+[0.95,0.99,1.0]
    func = min_pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "pc02"
    key1 = "nueid_par1_dx1"
    bb_v = np.arange(-1,1+0.05,0.05)
    func = pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "pc03"
    key1 = "nueid_par1_dy1"
    bb_v = np.arange(-1,1+0.05,0.05)
    func = pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "pc04"
    key1 = "nueid_par1_dz1"
    bb_v = np.arange(-1,1+0.05,0.05)
    func = pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)

    key0 = "pc05"
    key1 = "ccqe_evis"
    bb_v = np.arange(-5,5+0.2,0.2)
    func = CCQE_EVIS
    args = ["XXX",]
    pdf_m[key0] = (bb_v,func,args)

    key0 = "pc06"
    key1 = "nueid_par1_showerfrac_U"
    bb_v = np.arange(0,1+0.1,0.1)
    func = max_pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)
    
    key0 = "pc07"
    key1 = "nueid_par1_showerfrac_U"
    bb_v = np.arange(0,1+0.1,0.1)
    func = min_pparam
    args = ["XXX",key1,]
    pdf_m[key0] = (bb_v,func,args)

    key0 = "pc08"
    key1 = "angle3d"
    bb_v = np.arange(0,180+10,10)
    func = angle3d
    args = ["XXX",]
    pdf_m[key0] = (bb_v,func,args)

    key0 = "pc09"
    key1 = "dqdx_plane_best"
    bb_v = np.arange(0,15+0.5,0.5)
    func = dqdx_plane_best
    args = ["XXX",2,1,]
    pdf_m[key0] = (bb_v,func,args)

    return pdf_m

def write_nue_pdfs(sig_spec_m,
                   bkg_spec_m,
                   out_name,
                   DIR_OUT="."):

    fout = os.path.join(DIR_OUT,out_name)
    tf = ROOT.TFile(fout,"RECREATE")
    tf.cd()
    
    assert len(sig_spec_m.keys()) == len(bkg_spec_m.keys());

    for key in sig_spec_m.keys():
    
        sig_spec = sig_spec_m[key]
        bkg_spec = bkg_spec_m[key]
        spec_v   = [sig_spec, bkg_spec]
        prefix_v = ["sig", "bkg"]

        for spec, prefix in zip(spec_v,prefix_v):
        
            spec_bins = spec[0].astype(np.float64)
            spec_data = spec[1].astype(np.float64)

            spec_bin_min = spec_bins[0]
            spec_bin_max = spec_bins[-1]

            spec_sz     = int(spec_bins.size)
            spec_bins   = list(spec_bins) + [spec_bin_max + 1]
            spec_bins   = np.array(spec_bins, dtype=np.float64)
            
            th  = None
            res = None
            
            th  = ROOT.TH1D(prefix + "_" + key,";;",spec_sz,spec_bins)
            res = rn.array2hist(spec_data,th)
            
            res.Write()
        

    tf.Close()

    return True

def read_nue_pdfs(fin):
    print "Reading PDFs @fin=%s"%fin
    tf_in = ROOT.TFile(fin,"READ")
    tf_in.cd()
    
    keys_v = [key.GetName() for key in tf_in.GetListOfKeys()]
    print keys_v

    sig_spec_m = collections.OrderedDict()
    bkg_spec_m = collections.OrderedDict()

    for key in keys_v:
        hist = tf_in.Get(key)
        arr = rn.hist2array(hist,return_edges=True)
        
        data = arr[0]
        bins = arr[1][0][:-1]
        
        assert data.sum() > 0.99999

        type_ = key.split("_")[0]
        
        param = None
        if type_ == "sig":
            param = "_".join(key.split("_")[1:])
            sig_spec_m[param] = (bins,data)

        elif type_ == "bkg":
            param = "_".join(key.split("_")[1:])
            bkg_spec_m[param] = (bins,data)
        else:
            raise Exception

    tf_in.Close()
    print "... read"

    return (sig_spec_m,bkg_spec_m)

def nearest_id_v(spectrum_v,value_v):
    assert len(spectrum_v) == len(value_v)
    ret_v = [0 for _ in xrange(len(value_v))]
    ix=-1
    for spectrum, value in zip(spectrum_v,value_v):
        ix+=1
        diff_v  = value - spectrum[0]
        mdiff_v = np.where(diff_v>=0,True,False)
        if mdiff_v.sum()==0: continue
        diff_v[~mdiff_v] = np.nan
        ret_v[ix] = np.nanargmin(diff_v)
    return ret_v


def reco_good(row,pt,et):
    res = [0,-1,-1,0.0,0.0]
    
    par1_planes_v = row['nueid_par1_planes_v']
    par2_planes_v = row['nueid_par2_planes_v']
    
    p1p = 0.0
    p1e = 0.0
    
    p2p = 0.0
    p2e = 0.0
    
    for ix in xrange(3):
        alpha = plane_to_alpha[ix]
        if par1_planes_v[ix] != 0:
            SS = "nueid_par1_electron_frac_%s" % alpha
            this_e = row[SS]
            SS = "nueid_par1_proton_frac_%s" % alpha
            this_p = row[SS]
            this_e = float(this_e)
            this_p = float(this_p)
            p1p += this_p
            p1e += this_e
                        
        if par2_planes_v[ix] != 0:
            SS = "nueid_par2_electron_frac_%s" % alpha
            this_e = row[SS]
            SS = "nueid_par2_proton_frac_%s" % alpha
            this_p = row[SS]
            this_e = float(this_e)
            this_p = float(this_p)
            p2p += this_p
            p2e += this_e
            
    p1p /= float(par1_planes_v.sum())
    p1e /= float(par1_planes_v.sum())
    
    p2p /= float(par2_planes_v.sum())
    p2e /= float(par2_planes_v.sum())
    
    e_v = [p1e,p2e]
    p_v = [p1p,p2p]
    
    eid = np.argmax(e_v)
    pid = np.argmax(p_v)
    
    if eid == pid: return res

    if e_v[eid] < et: return res
    if p_v[pid] < pt: return res

    res[0] = 1
    res[1] = eid+1
    res[2] = pid+1
    res[3] = e_v[eid]
    res[4] = p_v[pid]

    return res


def generate_llem_pdfs(df1,df2,mcinfo1,mcinfo2):
    
    print "Start"

    precut = "((c01==1)and(c02==1)and(c34==1)and(c36==1)and(c37==1)and(c44==1)and(c45==1))"
    SS1_   = "(locv_scedr<5 and locv_selected1L1P==1 and locv_energyInit<600)"
    SS2_   = "(mcinfo_scedr<5 and " + aNC + ")"

    df_sig = pd.read_pickle(df1)
    print "Read df_sig sz=",df_sig.index.size
    df_sig.query(precut,inplace=True)
    print "... after precut sz=",df_sig.index.size

    df_bkg = pd.read_pickle(df2)
    print "Read df_bkg sz=",df_bkg.index.size
    df_bkg.query(precut,inplace=True)
    print "... after precut sz=",df_bkg.index.size

    df_sig_mcinfo = pd.DataFrame(rn.root2array(mcinfo1,treename="EventMCINFO_DL"))
    print "Read df_sig_mcinfo =",df_sig_mcinfo.index.size

    df_bkg_mcinfo = pd.DataFrame(rn.root2array(mcinfo2,treename="EventMCINFO_DL"))
    print "Read df_bkg_mcinfo =",df_bkg_mcinfo.index.size

    df_sig.set_index(RSE,inplace=True)
    df_bkg.set_index(RSE,inplace=True)

    df_sig_mcinfo.set_index(RSE,inplace=True)
    df_bkg_mcinfo.set_index(RSE,inplace=True)
    
    df_sig = df_sig.join(df_sig_mcinfo,rsuffix="mcinfo_")
    df_bkg = df_bkg.join(df_bkg_mcinfo,rsuffix="mcinfo_")
    
    df_sig.reset_index(inplace=True)
    df_bkg.reset_index(inplace=True)

    df_sig['mcinfo_scedr'] = df_sig.apply(bless_scedr,axis=1)
    df_bkg['mcinfo_scedr'] = df_bkg.apply(bless_scedr,axis=1)
    
    df_sig.query(SS1_,inplace=True)
    df_bkg.query(SS2_,inplace=True)

    df_sig.sort_values(['mcinfo_scedr'],ascending=True).groupby(RSE).head(1)
    df_bkg.sort_values(['mcinfo_scedr'],ascending=True).groupby(RSE).head(1)

    pdf_m = define_LLem_vars()

    df_sig.drop(pdf_m.keys(), axis=1, inplace=True, errors='ignore')
    df_bkg.drop(pdf_m.keys(), axis=1, inplace=True, errors='ignore')
    
    df_sig['reco_good'] = df_sig.apply(reco_good,args=(0.5,0.5,),axis=1)
    df_sig['ggood']     = df_sig['reco_good'].str[0].values.astype(np.int32)
    df_sig['geid']      = df_sig['reco_good'].str[1].values.astype(np.int32)
    df_sig['gpid']      = df_sig['reco_good'].str[2].values.astype(np.int32)
    df_sig['gefrac']    = df_sig['reco_good'].str[3].values.astype(np.float32)
    df_sig['gpfrac']    = df_sig['reco_good'].str[4].values.astype(np.float32)
    
    df_sig.query("ggood==1", inplace=True)

    df_sig = apply_ll_vars(df_sig,pdf_m,"geid")
    df_bkg = apply_ll_vars(df_bkg,pdf_m,"eid")
    
    sig_spec_m, bkg_spec_m = generate_ll_spec(df_sig,df_bkg,pdf_m)

    write_nue_pdfs(sig_spec_m, bkg_spec_m, "revised_nue_pdfs.root")
    
    print "Done"
    
def generate_llpc_pdfs(df1,df2,mcinfo1):
    
    print "Start"
    
    precut = "((c01==1)and(c02==1)and(c34==1)and(c36==1)and(c37==1)and(c44==1)and(c45==1))"
    SS1_   = "(locv_scedr<5 and locv_selected1L1P==1 and locv_energyInit<600)"

    df_sig = pd.read_pickle(df1)
    print "Read df_sig sz=",df_sig.index.size
    df_sig.query(precut,inplace=True)
    print "... after precut sz=",df_sig.index.size

    df_bkg = pd.read_pickle(df2)
    print "Read df_bkg sz=",df_bkg.index.size
    df_bkg.query(precut,inplace=True)
    print "... after precut sz=",df_bkg.index.size

    df_sig_mcinfo = pd.DataFrame(rn.root2array(mcinfo1,treename="EventMCINFO_DL"))
    print "Read df_sig_mcinfo =",df_sig_mcinfo.index.size

    df_sig.set_index(RSE,inplace=True)
    df_sig_mcinfo.set_index(RSE,inplace=True)

    df_sig = df_sig.join(df_sig_mcinfo,rsuffix="mcinfo_")

    df_sig.reset_index(inplace=True)

    df_sig['mcinfo_scedr'] = df_sig.apply(bless_scedr,axis=1)

    pdf_m = define_LLpc_vars()

    df_sig.drop(pdf_m.keys(), axis=1, inplace=True, errors='ignore')
    df_bkg.drop(pdf_m.keys(), axis=1, inplace=True, errors='ignore')
    
    df_sig['reco_good'] = df_sig.apply(reco_good,args=(0.5,0.5,),axis=1)
    df_sig['ggood']     = df_sig['reco_good'].str[0].values.astype(np.int32)
    df_sig['geid']      = df_sig['reco_good'].str[1].values.astype(np.int32)
    df_sig['gpid']      = df_sig['reco_good'].str[2].values.astype(np.int32)
    df_sig['gefrac']    = df_sig['reco_good'].str[3].values.astype(np.float32)
    df_sig['gpfrac']    = df_sig['reco_good'].str[4].values.astype(np.float32)

    df_sig.query(SS1_,inplace=True)

    df_sig.query("ggood==1", inplace=True)

    df_sig.sort_values(['mcinfo_scedr'],ascending=True).groupby(RSE).nth(0)

    df_sig.reset_index(inplace=True)
    
    df_sig = apply_ll_vars(df_sig,pdf_m,"gpid")
    df_bkg = apply_ll_vars(df_bkg,pdf_m,"pid")
    
    sig_spec_m, bkg_spec_m = generate_ll_spec(df_sig,df_bkg,pdf_m)

    write_nue_pdfs(sig_spec_m, bkg_spec_m, "revised_LLpc_pdfs.root")
    
    print "Done"

def make_ll(df,sig_spec_m,bkg_spec_m):
    
    cols = sig_spec_m.keys()
    df_slice = df[cols].values

    sig_res = np.array([nearest_id_v(sig_spec_m.values(),v) for v in df_slice])
    bkg_res = np.array([nearest_id_v(bkg_spec_m.values(),v) for v in df_slice])

    sig_res_v = np.array([spectrum[1][v] for spectrum,v in zip(sig_spec_m.values(),sig_res.T)])
    bkg_res_v = np.array([spectrum[1][v] for spectrum,v in zip(bkg_spec_m.values(),bkg_res.T)])

    for col,sig,bkg in zip(cols,sig_res_v,bkg_res_v):
        col = "LL"+col
        LL = np.array([])
        LL = np.log(sig / bkg)
        df[col] = LL.astype(np.float32)

    return df

def apply_ll(COMB_DF, LLEP, LLPC):

    comb_df = pd.read_pickle(COMB_DF)

    llem_sig_spec_m, llem_bkg_spec_m = read_nue_pdfs(LLEP)
    llpc_sig_spec_m, llpc_bkg_spec_m = read_nue_pdfs(LLPC)

    out_df = comb_df.copy()
    
    input_size = out_df.index.size

    print "input_size=",input_size

    pdf_em_m = define_LLem_vars()
    pdf_pc_m = define_LLpc_vars()
    
    precut = "((c01==1)and(c02==1)and(c34==1)and(c36==1)and(c37==1)and(c44==1)and(c45==1))"

    out_df['passed_ll_precuts'] = int(0)
    df_precut = out_df.query(precut).copy()
    df_rest   = out_df.drop(df_precut.index).copy()

    # no events passed the precuts
    if df_precut.empty == True:
        print "No vertices passed"
        df_rest = fill_empty_ll_vars(df_rest,pdf_em_m)
        df_rest = fill_empty_ll_vars(df_rest,pdf_pc_m)

    else:
        df_precut['passed_ll_precuts'] = int(1)

        assert (df_precut.index.size + df_rest.index.size) == out_df.index.size
    
        df_precut = apply_ll_vars(df_precut,pdf_em_m,"eid")
        df_precut = apply_ll_vars(df_precut,pdf_pc_m,"pid")

        df_precut = make_ll(df_precut,llem_sig_spec_m,llem_bkg_spec_m)
        df_precut = make_ll(df_precut,llpc_sig_spec_m,llpc_bkg_spec_m)

    out_df = pd.concat([df_precut,df_rest],ignore_index=True)

    output_size = out_df.index.size

    assert input_size == output_size
    
    return out_df

def select_ll(in_df):
    out_df = in_df.copy()

    out_df['LLem_LL'] = np.nan
    out_df['LLpc_LL'] = np.nan

    nstep = int(10)
    for i in xrange(0,nstep):
        SS1_ = 'LLem_pass%02d' % i
        SS2_ = 'LLpc_pass%02d' % i
        out_df[SS1_] = 0
        out_df[SS2_] = 0

    out_df['ll_selected'] = 0

    pass_df = out_df.query("passed_ll_precuts==1").copy()
    fail_df = out_df.drop(pass_df.index).copy()

    if pass_df.empty == True:
        return out_df

    pass_df['LLem_LL'] = pass_df.apply(LLem_LL,axis=1)
    pass_df['LLpc_LL'] = pass_df.apply(LLpc_LL,axis=1)

    pass_df['LLem_pass_v'] = pass_df.apply(LLem_cut,args=(nstep,),axis=1)
    pass_df['LLpc_pass_v'] = pass_df.apply(LLpc_cut,args=(nstep,),axis=1)

    for i in xrange(0,nstep):
        SS1_ = 'LLem_pass%02d' % i
        SS2_ = 'LLpc_pass%02d' % i
        pass_df[SS1_] = pass_df['LLem_pass_v'].str[i].values
        pass_df[SS2_] = pass_df['LLpc_pass_v'].str[i].values

    pass_df.drop(["LLem_pass_v"],axis=1,inplace=True)
    pass_df.drop(["LLpc_pass_v"],axis=1,inplace=True)

    ll_index = pass_df.sort_values(['LLem_LL'],ascending=False).groupby(RSE).head(1).index

    out_df = pd.concat([pass_df,fail_df],ignore_index=True)
    
    out_df.loc[ll_index, 'll_selected'] = 1
    
    return out_df
    

LLem_v = ["LLem%02d" % i for i in xrange(22)]
def LLem_LL(row):
    ret = np.nan
    
    if row['LLem00'] != row['LLem00']:
        return ret

    data_v = row[LLem_v].astype(np.float32)
    ret = float(data_v.sum())
    
    return ret

LLpc_v = ["LLpc%02d" % i for i in xrange(10)]
def LLpc_LL(row):
    ret = np.nan
    
    if row['LLpc00'] != row['LLpc00']:
        return ret

    data_v = row[LLpc_v].astype(np.float32)
    ret = float(data_v.sum())
    
    return ret

#
# LLem discriminant
#
def LLemfunc(x,offset):
    return 2*np.sqrt(x-120)-22+offset

#
# LLem scan
#
def LLem_cut(row,nstep):
    ret_v = [0 for _ in xrange(nstep)]

    LL = row['LLem_LL']

    if LL != LL:
        return ret_v

    energy = float(row['reco_energy'])

    if energy>400:
        energy = 400
    
    for o in xrange(nstep):
        if LL > LLemfunc(energy,o):
            ret_v[o] = 1

    return ret_v

#
# LLpc scan
#
def LLpc_cut(row,nstep):
    ret_v = [0 for _ in xrange(nstep)]

    LL = row['LLpc_LL']

    if LL != LL:
        return ret_v

    for o in xrange(nstep):
        if LL > o:
            ret_v[o] = 1

    return ret_v


