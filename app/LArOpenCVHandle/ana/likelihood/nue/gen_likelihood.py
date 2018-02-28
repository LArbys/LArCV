import collections

def define_LL_line():
    p0 = -0.9
    p1 =  2.5
    return (p0,p1)

def define_LL_vars():
    pdf_m = collections.OrderedDict()

    xlo= 0
    xhi= 10
    dx = 0.2
    key="p00_shr_dedx"
    pdf_m[key] = ((xlo,xhi,dx),key)

    xlo= -3.14
    xhi= 3.14
    dx = 3.14/25.0
    key="p01_shr_theta"
    pdf_m[key] = ((xlo,xhi,dx),key)

    xlo= -3.14
    xhi= 3.14
    dx = 3.14/25.0
    key="p02_shr_phi"
    pdf_m[key] = ((xlo,xhi,dx),key)

    xlo= 0
    xhi= 500
    dx = 10
    key='p03_shr_length'
    pdf_m[key] = ((xlo,xhi,dx),key)

    xlo= 0
    xhi= 10
    dx = 0.2
    key="p04_shr_mean_pixel_dist"
    pdf_m[key] = ((xlo,xhi,dx),key)

    xlo= 0
    xhi= 60
    dx = 1
    key="p05_shr_width"
    pdf_m[key] = ((xlo,xhi,dx),key)

    xlo= 0
    xhi= 1000
    dx = 20
    key="p06_shr_area"
    pdf_m[key] = ((xlo,xhi,dx),key)
    
    xlo= 0
    xhi= 80000
    dx = 2000
    key="p07_shr_qsum"
    pdf_m[key] = ((xlo,xhi,dx),key)
    
    xlo= 0
    xhi= 1.0
    dx = 0.05
    key="p08_shr_shower_frac"
    pdf_m[key] = ((xlo,xhi,dx),key)
    
    return pdf_m

if __name__ == '__main__':

    import os, sys, gc

    if len(sys.argv) != 4:
        print 
        print "sample_name0 = \"nue\""
        print "sample_file0 = str(sys.argv[1])"
        print
        print "sample_name1 = \"cosmic\""
        print "sample_file1 = str(sys.argv[2])"
        print
        print "OUT_DIR = str(sys.argv[3])"
        print 
        sys.exit(1)
        
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    import numpy as np
    import pandas as pd
    
    import ROOT
    import root_numpy as rn
    
    from util.common import *
    from util.ll_functions import *

    BASE_PATH = os.path.realpath(__file__)
    BASE_PATH = os.path.dirname(BASE_PATH)

    sample_name0 = "nue"
    sample_file0 = str(sys.argv[1])
    
    sample_name1 = "cosmic"
    sample_file1 = str(sys.argv[2])

    OUT_DIR = str(sys.argv[3])

    dfs = {}

    print "Reading..."
    for name,file_ in [(sample_name0,sample_file0),
                       (sample_name1,sample_file1)]:
    
        print "@name=",name
        comb_df = pd.DataFrame()
        comb_df = pd.read_pickle(file_)
        dfs[name] = comb_df.copy()
        del comb_df
        gc.collect()


    print "Prep..."
    dfs["nue"]    = prep_working_df(dfs["nue"],copy=False)
    dfs["cosmic"] = prep_test_df(dfs["cosmic"],copy=False)
    
    dfs["nue"]    = prep_two_par_df(dfs["nue"],copy=False)
    dfs["cosmic"] = prep_two_par_df(dfs["cosmic"],copy=False)
    
    dfs["nue"]    = prep_LL_vars(dfs["nue"],ismc=True)
    dfs["cosmic"] = prep_LL_vars(dfs["cosmic"],ismc=False)
    print "... preped"
    
    print "ID e&p..."
    lepton = int(3)
    dfs["nue"] = prep_true_par_id(dfs["nue"],lepton)

    leptonid_v = np.array(dfs["nue"]["par_leptonid"].values.astype(np.int32))
    protonid_v = np.array(dfs["nue"]["par_protonid"].values.astype(np.int32))
    print "...IDed"
    
    lepton_spec_m = {}
    proton_spec_m = {}
    cosmic_spec_m = {}
    
    DRAW = False

    print "Define..."
    pdf_m = define_LL_vars()
    line_param = define_LL_line()
    print "... defined"

    "Gen PDF..."
    for key,item in pdf_m.items():
        xlo,xhi,dx = item[0]
        name       = item[1]

        data_v = dfs["nue"][key]
        data_v = np.vstack(data_v)

        proton_data = data_v[np.arange(len(data_v)),protonid_v]
        lepton_data = data_v[np.arange(len(data_v)),leptonid_v]
        
        data0 = lepton_data.astype(np.float32)
        data0 = data0[data0 >= xlo]
        data0 = data0[data0 <= xhi]
        
        data1 = proton_data.astype(np.float32)
        data1 = data1[data1 >= xlo]
        data1 = data1[data1 <= xhi]
        
        data_v = dfs["cosmic"][key]
        data_v = np.hstack(data_v)
        
        data2 = data_v.astype(np.float32)
        data2 = data2[data2 >= xlo]
        data2 = data2[data2 <= xhi]
        
        lepton_h = np.histogram(data0,bins=np.arange(xlo,xhi+dx,dx))
        proton_h = np.histogram(data1,bins=np.arange(xlo,xhi+dx,dx))
        cosmic_h = np.histogram(data2,bins=np.arange(xlo,xhi+dx,dx))
        
        lep = lepton_h[0]  
        pro = proton_h[0]
        cos = cosmic_h[0]
        
        lep = np.where(lep==0,1,lep)
        pro = np.where(pro==0,1,pro)
        cos = np.where(cos==0,1,cos)
        
        centers=proton_h[1] + (proton_h[1][1] - proton_h[1][0]) / 2.0
        centers = centers[:-1]
        
        lep_norm = lep / float(lep.sum())
        pro_norm = pro / float(pro.sum())
        cos_norm = cos / float(cos.sum())
        
        lep_err = np.sqrt(lep)
        pro_err = np.sqrt(pro)
        cos_err = np.sqrt(cos)
        
        lep_err_norm = lep_err / float(lep.sum())
        pro_err_norm = pro_err / float(pro.sum())
        cos_err_norm = cos_err / float(cos.sum())
        
        lepton_spec_m[key] = (centers,lep_norm)
        proton_spec_m[key] = (centers,pro_norm)
        cosmic_spec_m[key] = (centers,cos_norm)
        
        if DRAW==False: continue
        
        fig,ax=plt.subplots(figsize=(10,6))
        data = proton_h[1][:-1]
        bins = proton_h[1]
        centers = data + (data[1] - data[0])/2.0
        
        ax.hist(data,bins=bins,weights=cos_norm,histtype='stepfilled',color='black',lw=1,alpha=0.1)
        ax.hist(data,bins=bins,weights=cos_norm,histtype='step',color='black',lw=2,label='Cosmic')
        
        ax.hist(data,bins=bins,weights=pro_norm,histtype='stepfilled',color='red',lw=1,alpha=0.1)
        ax.hist(data,bins=bins,weights=pro_norm,histtype='step',color='red',lw=2,label='Proton')

        ax.hist(data,bins=bins,weights=lep_norm,histtype='stepfilled',color='blue',lw=1,alpha=0.1)
        ax.hist(data,bins=bins,weights=lep_norm,histtype='step',color='blue',lw=2,label='Electron')

        ax.errorbar(centers,pro_norm,yerr=pro_err_norm,fmt='o',color='red',markersize=0,lw=2)
        ax.errorbar(centers,lep_norm,yerr=lep_err_norm,fmt='o',color='blue',markersize=0,lw=2)
        ax.errorbar(centers,cos_norm,yerr=cos_err_norm,fmt='o',color='black',markersize=0,lw=2)
    
        ax.set_ylabel("Fraction of Vertices",fontweight='bold',fontsize=20)
        ax.set_xlabel(name,fontweight='bold',fontsize=20)
        ax.set_xlim(xlo,xhi)
        ax.legend(loc='best')
        ax.grid()
        plt.savefig(os.path.join("dump",name + ".png"))
        plt.clf()
        plt.cla()
        plt.close()
    
    print "...gened"

    print "Writing..."
    write_nue_pdfs(lepton_spec_m,proton_spec_m,cosmic_spec_m,DIR_OUT=OUT_DIR)
    write_line_file(line_param,DIR_OUT=OUT_DIR)
    print "... wrote"

    sys.exit(0)
