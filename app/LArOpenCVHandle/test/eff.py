import sys,os

INDIR=sys.argv[1]
DIR=os.path.basename(INDIR)

SS="cp %s %s"%("lee1e1p_ana.cfg",INDIR)
print SS
os.system(SS)
os.chdir(INDIR)

# BASEDIR="/Users/vgenty/Desktop/intrinsic_nue/out_pyroi"
# SS="run_processor test.cfg c_out*.root %s"%os.path.join(BASEDIR,"larcv_fcn_out*.root")

SS="run_processor lee1e1p_ana.cfg c_comb*.root"
print SS
os.system(SS)


import pandas as pd
import root_numpy as rn
import numpy as np

f_ = open("eff.txt",'w+')

index_  = ['run','subrun','event']
FILE_   = "ana1.root"
ana_df  = pd.DataFrame(rn.root2array(FILE_,treename='tree'))
eana_df = pd.DataFrame(rn.root2array(FILE_,treename='event_tree'))
eana_df = eana_df.drop_duplicates(index_)


cutstr0 = ' q0/npx0 > 20 and q1/npx1 > 10'
cutstr1 = ' and q0/area0>20 and q1/area1>10'
cutstr2 = ' and npx0/area0 > 1.0 and 0.5 < npx1/area1 and npx1/area1 < 1.5'
cutstr3 = ' and len1 > 50'
cutstr4 = ' and len0 < 300'
cutstr=cutstr0+cutstr1+cutstr2+cutstr3+cutstr4

good_df=ana_df.query("scedr<5")
bad_df =ana_df.query("scedr>=5")

unique_evts=pd.unique(ana_df.reset_index().set_index(index_).index)
unique_good=pd.unique(good_df.reset_index().set_index(index_).index)
unique_bad =pd.unique(bad_df.reset_index().set_index(index_).index)
f_.write("Ana %d\n"%unique_evts.size)
f_.write("Good %d\n"%unique_good.size)
f_.write("Bad %d\n"%unique_bad.size)
f_.write("\n")

#eana_good_df = eana_df.query("(good_croi0 + good_croi1 + good_croi2)>=2")
#eana_bad_df  = eana_df.query("(good_croi0 + good_croi1 + good_croi2)<2")
eana_good_df = eana_df.query("good_croi_ctr>0")
eana_bad_df  = eana_df.query("good_croi_ctr==0")
unique_signal_evts = pd.unique(eana_good_df.reset_index().set_index(index_).index)
unique_bad_evts    = pd.unique(eana_bad_df.reset_index().set_index(index_).index)
f_.write("Total number input   %d\n"%pd.unique(eana_df.reset_index().set_index(index_).index).size)
f_.write("Number of GOOD CROIs %d\n"%unique_signal_evts.size)
f_.write("Number of BAD CROIs  %d\n"%unique_bad_evts.size)
f_.write("\n")

bad_df_temp = bad_df.set_index(index_).drop(unique_good)
f_.write("n events with no good reco vertex %d\n"%len(bad_df_temp.reset_index().groupby(index_)))
a=bad_df_temp.reset_index().set_index(index_).ix[unique_bad_evts].dropna()
a.index.names=index_
f_.write("n with no good CROI ................ %d\n"%len(a.reset_index().groupby(index_)))
f_.write("\n")
f_.write("\n")
rse_index = ['run','subrun','event']

f_.write("Cut 0) %s"%cutstr0)
f_.write("\n")
f_.write("Cut 1) %s"%cutstr1)
f_.write("\n")
f_.write("Cut 2) %s"%cutstr2)
f_.write("\n")
f_.write("Cut 3) %s"%cutstr3)
f_.write("\n")
f_.write("Cut 4) %s"%cutstr4)
f_.write("\n")

f_.write("<==All===>\n")
den_=ana_df.index.size
num_=len(ana_df.groupby(rse_index))
f_.write(str(("Vertex:",den_," events:",num_,"=",float(num_)/float(den_))))
f_.write("\n")         
den_=ana_df.query(cutstr0).index.size
num_=len(ana_df.query(cutstr0).groupby(rse_index))
f_.write(str(("     0)",den_," events:",num_,"=",float(num_)/float(den_))))
f_.write("\n")         
den_=ana_df.query(cutstr0+cutstr1).index.size
num_=len(ana_df.query(cutstr0+cutstr1).groupby(rse_index))
f_.write(str(("     1)",den_," events:",num_,"=",float(num_)/float(den_))))
f_.write("\n")             
den_=ana_df.query(cutstr0+cutstr1+cutstr2).index.size
num_=len(ana_df.query(cutstr0+cutstr1+cutstr2).groupby(rse_index))
f_.write(str(("     2)",den_," events:",num_,"=",float(num_)/float(den_))))
f_.write("\n")
den_=ana_df.query(cutstr0+cutstr1+cutstr2+cutstr3).index.size
num_=len(ana_df.query(cutstr0+cutstr1+cutstr2+cutstr3).groupby(rse_index))
f_.write(str(("     3)",den_," events:",num_,"=",float(num_)/float(den_))))
f_.write("\n")
den_=ana_df.query(cutstr).index.size
num_=len(ana_df.query(cutstr).groupby(rse_index))
if den_>0:
    r = float(num_)/float(den_)
else:
    r=0
f_.write(str(("     4)",den_," events:",num_,"=",r)))
f_.write("\n")

f_.write("<==Good==>\n")
den_=good_df.index.size
num_=len(good_df.groupby(rse_index))
f_.write(str(("Vertex:",den_," events:",num_,"=",float(num_)/float(den_))))
f_.write("\n")
den_=good_df.query(cutstr0).index.size
num_=len(good_df.query(cutstr0).groupby(rse_index))
f_.write(str(("     0)",den_," events:",num_,"=",float(num_)/float(den_))))
f_.write("\n")
den_=good_df.query(cutstr0+cutstr1).index.size
num_=len(good_df.query(cutstr0+cutstr1).groupby(rse_index))
f_.write(str(("     1)",den_," events:",num_,"=",float(num_)/float(den_))))
f_.write("\n")
den_=good_df.query(cutstr0+cutstr1+cutstr2).index.size
num_=len(good_df.query(cutstr0+cutstr1+cutstr2).groupby(rse_index))
f_.write(str(("     2)",den_," events:",num_,"=",float(num_)/float(den_))))
f_.write("\n")
den_=good_df.query(cutstr0+cutstr1+cutstr2+cutstr3).index.size
num_=len(good_df.query(cutstr0+cutstr1+cutstr2+cutstr3).groupby(rse_index))
f_.write(str(("     3)",den_," events:",num_,"=",float(num_)/float(den_))))
f_.write("\n")
den_=good_df.query(cutstr).index.size
num_=len(good_df.query(cutstr).groupby(rse_index))
if den_>0:
    r = float(num_)/float(den_)
else:
    r=0
f_.write(str(("     4)",den_," events:",num_,"=",r)))
f_.write("\n")

f_.write("<==Bad===>\n")
den_=bad_df.index.size
num_=len(bad_df.groupby(rse_index))
f_.write(str(("Vertex:",den_," events:",num_,"=",float(num_)/float(den_))))
f_.write("\n")         
den_=bad_df.query(cutstr0).index.size
num_=len(bad_df.query(cutstr0).groupby(rse_index))
f_.write(str(("     0)",den_," events:",num_,"=",float(num_)/float(den_))))
f_.write("\n")
den_=bad_df.query(cutstr0+cutstr1).index.size
num_=len(bad_df.query(cutstr0+cutstr1).groupby(rse_index))
f_.write(str(("     1)",den_," events:",num_,"=",float(num_)/float(den_))))
f_.write("\n")
den_=bad_df.query(cutstr0+cutstr1+cutstr2).index.size
num_=len(bad_df.query(cutstr0+cutstr1+cutstr2).groupby(rse_index))
f_.write(str(("     2)",den_," events:",num_,"=",float(num_)/float(den_))))
f_.write("\n")
den_=bad_df.query(cutstr0+cutstr1+cutstr2+cutstr3).index.size
num_=len(bad_df.query(cutstr0+cutstr1+cutstr2+cutstr3).groupby(rse_index))
f_.write(str(("     3)",den_," events:",num_,"=",float(num_)/float(den_))))
f_.write("\n")
den_=bad_df.query(cutstr).index.size
num_=len(bad_df.query(cutstr).groupby(rse_index))
if den_>0:
    r = float(num_)/float(den_)
else:
    r=0
f_.write(str(("     4)",den_," events:",num_,"=",r)))
f_.write("\n")
f_.close()
