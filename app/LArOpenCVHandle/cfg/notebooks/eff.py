import ROOT, sys, os
from ROOT import std

from larcv import larcv
from larlite import larlite as ll
from larlite import larutil as lu

import numpy as np

import scipy
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches

from ROOT import geo2d,cv
from ROOT.cv import Point_ as Vector
DTYPE='float'

import root_numpy as rn
import pandas as pd

geoh=lu.GeometryHelper.GetME()
geo=lu.Geometry.GetME()
larp=lu.LArProperties.GetME()
pygeo = geo2d.PyDraw()

INFILE=sys.argv[1]

base_index = ['run','subrun','event']
trees_m = { 
            'MCTree' : base_index,
            'EventTree' : base_index,
            'Vertex3DTree' : base_index + ['id'],
          }
           
df_m = {}
signal_df_m = {}
back_df_m = {}
signal_reco_df_m = {}
back_reco_df_m = {}

# drop duplicates
for tree_name_, index_ in trees_m.iteritems():
    df = pd.DataFrame(rn.root2array(INFILE,treename=tree_name_))
    df = df.drop_duplicates(subset=index_)
    df_m[tree_name_] = df.set_index(base_index)
 
signal_mc_idx = df_m['MCTree'].query('signal==1').index

for tree_name_, index_ in trees_m.iteritems():
    signal_df = df_m[tree_name_].ix[signal_mc_idx]

    #sometimes IX clobbers index name, reset the name
    signal_df.index.names = base_index
    signal_df_m[tree_name_] = signal_df.copy()


for tree_name_, index_ in trees_m.iteritems():
    signal_reco_df_m[tree_name_] = signal_df_m[tree_name_]#.ix[signal_reco_rse_idx].copy()

#set desired index
for tree_name_, index_ in trees_m.iteritems():
    signal_reco_df_m[tree_name_] = signal_reco_df_m[tree_name_].reset_index()
    signal_reco_df_m[tree_name_] = signal_reco_df_m[tree_name_].set_index(index_)


# In[13]:

s_vtx_tree = signal_reco_df_m['Vertex3DTree'].reset_index().set_index(base_index)
s_mc_tree  = signal_reco_df_m['MCTree'].reset_index().set_index(base_index)

def pick_good_vertex(sb_mc_tree,sb_vtx_tree):
    
    good_vtx_sb_v={}
    good_vtx_id_v={}
    
    for index, row in sb_mc_tree.iterrows():
        DEBUG=False
        # NOTE YOU HAVE TO BE CAREFUL HERE WITH THIS LINE BELOW
        vtx_entry = sb_vtx_tree.loc[index]
#         entry=signal_df_m['EventTree'].loc[index]['entry']
#         if entry==156:
#             DEBUG=True
        
        if type(vtx_entry) != pd.core.frame.DataFrame: 
            good_vtx_sb_v[index]  = False
            good_vtx_id_v[index] = -1
            continue

        vtx_x_vv= np.row_stack(vtx_entry.vtx2d_x_v.values)
        vtx_y_vv= np.row_stack(vtx_entry.vtx2d_y_v.values)

        dx = vtx_x_vv - row.vtx2d_t
        dy = vtx_y_vv - row.vtx2d_w

        dt = np.sqrt(dx*dx + dy*dy)            # compute the distance from true to all candidates
        min_idx=dt.mean(axis=1).argmin()       # get the smallest mean distance from candidates
        dt_b = (dt <= 7).sum(axis=1)           # vtx must be less than 7 pixels away
        n_close_vtx = len(np.where(dt_b>1)[0]) # event has >0 close verticies
        
        if DEBUG:
            print 
            print "Entry ",entry
            print "vtx_x_vv ",vtx_x_vv
            print "vtx_y_vv ",vtx_y_vv
            print "dx ",dx
            print "dy ",dy
            print "dt ",dt
            print "min_idx ",min_idx
            #break 
        good_vtx_sb_v[index]  = n_close_vtx>0
        good_vtx_id_v[index]  = vtx_entry.id.values[min_idx]

    good_vtx_sb_v  = pd.Series(good_vtx_sb_v)
    good_vtx_id_v  = pd.Series(good_vtx_id_v) 

    sb_vtx_df=pd.DataFrame([good_vtx_sb_v,good_vtx_id_v]).T
    sb_vtx_df.columns=['good','idx']

    return sb_vtx_df

s_vtx_df = pick_good_vertex(s_mc_tree,s_vtx_tree)
s_vtx_df.index.names = base_index
#b_vtx_df = pick_good_vertex(b_mc_tree,b_vtx_tree)
#b_vtx_df.index.names = base_index


# In[4]:

#get the signal dataframe
sig_vtx3d=signal_df_m["Vertex3DTree"].reset_index().set_index(base_index + ["id"])
#......signal........
#get the good vertex dataframe
good_vtx_tmp=s_vtx_df.copy()
good_vtx_tmp=good_vtx_tmp.reset_index()
good_vtx_tmp.columns=base_index+['good','id']
good_vtx_tmp=good_vtx_tmp.set_index(base_index + ['id'])
good_vtx_tmp=good_vtx_tmp.query("good==1.0")
sig_good_vtx_df=sig_vtx3d.ix[good_vtx_tmp.index]
del good_vtx_tmp


# In[5]:

print signal_df_m['MCTree'].index.size
print len(signal_df_m['Vertex3DTree'].reset_index().groupby(base_index))
print sig_good_vtx_df.index.size


# In[6]:

a=sig_good_vtx_df.reset_index().set_index(base_index)


# In[7]:

bad_mc=signal_df_m['MCTree'].drop(a.index,inplace=False)
bad_reco=signal_df_m['EventTree'].drop(a.index,inplace=False)


# In[8]:

#print bad_reco.entry.values.size
bad_reco.entry.values


# In[ ]:



