import os, sys, gc
import numpy as np
import pandas as pd

rse   = ['run','subrun','event']
rsev  = ['run','subrun','event','vtxid']
rsec  = ['run','subrun','event','cvtxid']
rserv = ['run','subrun','event','roid','vtxid']

RSE   = list(rse)
RSEV  = list(rsev)
RSEVC = list(rsec)
RSERV = list(rserv)

#
# Drop if column name ends in _y
#
def drop_q(df):
    to_drop = [x for x in df if x.endswith('_q')]
    df.drop(to_drop, axis=1, inplace=True)

#
# Define, given value, how to get prob. from PDF
#
def nearest_id(spectrum,value):
    return np.argmin(np.abs(spectrum - value))

def nearest_id_v(spectrum_v,value_v):
    return np.array([np.argmin(np.abs(spectrum[0] - value)) for spectrum, value in zip(spectrum_v,value_v)])

#
# print whole data frame out
#
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')
    

#
# concat dataframe
#
def concat_pkls(flist_v,OUT_NAME="out"):
    df_v = []
    row_ctr = 0

    for ix,f in enumerate(flist_v):
        try:
            df = pd.read_pickle(f)
        except:
            print "\033[91m" + "ERROR @file=%s" % os.path.basename(f) + "\033[0m"
            continue
                    
        df_v.append(df)

        row_ctr += int(df_v[-1].index.size)
        print "\r(%02d/%02d)... read %d rows" % (ix,len(flist_v),row_ctr),
        sys.stdout.flush()
        
    if len(df_v)==0: 
        print "nothing to see..."
        return False
    
    df = pd.concat(df_v,ignore_index=True)
    print "...concat"
    SS = OUT_NAME + ".pkl"
    df.to_pickle(SS)
    dsk  = os.path.getsize(SS)
    dsk /= (1024.0 * 1024.0)
    print "...saved {:.02f} MB".format(dsk)
    del df
    gc.collect()
    print "...reaped"
    return True
