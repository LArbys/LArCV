import numpy as np
import pandas as pd

rse   = ['run','subrun','event']
rsev  = ['run','subrun','event','vtxid']
rsec  = ['run','subrun','event','cvtxid']
rserv = ['run','subrun','event','roid','vtxid']

#
# Drop if column name ends in _y
#
def drop_y(df):
    to_drop = [x for x in df if x.endswith('_y')]
    df.drop(to_drop, axis=1, inplace=True)
        

#
# Define, given value, how to get prob. from PDF
#
def nearest_id(spectrum,value):
    return np.argmin(np.abs(spectrum - value))

def nearest_id_v(spectrum_v,value_v):
    return np.array([np.argmin(np.abs(spectrum[0] - value)) for spectrum, value in zip(spectrum_v,value_v)])
    
#
# Define the LL
#
def LL(row,sig_spectrum_m,bkg_spectrum_m):
    cols    = row[sig_spectrum_m.keys()]
    sig_res = nearest_id_v(sig_spectrum_m.values(),cols.values)
    bkg_res = nearest_id_v(bkg_spectrum_m.values(),cols.values)
    
    sig_res = np.array([spectrum[1][v] for spectrum,v in zip(sig_spectrum_m.values(),sig_res)])
    bkg_res = np.array([spectrum[1][v] for spectrum,v in zip(bkg_spectrum_m.values(),bkg_res)])
    
    LL = np.log( sig_res / (sig_res + bkg_res) )

    return LL.sum()
