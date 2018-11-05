from constants import *
import numpy as np

#
#
#
def pparam(row, pid, prefix):
    res = float(-1)
    
    par = int(row[pid])

    if pid < 0:
        return res

    par_id = "nueid_par%d" % par
    ps = prefix.split("_")
    param = par_id + "_" + "_".join(ps[2:])
    
    res = float(row[param])

    return res
    
def pparam_plane(row, pid, prefix, plane):
    res = float(-1)
    
    par = int(row[pid])

    if pid < 0:
        return res
    
    par_id = "nueid_par%d" % par
    ps = prefix.split("_")
    param = par_id + "_" + "_".join(ps[2:])

    param = param.replace("U",PLANE_TO_ALPHA[plane])
    param = param.replace("V",PLANE_TO_ALPHA[plane])
    param = param.replace("Y",PLANE_TO_ALPHA[plane])
    
    res = float(row[param])
    
    return res

def pparam_plane_ratio(row,pid,prefix1,prefix2,plane):

    val1 = pparam_plane(row,pid,prefix1,plane)
    val2 = pparam_plane(row,pid,prefix2,plane)
    
    ratio = float(0.0)
    ratio = val1 / val2

    return ratio

def pparam_v(row, pid, prefix, func):
    res = float(-1)
    
    par = int(row[pid])

    if par < 0:
        return res
    
    par_id = "nueid_par%d" % par
    ps = prefix.split("_")
    param = par_id + "_" + str("_".join(ps[2:]))

    val_v = row[param]
    idx_v = np.where(val_v>=0)[0]
    val_v = val_v[idx_v]

    res   = float(func(val_v))
    
    return res

def param_v(row, prefix, func):
    res = float(-1)
    
    val_v = row[prefix]
    idx_v = np.where(val_v>=0)[0]
    val_v = val_v[idx_v]
    res = float(func(val_v))
    
    return res

#
#
#
def param_array(row,prefix):
    res = []

    ps = prefix.split("_")
    par_id = "nueid_"+ps[1]
    prefix = "_".join(ps[2:-1])
    SS = par_id + "_planes_v"
    par_planes_v = row[SS]
    
    res_v = []
    for ix in xrange(3):
        if par_planes_v[ix] == 0:
            continue
        alpha = PLANE_TO_ALPHA[ix]
        SS = par_id + "_" + prefix + "_" + alpha
        val = row[val]
        res_v.append(val)
    
    return res_v

def max_param(row,prefix):
    res = float(-1)
    
    res_v = param_array(row,prefix)
    res_v = np.array(res_v)
    res   = np.max(res_v)
    
    return res

def min_param(row,prefix):
    res = float(-1)
    
    res_v = param_array(row,prefix)
    res_v = np.array(res_v)
    res   = np.min(res_v)
    
    return res

def mean_param(row,prefix):
    
    res = float(-1)
    
    res_v = param_array(row,prefix)
    res_v = np.array(res_v)
    res   = np.mean(res_v)
    
    return res
    
#
#
#
def pparam_array(row,pid,prefix):
    res = []

    par = int(row[pid])
    
    if par < 0:
        return res

    par_id = "nueid_par%d" % par
    ps = prefix.split("_")
    prefix = "_".join(ps[2:-1])
    SS = par_id + "_planes_v"
    par_planes_v = row[SS]
    
    res_v = []
    for ix in xrange(3):
        if par_planes_v[ix] == 0:
            continue
        alpha = PLANE_TO_ALPHA[ix]
        SS = par_id + "_" + prefix + "_" + alpha
        val = row[SS]
        res_v.append(val)
    
    return res_v

def max_pparam(row,pid,prefix):
    res = float(-1)
    
    res_v = pparam_array(row,pid,prefix)
    res_v = np.array(res_v)

    if res_v.size==0 :
        return res

    res = np.max(res_v)
    
    return res

def min_pparam(row,pid,prefix):
    res = float(-1)
    
    res_v = pparam_array(row,pid,prefix)
    res_v = np.array(res_v)

    if res_v.size==0:
        return res
    
    res = np.min(res_v)
    
    return res

def mean_pparam(row,pid,prefix):
    
    res = float(-1)
    
    res_v = pparam_array(row,pid,prefix)
    res_v = np.array(res_v)
    
    if res_v.size==0:
        return res

    res = np.mean(res_v)
    
    return res

#
#
#

def max_pparam_v(row,pid,prefix):
    res = float(-1)
    
    par = int(row[pid])

    if par < 0: 
        return res
    
    ps = prefix.split("_")
    par_id = "nueid_par%d" % par
    prefix = "_".join(ps[2:])
    
    U = (par_id + "_" + prefix)
    V = U.replace("U","V")
    Y = U.replace("U","Y")
    if row[U].size>0:
        res = np.maximum(res,np.max(row[U]))
    if row[V].size>0:
        res = np.maximum(res,np.max(row[V]))
    if row[Y].size>0:
        res = np.maximum(res,np.max(row[Y]))
    
    return res

def multi_pparam(row,pid,prefix):
    res_v = []

    par = int(row[pid])
    
    if par < 0:
        return res_v
    
    par_id = "nueid_par%d" % par
    ps = prefix.split("_")
    prefix = "_".join(ps[2:-1])
    SS = "%s_planes_v" % par_id
    par_planes_v = row[SS]
    
    res_v = []
    for ix in xrange(3):
        if par_planes_v[ix] == 0: continue
        alpha = PLANE_TO_ALPHA[ix]
        SS="%s_%s_%s"% (par_id,prefix,alpha)
        val = row[SS]
        res_v.append(val)

    res_v = np.array(res_v)
    idx_v = np.argsort(res_v)[::-1]
    
    return [res_v[idx_v[0]],res_v[idx_v[1]]]

def multi_pparam_v(row,pid,prefix):
    res_v = []

    par = int(row[pid])

    if par<0: 
        return res_v
    
    par_id = "nueid_par%d" % par
    ps = prefix.split("_")
    prefix = "_".join(ps[2:-2])
    SS = "%s_planes_v" % par_id
    par_planes_v = row[SS]
    
    res_v = []
    for ix in xrange(3):
        if par_planes_v[ix] == 0: continue
        alpha = PLANE_TO_ALPHA[ix]
        SS="%s_%s_%s_v"% (par_id,prefix,alpha)
        val = np.max(row[SS])
        res_v.append(val)

    res_v = np.array(res_v)
    idx_v = np.argsort(res_v)[::-1]
    
    return [res_v[idx_v[0]],res_v[idx_v[1]]]


def multi_pparam_v_plane(row,pid,prefix,plane):
    res = float(-1)

    par = int(row[pid])

    if par<0: 
        return res
    
    par_id = "nueid_par%d" % par
    ps = prefix.split("_")
    prefix = "_".join(ps[2:-2])
    SS = "%s_planes_v" % par_id
    par_planes_v = row[SS]
    
    if par_planes_v[plane] == 0:
        return res

    alpha = PLANE_TO_ALPHA[plane]
    SS="%s_%s_%s_v"% (par_id,prefix,alpha)
    val = np.max(row[SS])
    
    res = val
    
    return res


def multi_pparam_max(row,pid,prefix):
    ret = -1
    par = int(row[pid])
    if par<0: 
        return ret
    
    ret = multi_pparam_v(row,pid,prefix)[0]
    return ret


def multi_pparam_min(row,pid,prefix):
    ret = -1
    par = int(row[pid])
    if par<0: 
        return ret
    
    ret = multi_pparam_v(row,pid,prefix)[1]
    return ret


def multi_pparam_ratio(row,pid,prefix1,prefix2):
    res_v = []

    par = int(row[pid])
    
    if par<0: 
        return res_v
    
    par_id = "nueid_par%d" % par
    ps1 = prefix1.split("_")
    ps2 = prefix2.split("_")
    
    prefix1 = "_".join(ps1[2:-1])
    prefix2 = "_".join(ps2[2:-1])
    SS = "%s_planes_v" % par_id
    par_planes_v = row[SS]
    
    res_v = []
    for ix in xrange(3):
        if par_planes_v[ix] == 0: continue
        alpha = PLANE_TO_ALPHA[ix]
        SS1 = "%s_%s_%s"% (par_id,prefix1,alpha)
        SS2 = "%s_%s_%s"% (par_id,prefix2,alpha)
        ratio = 0
        r1 = float(row[SS1])
        r2 = float(row[SS2])
        if r2>0:
            ratio = r1 / r2;
        res_v.append(ratio)

    res_v = np.array(res_v)
    idx_v = np.argsort(res_v)[::-1]
    
    return [res_v[idx_v[0]],res_v[idx_v[1]]]

def multi_pparam_ratio_max(row,pid,prefix1,prefix2):
    res = -1
    par = int(row[pid])

    if par<0: 
        return res
        
    res = multi_pparam_ratio(row,pid,prefix1,prefix2)[0]
    return res

def multi_pparam_ratio_min(row,pid,prefix1,prefix2):
    res = -1
    par = int(row[pid])

    if par<0: 
        return res
        
    res = multi_pparam_ratio(row,pid,prefix1,prefix2)[1]
    return res

def bless_scedr(row):

    dx = row['parentSCEX'] - row['nueid_vertex_x']
    dy = row['parentSCEY'] - row['nueid_vertex_y']
    dz = row['parentSCEZ'] - row['nueid_vertex_z']

    dr = dx*dx+dy*dy+dz*dz
    dr = np.sqrt(dr)
    return dr
    
