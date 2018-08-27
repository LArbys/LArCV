from common_functions import *

#
#
#
def dqdx_plane(row,pid,plane):
    res = [0,[],[],0,0]
    
    par_id = int(row[pid])
    if par_id < 1 : 
        return res
    
    par_id = "nueid_par%d" % par_id
    
    param  = par_id + "_" + "tdqdx_" + PLANE_TO_ALPHA[plane] + "_v"
    dqdx_v = row[param]
    if dqdx_v.size==0: 
        return res
    
    dx_v  = np.arange(0,dqdx_v.size,1)*0.3
    maxx  = np.max(dx_v)
    dx_v  = maxx - dx_v
    
    nzero = dqdx_v>0
    dqdx_v = dqdx_v[nzero]
    dx_v   = dx_v[nzero]
    
    res = [1,list(dqdx_v),list(dx_v),maxx]
    
    return res

def tdqdx_study(row,pid,dmax):
    ret_v = [float(-1),float(-1)]
    
    ppid = int(row[pid])

    if ppid < 0: return ret_v

    SS='nueid_par%d_tdqdx_Y_v' % ppid

    dqdx_v = row[SS]
    
    if dqdx_v.size == 0: return ret_v

    dx_v = np.arange(0,dqdx_v.size)*0.3

    nz_dqdx = np.where(dqdx_v!=0)[0]

    dqdx_v = dqdx_v[nz_dqdx]
    dx_v   = dx_v[nz_dqdx]
    
    if dqdx_v.size == 0: return ret_v

    dqdx_v = dqdx_v[dx_v<dmax]
    dx_v   = dx_v[dx_v<dmax]

    if dqdx_v.size == 0: return ret_v
    
    ret_v[0]  = np.mean(dqdx_v)
    ret_v[1]  = np.median(dqdx_v)
    
    return ret_v

    
#
#
#
def bless_scedr(row):
    dx = row['parentSCEX'] - row['nueid_vertex_x']
    dy = row['parentSCEY'] - row['nueid_vertex_y']
    dz = row['parentSCEZ'] - row['nueid_vertex_z']

    dr = dx*dx+dy*dy+dz*dz
    dr = np.sqrt(dr)

    return dr


#
# 
#
def define_ep(row):
    ret_v = [-1,-1,-1,-1]
    
    par1_v = np.array([row['nueid_par1_linefrac_U'],row['nueid_par1_linefrac_V'],row['nueid_par1_linefrac_Y']])
    par2_v = np.array([row['nueid_par2_linefrac_U'],row['nueid_par2_linefrac_V'],row['nueid_par2_linefrac_Y']])
    
    par1_len_v = np.array([row['nueid_par1_linelength_U'],row['nueid_par1_linelength_V'],row['nueid_par1_linelength_Y']])
    par2_len_v = np.array([row['nueid_par2_linelength_U'],row['nueid_par2_linelength_V'],row['nueid_par2_linelength_Y']])
    
    if np.isnan(par1_v[0]) == True:
	return ret_v 

    par1_idx_v = np.where(par1_v>0)[0]
    if par1_idx_v.size == 0: 
        return ret_v
    
    par2_idx_v = np.where(par2_v>0)[0]
    if par2_idx_v.size == 0: 
        return ret_v
    
    par1_v = par1_v[par1_idx_v]
    par2_v = par2_v[par2_idx_v]

    par1_max = np.max(par1_v)
    par2_max = np.max(par2_v)
    
    par1_min = np.min(par1_v)
    par2_min = np.min(par2_v)
    
    # check line fraction
    # the proton is the most straight

    if par1_max > par2_max:
        ret_v[0] = 1
        ret_v[1] = 2
    elif par2_max > par1_max:
        ret_v[0] = 2
        ret_v[1] = 1
    
    else: 
        # equal line frac: use the minimum
        if par1_min > par2_min:
            ret_v[0] = 1
            ret_v[1] = 2
        elif par2_min > par1_min:
            ret_v[0] = 2
            ret_v[1] = 1
            
        else:
            # both are the same: use length, longer=>electron
            par1_len_v = par1_len_v[par1_idx_v]
            par2_len_v = par2_len_v[par2_idx_v]
            
            par1_len_max = np.max(par1_len_v)
            par2_len_max = np.max(par2_len_v)
            
            if par2_len_max >= par1_len_max :
                ret_v[0] = 1
                ret_v[1] = 2
            else:
                ret_v[0] = 2
                ret_v[1] = 1
    
    return ret_v

#
#
#

def electron_func_PX_to_MEV(x):
    return (x - 1972.47) / 72.66

def proton_func_PX_to_MEV(x):
    return (x + 2229.55) / 72.66

def sum_charge(row,pid):
    ret = float(-1)

    par = int(row[pid])

    if par < 0:
        return par

    par_id = "nueid_par%d" % par

    pU1 = par_id + "_" + "expand_charge_U"
    pV1 = par_id + "_" + "expand_charge_V"
    pY1 = par_id + "_" + "expand_charge_Y"
    
    pU2 = par_id + "_" + "line_vtx_charge_U"
    pV2 = par_id + "_" + "line_vtx_charge_V"
    pY2 = par_id + "_" + "line_vtx_charge_Y"
    
    eU = float(row[pU1]+row[pU2])
    eV = float(row[pV1]+row[pV2])
    eY = float(row[pY1]+row[pY2])

    p_v = np.array([eU,eV,eY])
    
    eE = np.max(p_v)

    if pid=="eid":
        ret = electron_func_PX_to_MEV(eE)
    elif pid=="pid":
        ret = proton_func_PX_to_MEV(eE)
    else: 
        raise Exception
    
    return ret

def reco_energy(row):
    ret = float(-1)

    if int(row['eid']) < 0: 
        return ret

    if int(row['pid']) < 0: 
        return ret
    
    e_charge = sum_charge(row,"eid")
    p_charge = sum_charge(row,"pid")
    
    ret = e_charge + p_charge
    
    return ret

def reco_energy2(row,pid):
    ret = float(-1)

    par = int(row[pid])

    if par<0: 
        return ret

    ret = sum_charge(row,pid)
    
    return ret

def reco_electron_energy(row):
    ret = float(-1)

    if int(row['eid']) < 0: 
        return ret
    
    e_charge = sum_charge(row,"eid")
    
    ret = e_charge
    
    return ret

def reco_proton_energy(row):
    ret = float(-1)

    if int(row['pid']) < 0: 
        return ret
    
    p_charge = sum_charge(row,"pid")
    
    ret = p_charge
    
    return ret

# ii = [energy,cosangle]
def CCQE_p(ii):    
    ret = float(-1)

    Mn = 939.56
    Mp = 938.27
    Me = 0.511
    B = 40
    
    KEp = float(ii[1])
    Cp  = float(ii[0])
    
    top = 0.5*(2*(Mn-B)*(KEp+Mp)-((Mn-B)**2+Mp**2-Me**2))
    bot = (Mn-B)-(KEp+Mp)+np.sqrt((KEp+Mp)**2-Mp**2)*Cp

    ret = 0

    if bot != 0:
        ret = top / bot

    return ret

def reco_ccqe_energy(row):

    ret = float(-1)

    if int(row['pid'])<0: 
        return ret

    p_E_cal = reco_proton_energy(row)
    p_angle = pparam(row,"pid","nueid_par1_dz1")
    
    p_v = [p_angle, p_E_cal]
    
    ret = CCQE_p(p_v)
    
    return ret

#
#
#
def yes_brem(row,pid):
    ret = 0

    par = int(row[pid])

    if par < 0: 
        return ret
     
    par_id = "nueid_par%d" % par
    
    pU  = par_id + "_triangle_brem_U"
    pV  = par_id + "_triangle_brem_V"
    pY  = par_id + "_triangle_brem_Y"
    
    p_v = [row[pU],row[pV],row[pY]]
    p_v = np.array(p_v)

    if len(np.where(p_v>0)[0]) > 1:
        ret = 1;
        
    return ret

def no_brem(row,pid):
    ret = 0

    par = int(row[pid])

    if par < 0: 
        return ret
     
    par_id = "nueid_par%d" % par
    
    pU  = par_id + "_triangle_brem_U"
    pV  = par_id + "_triangle_brem_V"
    pY  = par_id + "_triangle_brem_Y"
    
    p_v = [row[pU],row[pV],row[pY]]
    p_v = np.array(p_v)

    if len(np.where(p_v==0)[0]) > 1:
        ret = 1;
    
    return ret

def angle(row,p):
    res = float(0)
    
    dx1 = float(row["nueid_par1_linedx_%s"%p])
    dx2 = float(row["nueid_par2_linedx_%s"%p])
    
    dy1 = float(row["nueid_par1_linedy_%s"%p])
    dy2 = float(row["nueid_par2_linedy_%s"%p])
    
    if dx1 < -1e3: return res
    if dx2 < -1e3: return res
    if dy1 < -1e3: return res
    if dy2 < -1e3: return res
    
    cos  = dx1*dx2 + dy1*dy2
    cos /= np.sqrt(dx1*dx1 + dy1*dy1)
    cos /= np.sqrt(dx2*dx2 + dy2*dy2)
    cos  = np.arccos(cos)
    cos *= 180/3.14

    res = float(cos)
    
    return res

def zfunc0_test(x):
    return -0.8*np.exp(x-0.25)+1.05

