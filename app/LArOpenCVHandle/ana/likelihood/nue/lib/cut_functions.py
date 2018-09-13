from common_functions import *

#
#
#
def determine_flash(row):

    ret = 1e9

    eid = int(row['eid'])
    pid = int(row['pid'])

    if eid < 1 or pid < 1:
        return ret

    eid -= 1
    pid -= 1

    proton_shower_idx_vv = np.vstack(row['flash_proton_shower_pair_vv'])

    par_bool=((pid==proton_shower_idx_vv[:,0])&(eid==proton_shower_idx_vv[:,1]))
    proton_shower_idx = np.where(par_bool)

    if len(proton_shower_idx)==0:
        return ret
    
    proton_shower_idx = int(proton_shower_idx[0])
    
    chi2_shape_1e1p_v = row['flash_proton_shower_chi2_shape_1e1p_v']

    if proton_shower_idx >= chi2_shape_1e1p_v.size:
        return ret

    chi2_shape = chi2_shape_1e1p_v[proton_shower_idx]

    ret = float(chi2_shape)

    return ret
#
#
#
def read_config(cut_cfg):
    ret = ""

    data = None
    with open(cut_cfg,'r') as f:
        data = f.read()

    if data is None:
        return ret
    
    data_v = data.split("\n")
    data_v = [d for d in data_v if d!='']
    
    ret = "&".join(data_v)
        
    return ret
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
    
    nzero  = dqdx_v>0
    dqdx_v = dqdx_v[nzero]
    dx_v   = dx_v[nzero]
    
    res = [1,list(dqdx_v),list(dx_v),maxx]
    
    return res


def dqdx_plane_median(row,pid,plane):
    ret = -1
    
    par_id = int(row[pid])
    if par_id < 1:
        return ret
    
    dqdx_v = dqdx_plane(row,pid,plane)
    
    if dqdx_v[0] <= 0:
        return ret
    
    dqdx_med = float(np.median(dqdx_v[1]))
    
    if dqdx_med != dqdx_med:
        return ret
    
    if np.isinf(dqdx_med) == True:
        return ret
    
    dqdx_med /= float(72.66)

    ret = dqdx_med
    
    return ret

def dqdx_plane_best(row,pid,plane1,plane2):
    ret = -1

    par_id = int(row[pid])
    if par_id < 1:
        return ret
    
    dqdx1 = dqdx_plane_median(row,pid,plane1)
    dqdx2 = dqdx_plane_median(row,pid,plane2)
    
    if dqdx1 > 0:
        ret = dqdx1
    else:
        ret = dqdx2
    
    return ret


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


def CCQE_predicted_lepton(CCQE,Cose):

    ret = np.nan

    Mn = 939.56
    Mp = 938.27
    Me = 0.511
    B  = 40

    # Solve F = 0.5 * (2*A*x - H) / (A - x - D*sqrt(x^2-G^2)) for x
    # https://www.nevis.columbia.edu/~vgenty/public/fop.png

    A = Mn-B
    C = Me*Me-Mp*Mp
    H = A*A+C
    D = Cose
    G = Me*Me
    F = CCQE
    
    top1 =  4 * A**2 * D**2 * F**4 - 4 * A**2 * D**2 * F**2 * G**2
    top2 = -8 * A * D**2 * F**3 * G**2 + 4 * A * D**2 * F**3 * H
    top3 =  4 * D**4 * F**4 * G**2 - 4 * D**2 * F**4 * G**2
    top4 =  D**2 * F**2 * H**2
    top5 =  2 * A**2 * F + 2 * A * F**2 + A * H + F * H
    
    bot1 =  A**2 + 2 * A * F - D**2 * F**2 + F**2
    
    top = np.sqrt(top1+top2+top3+top4) + top5
    bot = 2*bot1
    
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
    bot1 = np.sqrt(dx1*dx1 + dy1*dy1)
    bot2 = np.sqrt(dx2*dx2 + dy2*dy2)

    if bot1==0 or bot2==0: 
        return res

    cos /= bot1
    cos /= bot2
    
    if cos < -1 or cos > 1:
        return res

    cos  = np.arccos(cos)
    cos *= 180/3.14

    res = float(cos)
    
    return res

def angle3d(row,pid):
    res = float(-1)

    dx1 = float(row["nueid_par1_dx1"])
    dx2 = float(row["nueid_par2_dx1"])
    
    dy1 = float(row["nueid_par1_dy1"])
    dy2 = float(row["nueid_par2_dy1"])
    
    dz1 = float(row["nueid_par1_dz1"])
    dz2 = float(row["nueid_par2_dz1"])

    cos  = dx1*dx2 + dy1*dy2 + dz1*dz2
    bot1 = np.sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
    bot2 = np.sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)
    
    if bot1==0 or bot2==0: 
        return res
    
    cos /= bot1
    cos /= bot2
    
    if cos<-1 or cos>1:
        return res
    
    cos  = np.arccos(cos)
    cos *= 180/3.14

    res = float(cos)

    return res

def zfunc0_test(x):
    return -0.8*np.exp(x-0.25)+1.05


def CCQE_EVIS(row,pid):

    ret = float(0)
    
    if int(row['pid']) < 0: 
        return ret

    if int(row['eid']) < 0: 
        return ret
    
    p_E_cal = row['reco_proton_energy']
    e_E_cal = row['reco_electron_energy']

    e_p_E_cal = p_E_cal + e_E_cal 
    
    p_angle = pparam(row,"pid","nueid_par1_dz1")
    
    p_v = [p_angle, p_E_cal]
    
    CCQE_e = CCQE_p(p_v)

    E_vis  = float(e_p_E_cal+0.511+40)
    
    dE = (CCQE_e - E_vis) / E_vis

    ret = dE

    return ret
