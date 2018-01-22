from array import array
from pd2root import *
import numpy as np

kINVALID_INT    = ROOT.std.numeric_limits("int")().lowest()
kINVALID_FLOAT  = ROOT.std.numeric_limits("float")().lowest()
kINVALID_DOUBLE = ROOT.std.numeric_limits("double")().lowest()

VDOUBLE = "ROOT.std.vector(\"double\")()"
VFLOAT  = "ROOT.std.vector(\"float\")()"
VINT    = "ROOT.std.vector(\"int\")()"

VVDOUBLE = "ROOT.std.vector(\"std::vector<double>\")()"
VVFLOAT  = "ROOT.std.vector(\"std::vector<float>\")()"
VVINT    = "ROOT.std.vector(\"std::vector<int>\")()"

class ROOTData:
    def __init__(self,df):

        self.vmem_v  = []
        self.vvmem_v = []

        self.smem_v  = []
        self.stype_v = []

        self.irange = 50

        for column in df:
            rtype = dtype2rtype(df[column].dtype)

            # set vector variables
            if rtype == "x": 
                assert type(df.iloc[0][column]) in [np.ndarray,list,float]
                
                shape = -1
                for ir in xrange(df.shape[0]):
                    if ir>self.irange: break
                    try: 
                        shape = np.maximum(shape,np.vstack(df.iloc[ir][column]).shape[1])
                    except (ValueError,TypeError):
                        continue


                SS = "self.%s = %s"                
                if shape==1:
                    SS = SS % (column, VFLOAT)
                    exec(SS)
                    self.vmem_v.append(column)
                elif shape>1:
                    SS = SS % (column, VVFLOAT)
                    exec(SS)
                    self.vvmem_v.append(column)
                else:
                    raise Exception("Unhandled shape")
                    
            # set scalar variables
            else:
                default = rtype2default(rtype)
                SS = "self.%s = array( \"%s\" , [ %s ])"
                SS = SS % (column, rtype, default)
                exec(SS)
                
                self.smem_v.append(column)
                
                typecode = None
                SS = "typecode = self.%s.typecode"
                SS = SS % column
                exec(SS)
                self.stype_v.append(typecode)

    def fill(self,row,tree):
        for smem,stype in zip(self.smem_v,self.stype_v):
            value = row[smem]
            if np.isnan(value): continue
            SS = "self.%s[0] = %s(%s)"
            SS = SS % (smem, typecode2py(stype), value)
            exec(SS)
        
        for vmem in self.vmem_v:
            value = row[vmem]
            type_ = type(value)
            if type_ == float: value = []
            value = np.array(value,dtype=np.float32)
            shape = int(value.shape[0])
            if shape == 0: continue
            SS = "self.%s.resize(%d)"
            SS = SS % (vmem,shape)
            exec(SS)

            for i in xrange(shape):
                SS = "self.%s[%d] = float(value[%d])"
                SS = SS % (vmem,i,i)
                exec(SS)

        for vvmem in self.vvmem_v:
            value = row[vvmem]
            type_ = type(value)
            if type_ == float: value = []
            value = np.array(value)
            shape = int(value.shape[0])
            if shape == 0: continue
            SS = "self.%s.resize(%d)"
            SS = SS % (vvmem,shape)
            exec(SS)

            for ix,subarr in enumerate(value):
                shape = int(subarr.shape[0])
                SS = "self.%s[%d].resize(%d)"
                SS = SS % (vvmem,ix,shape)
                exec(SS)
                for i in xrange(shape):
                    SS = "self.%s[%d][%d] = float(subarr[%d])"
                    SS = SS % (vvmem,ix,i,i)
                    exec(SS)

        tree.Fill()

    def reset(self):
        for smem,stype in zip(self.smem_v,self.stype_v):
            SS = "self.%s[0] = %s"
            SS = SS % (smem, rtype2default(stype))
            exec(SS)

        for vmem in self.vmem_v:
            SS = "self.%s.clear()"
            SS = SS % (vmem)
            exec(SS)

        for vvmem in self.vvmem_v:
            SS = "self.%s.clear()"
            SS = SS % (vvmem)
            exec(SS)
        
    def init_tree(self,tree):
        for smem,stype in zip(self.smem_v,self.stype_v):
            SS = "tree.Branch(\"%s\", self.%s, \"%s/%s\")"
            SS = SS % (smem, smem, smem, typecode2root(stype))
            exec(SS)
        
        for vmem in self.vmem_v:
            SS = "tree.Branch(\"%s\", self.%s)"
            SS = SS % (vmem,vmem)
            exec(SS)

        for vvmem in self.vvmem_v:
            SS = "tree.Branch(\"%s\", self.%s)"
            SS = SS % (vvmem,vvmem)
            exec(SS)




