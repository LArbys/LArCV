import ROOT
import numpy as np


def dtype2rtype(dtype):
    if   dtype == np.float32:
        return "f"
    elif dtype == np.float64:
        return "d"
    elif dtype == np.uint16:
        return "i"
    elif dtype == np.uint32:
        return "i"
    elif dtype == np.uint64:
        return "i"
    elif dtype == np.int16:
        return "i"
    elif dtype == np.int32:
        return "i"
    elif dtype == np.int64:
        return "i"
    elif dtype == np.bool_:
        return "i"
    elif dtype == float:
        return "f"
    elif dtype == int:
        return "i"
    elif dtype == bool:
        return "i"
    else:
        return "x"
    
def rtype2default(rtype):
    if   rtype == "f":
        return "kINVALID_FLOAT"
    elif rtype == "d":
        return "kINVALID_DOUBLE"
    elif rtype == "i":
        return "kINVALID_INT"
    else:
        return "aho"
    
def typecode2root(typecode):
    if   typecode == 'd':
        return "D"
    elif typecode == 'f':
        return "F"
    elif typecode == 'i':
        return "I"
    else:
        return "aho"

def typecode2py(typecode):
    if   typecode == 'd':
        return "float"
    elif typecode == 'f':
        return "float"
    elif typecode == 'i':
        return "int"
    else:
        return "aho"


def typecode2np(typecode):
    if   typecode == 'd':
        return "np.float64"
    elif typecode == 'f':
        return "np.float32"
    elif typecode == 'i':
        return "np.int32"
    else:
        return "aho"
