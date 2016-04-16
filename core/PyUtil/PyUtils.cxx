#ifndef __LARCV_PYUTILS_CXX__
#define __LARCV_PYUTILS_CXX__

#include "PyUtils.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

namespace larcv {

  void SetPyUtil()
  {
    static bool once=false;
    if(!once) { import_array(); once=true; }
  }
  
  PyObject* as_ndarray(const Image2D& img)
  {
    SetPyUtil();
    int* dim_data = new int[2];
    dim_data[0] = img.meta().cols();
    dim_data[1] = img.meta().rows();
    auto const& vec = img.as_vector();
    return PyArray_FromDimsAndData( 2, dim_data, NPY_FLOAT, (char*) &(vec[0]));
  }

}

#endif
