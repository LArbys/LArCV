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

  PyObject* as_bbox(const ROI& roi,PlaneID_t pl)
  {
    const auto& meta = roi.BB( pl );
    
    PyObject* pdic = PyDict_New();
    PyObject* pset = PyTuple_New(2);
      
    PyTuple_SetItem(pset,0,PyFloat_FromDouble(meta.tl().x));
    PyTuple_SetItem(pset,1,PyFloat_FromDouble(meta.tl().y));

    PyDict_SetItem(pdic,PyString_FromString("xy")       ,pset);
    PyDict_SetItem(pdic,PyString_FromString("height")   ,PyFloat_FromDouble(meta.height()));
    PyDict_SetItem(pdic,PyString_FromString("width")    ,PyFloat_FromDouble(meta.width()));
    PyDict_SetItem(pdic,PyString_FromString("linewidth"),PyFloat_FromDouble(3.5));
    PyDict_SetItem(pdic,PyString_FromString("edgecolor"),PyString_FromString("red"));
    PyDict_SetItem(pdic,PyString_FromString("fill")     ,Py_False);

    return pdic;
  }
}

#endif
