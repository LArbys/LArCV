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
    //best for matplotlib, not pyqtgraph
    
    const auto& roi_meta = roi.BB( pl );
    
    PyObject* pdic = PyDict_New();

    PyObject* pset1 = PyTuple_New(2);
    PyObject* pset2 = PyTuple_New(2);
    PyObject* pset3 = PyTuple_New(2);
    PyObject* pset4 = PyTuple_New(2);
    
    PyTuple_SetItem(pset1,0,PyFloat_FromDouble(roi_meta.tl().x));
    PyTuple_SetItem(pset1,1,PyFloat_FromDouble(roi_meta.tl().y));

    PyTuple_SetItem(pset2,0,PyFloat_FromDouble(roi_meta.bl().x));
    PyTuple_SetItem(pset2,1,PyFloat_FromDouble(roi_meta.bl().y));
    
    PyTuple_SetItem(pset3,0,PyFloat_FromDouble(roi_meta.br().x));
    PyTuple_SetItem(pset3,1,PyFloat_FromDouble(roi_meta.br().y));
    
    PyTuple_SetItem(pset4,0,PyFloat_FromDouble(roi_meta.tr().x));
    PyTuple_SetItem(pset4,1,PyFloat_FromDouble(roi_meta.tr().y));

    PyDict_SetItem(pdic,PyString_FromString("tl"),pset1);
    PyDict_SetItem(pdic,PyString_FromString("bl"),pset2);
    PyDict_SetItem(pdic,PyString_FromString("br"),pset3);
    PyDict_SetItem(pdic,PyString_FromString("tr"),pset4);
    
    PyDict_SetItem(pdic,PyString_FromString("height"),PyFloat_FromDouble(roi_meta.height()));
    PyDict_SetItem(pdic,PyString_FromString("width") ,PyFloat_FromDouble(roi_meta.width()));
    // PyDict_SetItem(pdic,PyString_FromString("linewidth"),PyFloat_FromDouble(3.5));
    // PyDict_SetItem(pdic,PyString_FromString("edgecolor"),PyString_FromString("red"));
    // PyDict_SetItem(pdic,PyString_FromString("fill")     ,Py_False);

    return pdic;
  }

}

#endif
