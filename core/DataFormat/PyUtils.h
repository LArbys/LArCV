#ifndef __LARCV_PYUTILS_H__
#define __LARCV_PYUTILS_H__

struct _object;
typedef _object PyObject;

#ifndef __CLING__
#ifndef __CINT__
#include <Python.h>
#endif
#endif

#include "Image2D.h"
#include "ROI.h"

namespace larcv {

  void SetPyUtil();

  PyObject* as_ndarray(const Image2D& img);
  PyObject* as_bbox(const ROI& roi,PlaneID_t pl);
  
}

#endif
