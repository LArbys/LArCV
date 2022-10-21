#ifndef __LARCV_CORE_PYUTILS_SPARSETENSOR_H__
#define __LARCV_CORE_PYUTILS_SPARSETENSOR_H__

#ifndef __CLING__
#ifndef __CINT__
#include <Python.h>
#include "bytesobject.h"
#endif
#endif

#include "larcv/core/DataFormat/SparseTensor2D.h"
#include "larcv/core/DataFormat/SparseTensor3D.h"

namespace larcv {

  // sparsetensor2d into dense numpy array (np.float)
  PyObject *as_ndarray(const SparseTensor2D& data, bool clear_mem=false);  

  // sparsetensor3d into a dense numpy array
  void fill_3d_voxels(const SparseTensor3D& data, PyObject* pyarray, PyObject* select);
  
}

#endif
