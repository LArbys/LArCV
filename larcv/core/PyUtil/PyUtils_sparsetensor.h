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
  int fill_3d_voxels(const SparseTensor3D& data, PyObject* pyarray, PyObject* select=nullptr);

  /// larcv::VoxelSet to 3D numpy point cloud array (NUM_POINTS,1/3/4) converter
  int fill_3d_pcloud(const SparseTensor3D &data, PyObject* pyarray, PyObject* select=nullptr);
  
  /// larcv::VOxelSet to 3D numpy point cloud array (NUM_POINTS,1/3/4) converter
  int fill_3d_pcloud(const VoxelSet &data, const Voxel3DMeta& meta, PyObject* pyarray, PyObject* select=nullptr);
  

  class PyUtils_sparsetensor {

  public:
    static bool _import_numpy;
  };
  
}

#endif
