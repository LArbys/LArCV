#include "PyUtils_sparsetensor.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifdef USE_PYTHON3
#include "numpy/arrayobject.h"
#include <cassert>
#else
#include <numpy/ndarrayobject.h>
#endif

#include "larcv/core/Base/larcv_logger.h"

namespace larcv {

  bool PyUtils_sparsetensor::_import_numpy = false;
  
  PyObject *as_ndarray(const SparseTensor2D& data, bool clear_mem) {
    
    if ( !PyUtils_sparsetensor::_import_numpy ) {
      // needed for numpy api
      import_array1(0);
      PyUtils_sparsetensor::_import_numpy = true;
    }
    
    npy_intp dim_data[2];
    dim_data[0] = data.meta().cols();
    dim_data[1] = data.meta().rows();
    
    static std::vector<float> local_data;
    local_data.resize(data.meta().size());
    for(auto &v : local_data) v = 0.;
    
    for(auto const& vox : data.as_vector()) local_data[vox.id()]=vox.value();
    
    auto res = PyArray_Transpose(((PyArrayObject*)(PyArray_SimpleNewFromData(2, dim_data, NPY_FLOAT, (char *)&(local_data[0])))),NULL);
    //return PyArray_FromDimsAndData(2, dim_data, NPY_FLOAT, (char *)&(vec[0]));
    
    if(clear_mem) local_data.clear();
    return res;
  }
  
  /**
   * @brief convert SparseTensor3D into a numpy array
   *
   */
  int fill_3d_voxels(const SparseTensor3D& data, PyObject* pyarray, PyObject* select) {

    if ( !PyUtils_sparsetensor::_import_numpy ) { 
      import_array1(0);
      PyUtils_sparsetensor::_import_numpy = true;
    }

    int **carray;
    const int dtype = NPY_INT;
    PyArray_Descr *descr = PyArray_DescrFromType(dtype);
    npy_intp dims[2];
    if (PyArray_AsCArray(&pyarray, (void **)&carray, dims, 2, descr) < 0) {
      logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
				 "ERROR: cannot convert pyarray to 2D C-array");
      throw larbys();
    }
    
    size_t npts = data.size();
    int* select_ptr = nullptr;
    if(select) {
      auto select_pyptr = (PyArrayObject *)(select);
      // Check dimension size is 1:
      if (PyArray_NDIM(select_pyptr) != 1){
	logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
				   "ERROR: select array must be 1D!");
	throw larbys();
      }
      if((int)npts < PyArray_SIZE(select_pyptr)) {
	logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
				   "ERROR: select array size exceeds max data length!");
	throw larbys();
      }
      npts = PyArray_SIZE(select_pyptr);
      npy_intp loc[1];
      loc[0] = 0;
      select_ptr = (int*)(PyArray_GetPtr(select_pyptr,loc));
    }
    
    if(npts > data.size() || dims[1] != 3 || dims[0] != npts) {
      logger::get("PyUtil").send(larcv::msg::kCRITICAL,__FUNCTION__,__LINE__,
				 "ERROR: dimension mismatch");
      throw larbys();
    }
    
    auto const& vs = data.as_vector();
    size_t ix,iy,iz;
    for(size_t i=0; i<npts; ++i) {
      size_t index = i;
      if(select_ptr)
	index = select_ptr[i];
      
      auto const& vox = vs[index];
      data.meta().id_to_xyz_index(vox.id(),ix,iy,iz);
      carray[i][0] = ix;
      carray[i][1] = iy;
      carray[i][2] = iz;
    }
    
    PyArray_Free(pyarray,  (void *)carray);
    
    return 0;
  }

  int fill_3d_pcloud(const SparseTensor3D& data, PyObject* pyarray, PyObject* select) {
    return fill_3d_pcloud((larcv::VoxelSet)data, data.meta(), pyarray, select );
  }


  int fill_3d_pcloud(const VoxelSet& data, const Voxel3DMeta& meta, PyObject* pyarray, PyObject* select) {
    
    if ( !PyUtils_sparsetensor::_import_numpy ) { 
      import_array1(0);
      PyUtils_sparsetensor::_import_numpy = true;
    }
    
    float **carray;
    const int dtype = NPY_FLOAT;
    PyArray_Descr *descr = PyArray_DescrFromType(dtype);
    npy_intp dims[2];
    if (PyArray_AsCArray(&pyarray, (void **)&carray, dims, 2, descr) < 0) {
      logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
				 "ERROR: cannot convert pyarray to 2D C-array");
      throw larbys();
    }
    
    size_t npts = data.size();
    int* select_ptr = nullptr;
    if(select) {
      auto select_pyptr = (PyArrayObject *)(select);
      // Check dimension size is 1:
      if (PyArray_NDIM(select_pyptr) != 1){
	logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
				   "ERROR: select array must be 1D!");
	throw larbys();
      }
      if((int)npts < PyArray_SIZE(select_pyptr)) {
	logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
				   "ERROR: select array size exceeds max data length!");
	throw larbys();
      }
      npts = PyArray_SIZE(select_pyptr);
      npy_intp loc[1];
      loc[0] = 0;
      select_ptr = (int*)(PyArray_GetPtr(select_pyptr,loc));
    }
    
    if(npts > data.size() || (dims[1] != 1 && dims[1] != 3 && dims[1] != 4) || dims[0] != data.size()) {
      logger::get("PyUtil").send(larcv::msg::kCRITICAL,__FUNCTION__,__LINE__,
				 "ERROR: dimension mismatch");
      throw larbys();
    }
    
    auto const& vs = data.as_vector();
    
    for(size_t i=0; i<npts; ++i) {
      size_t index = i;
      if(select_ptr)
	index = select_ptr[i];
      
      auto const& vox = vs[index];
      auto pt = meta.position(vox.id());
      //if(dims[1] == 1 && !(isnan(vox.value())))
      if(dims[1] == 1)
	carray[i][0] = vox.value();
      else if(dims[1] == 3) {
	carray[i][0] = pt.x;
	carray[i][1] = pt.y;
	carray[i][2] = pt.z;
      }
      if(dims[1] == 4) {
	carray[i][0] = pt.x;
	carray[i][1] = pt.y;
	carray[i][2] = pt.z;
	carray[i][3] = vox.value();
      }
    }
    PyArray_Free(pyarray,  (void *)carray);
    return 0;
  }
 
  

}
