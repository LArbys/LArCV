#ifndef __LARCV_PYUTILS_CXX__
#define __LARCV_PYUTILS_CXX__

#include "Base/larcv_logger.h"
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

  larcv::Image2D as_image2d(PyObject* pyarray,ImageMeta meta)
  {
    SetPyUtil();
    double **carray;
    //Create C arrays from numpy objects:
    const int dtype = NPY_FLOAT;
    PyArray_Descr *descr = PyArray_DescrFromType(dtype);
    npy_intp dims[3];
    if (PyArray_AsCArray(&pyarray, (void **)&carray, dims, 2, descr) < 0) {
      logger::get("PyUtil").send(larcv::msg::kCRITICAL,__FUNCTION__,__LINE__,"ERROR: cannot convert to 2D C-array");
      throw larbys();
    }

    std::vector<float> res_data(dims[0]*dims[1],0.);
    for(int i=0; i<dims[0]; ++i) {
      for(int j=0; j<dims[1]; ++j) {
	res_data[i * dims[1] + j] = (float)(carray[i][j]);
      }
    }
    PyArray_Free(pyarray,(void *)carray);

    Image2D res(std::move(meta),std::move(res_data));
    return res;
  }

  larcv::Image2D as_image2d(PyObject* pyarray)
  {
    SetPyUtil();
    double **carray;
    //Create C arrays from numpy objects:
    const int dtype = NPY_FLOAT;
    PyArray_Descr *descr = PyArray_DescrFromType(dtype);
    npy_intp dims[3];
    if (PyArray_AsCArray(&pyarray, (void **)&carray, dims, 2, descr) < 0) {
      logger::get("PyUtil").send(larcv::msg::kCRITICAL,__FUNCTION__,__LINE__,"ERROR: cannot convert to 2D C-array");
      throw larbys();
    }

    std::vector<float> res_data(dims[0]*dims[1],0.);
    for(int i=0; i<dims[0]; ++i) {
      for(int j=0; j<dims[1]; ++j) {
	res_data[i * dims[1] + j] = (float)(carray[i][j]);
      }
    }
    PyArray_Free(pyarray,(void *)carray);

    ImageMeta meta((double)(dims[0]), (double)(dims[1]),
		   (size_t)(dims[1]), (size_t)(dims[0]),
		   0., 0.,
		   larcv::kINVALID_PLANE);

    Image2D res(std::move(meta),std::move(res_data));
    return res;
  }
}

#endif
