#ifndef __LARCV_PYUTILS_CXX__
#define __LARCV_PYUTILS_CXX__

#include <iostream>
#include "PyUtils.h"
#include "larcv/core/Base/larcv_logger.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
//#include "numpy/arrayobject.h"

namespace larcv {

void SetPyUtil() {
  static bool once = false;
  if (!once) {
    import_array();
    once = true;
  }
}

PyObject *as_ndarray(const std::vector<float> &vec) {
  SetPyUtil();

  if (vec.size() >= INT_MAX) {
    LARCV_CRITICAL() << "Length of data vector too long to specify ndarray. "
                        "Use by batch call."
                     << std::endl;
    throw larbys();
  }
  int nd = 1;
  npy_intp dims[1];
  dims[0] = (int)vec.size();
  PyArrayObject *array = (PyArrayObject *)PyArray_SimpleNewFromData(
      nd, dims, NPY_FLOAT, (char *)&(vec[0]));
  return PyArray_Return(array);
}

PyObject *as_ndarray(const Image2D &img) {
  SetPyUtil();
  int* dim_data = new int[2];
  dim_data[0] = img.meta().cols();
  dim_data[1] = img.meta().rows();
  auto const &vec = img.as_vector();
  return PyArray_FromDimsAndData(2, dim_data, NPY_FLOAT, (char *)&(vec[0]));
}

void copy_array(PyObject *arrayin, const std::vector<float> &cvec) {
  SetPyUtil();
  PyArrayObject *ptr = (PyArrayObject *)(arrayin);
  /*
  std::cout<< PyArray_NDIM(ptr) << std::endl
           << PyArray_DIM(ptr,0)<<std::endl
           << PyArray_SIZE(ptr) << std::endl;
  */

  // Check dimension size is 1:
  if (PyArray_NDIM(ptr) != 1){
    throw std::exception();
  }

  if (cvec.size() != PyArray_SIZE(ptr))
    throw std::exception();
  npy_intp loc[1];
  loc[0] = 0;
  auto fptr = (float *)(PyArray_GetPtr(ptr, loc));
  for (size_t i = 0; i < size_t(PyArray_SIZE(ptr)); ++i) {
    // std::cout << fptr[i] << std::endl;
    fptr[i] = cvec[i];
  };
}

PyObject *as_caffe_ndarray(const Image2D &img) {
  SetPyUtil();
  int *dim_data = new int[2];
  dim_data[0] = img.meta().rows();
  dim_data[1] = img.meta().cols();
  auto const &vec = img.as_vector();
  return PyArray_FromDimsAndData(2, dim_data, NPY_FLOAT, (char *)&(vec[0]));
}

larcv::Image2D as_image2d_meta(PyObject *pyarray, ImageMeta meta) {
  SetPyUtil();
  float **carray;
  // Create C arrays from numpy objects:
  const int dtype = NPY_FLOAT;
  PyArray_Descr *descr = PyArray_DescrFromType(dtype);
  npy_intp dims[3];
  if (PyArray_AsCArray(&pyarray, (void **)&carray, dims, 2, descr) < 0) {
    logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
                               "ERROR: cannot convert to 2D C-array");
    throw larbys();
  }

  std::vector<float> res_data(dims[0] * dims[1], 0.);
  for (int i = 0; i < dims[0]; ++i) {
    for (int j = 0; j < dims[1]; ++j) {
      res_data[i * dims[1] + j] = (float)(carray[i][j]);
    }
  }
  PyArray_Free(pyarray, (void *)carray);

  Image2D res(std::move(meta), std::move(res_data));
  return res;
}

larcv::Image2D as_image2d(PyObject *pyarray) {
  SetPyUtil();
  float **carray;
  // Create C arrays from numpy objects:
  const int dtype = NPY_FLOAT;
  PyArray_Descr *descr = PyArray_DescrFromType(dtype);
  npy_intp dims[3];
  if (PyArray_AsCArray(&pyarray, (void **)&carray, dims, 2, descr) < 0) {
    logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
                               "ERROR: cannot convert to 2D C-array");
    throw larbys();
  }

  std::vector<float> res_data(dims[0] * dims[1], 0.);
  for (int i = 0; i < dims[0]; ++i) {
    for (int j = 0; j < dims[1]; ++j) {
      res_data[i * dims[1] + j] = (float)(carray[j][i]);
    }
  }
  PyArray_Free(pyarray, (void *)carray);

  ImageMeta meta((double)(dims[0]), (double)(dims[1]), (size_t)(dims[1]),
                 (size_t)(dims[0]), 0., 0., larcv::kINVALID_PLANE);

  Image2D res(std::move(meta), std::move(res_data));
  return res;
}

void fill_img_col(Image2D &img, std::vector<short> &adcs, const int col,
                  const int timedownsampling, const float pedestal) {
  if (col < 0 || col >= img.meta().cols())
    return;

  for (int iadc = 0; iadc < (int)adcs.size(); iadc++) {
    if (iadc <= img.meta().min_y() || iadc >= img.meta().max_y())
      continue;
    int irow = img.meta().row(iadc);
    float val = img.pixel(irow, col);
    img.set_pixel(irow, col, val + ((float)adcs.at(iadc) - pedestal));
  }
}

  // =================================================================
  // CHSTATUS UTILITIES
  // -------------------
  
  PyObject* as_ndarray( const ChStatus& status ) {
    // NOTE: CREATES WRAPPER    
    SetPyUtil();

    int nd = 1;
    int dim_data[1];
    dim_data[0] = status.as_vector().size();
    auto const &stat_v = status.as_vector();
        
    return PyArray_FromDimsAndData( nd, dim_data, NPY_USHORT, (char*)(&stat_v[0]) );
  }


  PyObject* as_ndarray( const EventChStatus& evstatus ) {
    // NOTE: CREATES NEW ARRAY
    
    SetPyUtil();

    int nd = 2;
    npy_intp dim_data[2];
    dim_data[0] = evstatus.ChStatusMap().size(); //  num planes
    int maxlen = 0;
    for (size_t p=0; p<evstatus.ChStatusMap().size(); p++) {
      int planelen = evstatus.Status( (larcv::PlaneID_t)p ).as_vector().size();
      if ( planelen > maxlen )
	maxlen = planelen;
    }
    dim_data[1] = maxlen;

    PyArrayObject* arr = (PyArrayObject*)PyArray_ZEROS( nd, dim_data, NPY_USHORT, 0 );

    short* data = (short*)PyArray_DATA(arr);
    
    for (size_t p=0; p<evstatus.ChStatusMap().size(); p++) {
      const std::vector<short>& chstatus = evstatus.Status( (larcv::PlaneID_t)p ).as_vector();
      for (size_t wire=0; wire<chstatus.size(); wire++) {
	*(data + p*dim_data[1] + wire) = chstatus[wire];
      }
    }
    
    return (PyObject*)arr;
  }

  ChStatus as_chstatus( PyObject* pyarray, const int planeid ) {
    
    SetPyUtil();
    const int dtype = NPY_USHORT;
    PyArray_Descr *descr = PyArray_DescrFromType(dtype);
    npy_intp dims[1];
    float *carray;
    if ( PyArray_AsCArray(&pyarray, (void *)&carray, dims, 1, descr) < 0 ) {
      logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
				 "ERROR: cannot convert numpy array to ChStatus Object");
      throw larbys();
    }

    std::vector<short> status_v( dims[0], 0 );
    for (int i = 0; i < dims[0]; ++i)
      status_v[i] = carray[i];
    PyArray_Free(pyarray,(void*)carray);
    
    ChStatus out( (larcv::PlaneID_t)planeid, std::move(status_v) );
    
    return out;
  }
  
  EventChStatus as_eventchstatus( PyObject* pyarray ) {
    
    SetPyUtil();
    const int dtype      = NPY_USHORT;
    PyArray_Descr *descr = PyArray_DescrFromType(dtype);
    int nd               = PyArray_NDIM( (PyArrayObject*)pyarray );
    npy_intp* dims       = PyArray_DIMS( (PyArrayObject*)pyarray );

    if ( nd!=2 ) {
      logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
				 "ERROR: unexpected dimension size for EventChStatus numpy array (should be two).");
      throw larbys();
    }
    
    short **carray;
    if ( PyArray_AsCArray(&pyarray, (void **)&carray, dims, nd, descr) < 0 ) {
      logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
				 "ERROR: cannot convert numpy array to ChStatus Object");
      throw larbys();
    }

    EventChStatus evchstatus;
    for (int i = 0; i < dims[0]; ++i) {    
      std::vector<short> status_v( dims[1], 0 );
      for (int j=0; j < dims[1]; ++j )
	status_v[ j ] = carray[i][j];
      ChStatus chstatus( (larcv::PlaneID_t)i, std::move(status_v) );
      evchstatus.Emplace( std::move(chstatus) );
    }
    
    PyArray_Free(pyarray,(void*)carray);
    
    return evchstatus;    
  }


  /**
   * convert std::vector<std::uint8_t> to a PyString
   *
   * note: used for json handling. performs a copy.
   *
   */
  PyObject* as_pystring( const std::vector<std::uint8_t>& buf ) {
    SetPyUtil();    
    return PyString_FromStringAndSize( (const char*)buf.data(), buf.size() );
  }

  /**
   * convert std::image2d into list of pixel coordinates and values
   *
   *
   */
  PyObject* as_pixelarray( const larcv::Image2D& img, const float threshold, larcv::msg::Level_t verbosity ) {
    SetPyUtil();
    // first, get a list of pixel values
    // each entry is (row,col,pixelvalue)
    larcv::logger::get("pyutils::as_pixelarray").set(verbosity);

    larcv::logger::get("pyutils::as_pixelarray").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
      << "extracting pixel list from image" << std::endl;
    std::vector<float> data_v;
    data_v.reserve( 3*img.meta().rows()*img.meta().cols() ); // maximum size
    size_t npts = 0;
    for ( size_t r=0; r<img.meta().rows(); r++ ) {
      for ( size_t c=0; c<img.meta().cols(); c++ ) {
        if ( img.pixel(r,c)>=threshold ) {
          data_v.push_back( (float)r );
          data_v.push_back( (float)c );
          data_v.push_back( (float)img.pixel(r,c) );
          npts++;
        }
      }
    }

    larcv::logger::get("pyutils::as_pixelarray").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
      << "create new array with " << npts << " pixels "
      << " (of max " << img.meta().rows()*img.meta().cols()
      << ", " << float(npts)/float(img.meta().rows()*img.meta().cols()) << " fraction)"
      << std::endl;
    
    npy_intp *dim_data = new npy_intp[2];    
    dim_data[0] = npts;
    dim_data[1] = 3;
    PyArrayObject* array = nullptr;
    try {
      array = (PyArrayObject*)PyArray_SimpleNew( 2, dim_data, NPY_FLOAT );
    }
    catch (std::exception& e ) {
      larcv::logger::get("pyutils::as_pixelarray").send( larcv::msg::kCRITICAL, __FUNCTION__, __LINE__, __FILE__ )
        << "trouble allocating new pyarray: " << e.what() << std::endl;
    }
    
    larcv::logger::get("pyutils::as_pixelarray").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
      << "fill array with " << npts << " points" << std::endl;
    
    for ( size_t ipt=0; ipt<npts; ipt++ ) {
      *((float*)PyArray_GETPTR2( array, ipt, 0 )) = data_v[3*ipt+0];
      *((float*)PyArray_GETPTR2( array, ipt, 1 )) = data_v[3*ipt+1];
      *((float*)PyArray_GETPTR2( array, ipt, 2 )) = data_v[3*ipt+2];      
    }

    larcv::logger::get("pyutils::as_pixelarray").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
      << "returned array" << std::endl;
    
    return (PyObject*)array;
  }

}

#endif
