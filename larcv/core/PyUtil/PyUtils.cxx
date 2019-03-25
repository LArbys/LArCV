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
   * @param[in] img Input image from which to take values
   * @param[in] threshold pixel value must be greater than or equal to threshold to be included
   * @param[in] verbosity level of verbosity for function
   * @return numpy array with shape (N,3)
   *
   */
  PyObject* as_pixelarray( const larcv::Image2D& img, const float threshold, larcv::msg::Level_t verbosity ) {
    SetPyUtil();
    // first, get a list of pixel values
    // each entry is (row,col,pixelvalue)
    larcv::logger::get("pyutils::as_pixelarray").set(verbosity);

    if ( verbosity==larcv::msg::kDEBUG )
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

    if ( verbosity==larcv::msg::kDEBUG )
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
      throw larbys();
    }


    if ( verbosity==larcv::msg::kDEBUG )
      larcv::logger::get("pyutils::as_pixelarray").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
        << "fill array with " << npts << " points" << std::endl;

    for ( size_t ipt=0; ipt<npts; ipt++ ) {
      *((float*)PyArray_GETPTR2( array, ipt, 0 )) = data_v[3*ipt+0];
      *((float*)PyArray_GETPTR2( array, ipt, 1 )) = data_v[3*ipt+1];
      *((float*)PyArray_GETPTR2( array, ipt, 2 )) = data_v[3*ipt+2];
    }

    if ( verbosity==larcv::msg::kDEBUG )
      larcv::logger::get("pyutils::as_pixelarray").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
        << "returned array" << std::endl;

    return (PyObject*)array;
  }

  /**
   * convert std::image2d into list of pixel coordinates and values.
   * which pixels that are saved are conditioned on values of another image
   *
   * @param[in] value_img Image2D from which we get the values of the pixel list
   * @param[in] select_img Image2D with which we select the values of the list
   * @param[in] threshold threshold value for each pixel
   * @param[in] selectabove if true, pixel in select_img must be above some threshold, below if false
   * @param[in] verbosity level of debug statements
   *
   */
  PyObject* as_pixelarray_with_selection( const larcv::Image2D& value_img,
                                          const larcv::Image2D& select_img,
                                          const float threshold, bool selectifabove,
                                          larcv::msg::Level_t verbosity ) {
    SetPyUtil();
    // first, get a list of pixel values
    // each entry is (row,col,pixelvalue)
    larcv::logger::get("pyutils::as_pixelarray").set(verbosity);

    // this only makes sense if the metas mean the same thing
    if ( value_img.meta()!=select_img.meta() ) {
      larcv::logger::get("pyutils::as_pixelarray").send( larcv::msg::kCRITICAL, __FUNCTION__, __LINE__, __FILE__ )
        << "value and select images must have the same meta" << std::endl;
      throw larbys();
    }

    if ( verbosity==larcv::msg::kDEBUG )
      larcv::logger::get("pyutils::as_pixelarray").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
      << "extracting pixel list from image" << std::endl;
    std::vector<float> data_v;
    data_v.reserve( 3*value_img.meta().rows()*value_img.meta().cols() ); // maximum size
    size_t npts = 0;
    for ( size_t r=0; r<value_img.meta().rows(); r++ ) {
      for ( size_t c=0; c<value_img.meta().cols(); c++ ) {
        float selectpix = select_img.pixel(r,c);
        if ( (selectifabove && selectpix>=threshold) || (!selectifabove && selectpix<=threshold) ) {
          data_v.push_back( (float)r );
          data_v.push_back( (float)c );
          data_v.push_back( (float)value_img.pixel(r,c) );
          npts++;
        }
      }
    }

    if ( verbosity==larcv::msg::kDEBUG )
      larcv::logger::get("pyutils::as_pixelarray").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
        << "create new array with " << npts << " pixels "
        << " (of max " << value_img.meta().rows()*value_img.meta().cols()
        << ", " << float(npts)/float(value_img.meta().rows()*value_img.meta().cols()) << " fraction)"
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
      throw larbys();
    }


    if ( verbosity==larcv::msg::kDEBUG )
      larcv::logger::get("pyutils::as_pixelarray").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
        << "fill array with " << npts << " points" << std::endl;

    for ( size_t ipt=0; ipt<npts; ipt++ ) {
      *((float*)PyArray_GETPTR2( array, ipt, 0 )) = data_v[3*ipt+0];
      *((float*)PyArray_GETPTR2( array, ipt, 1 )) = data_v[3*ipt+1];
      *((float*)PyArray_GETPTR2( array, ipt, 2 )) = data_v[3*ipt+2];
    }

    if ( verbosity==larcv::msg::kDEBUG )
      larcv::logger::get("pyutils::as_pixelarray").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
        << "returned array" << std::endl;

    return (PyObject*)array;
  }


  /**
   * convert vector< std::image2d > into a union of pixel coordinates and values
   *
   * @param[in] pimg_v vector of image (pointers) to be get pixel values from
   * @param[in] pixel value for at least one image, must be greater than or equal to threshold to be included
   * @param[in] verbosity level of verbosity for function
   * @return numpy array with shape (N,2+M) where M=images given, N=pixels above threshold in one image
   *
   */
  PyObject* as_union_pixelarray( const std::vector<const larcv::Image2D*> pimg_v,
                                 const float threshold,
                                 larcv::msg::Level_t verbosity ) {
    SetPyUtil();
    // first, get a list of pixel values
    // each entry is (row,col,pixelvalue)
    larcv::logger::get("pyutils::as_union_pixelarray").set(verbosity);

    if ( verbosity==larcv::msg::kDEBUG )
      larcv::logger::get("pyutils::as_union_pixelarray").send( larcv::msg::kDEBUG,
                                                               __FUNCTION__, __LINE__, __FILE__ )
        << "extracting pixel list from image" << std::endl;

    if ( pimg_v.size()==0 ) {
      larcv::logger::get("pyutils::as_union_pixelarray").send( larcv::msg::kCRITICAL,
                                                               __FUNCTION__, __LINE__, __FILE__ )
        << "no images given" << std::endl;
      throw std::runtime_error("pyutils::as_union_pixelarray: no imaes given");
    }

    size_t nimgs = pimg_v.size();

    // first image defines the array size
    size_t ncols = pimg_v.front()->meta().cols();
    size_t nrows = pimg_v.front()->meta().rows();

    // check the others
    for ( size_t iimg=1; iimg<nimgs; iimg++ ) {
      auto const& pimg = pimg_v[iimg];
      if ( ncols<pimg->meta().cols() ) {
        throw std::runtime_error("pyutils::as_union_pixelarray: image ncols bigger than first");
      }
      if ( nrows<pimg->meta().rows() ) {
        throw std::runtime_error("pyutils::as_union_pixelarray: image nrows bigger than first");
      }
    }

    std::vector<float> data_v;
    data_v.reserve( (2+nimgs)*nrows*ncols ); // maximum size
    size_t npts = 0;

    for ( size_t c=0; c<ncols; c++ ) {
      for ( size_t r=0; r<nrows; r++ ) {
        // for every pixel, get value in all images in input vector
        // mark for save if any are above threshold
        // if saving, add to data vector
        std::vector<float> pixvals(nimgs,0);
        bool fill = false;
        for ( size_t iimg=0; iimg<nimgs; iimg++ ) {
          auto const& pimg = pimg_v[iimg];
          float pixval = pimg->pixel(r,c);
          if ( pixval>=threshold ) {
            fill = true;
            pixvals[iimg] = pixval;
          }
        }

        if ( fill ) {
          data_v.push_back( (float)r );
          data_v.push_back( (float)c );
          for ( auto& pixval : pixvals ) {
            data_v.push_back( pixval );
            npts++;
          }
        }

      }//end of col loop
    }//end of row loop

    if ( verbosity==larcv::msg::kDEBUG )
      larcv::logger::get("pyutils::as_union_pixelarray").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
        << "create new array with " << npts << " pixels "
        << " (of max " << nimgs*ncols*nrows
        << ", " << float(npts)/float(nimgs*ncols*nrows) << " fraction)"
        << std::endl;

    npy_intp *dim_data = new npy_intp[2];
    dim_data[0] = npts;
    dim_data[1] = 2+(int)nimgs;
    PyArrayObject* array = nullptr;
    try {
      array = (PyArrayObject*)PyArray_SimpleNew( 2, dim_data, NPY_FLOAT );
    }
    catch (std::exception& e ) {
      larcv::logger::get("pyutils::as_union_pixelarray").send( larcv::msg::kCRITICAL, __FUNCTION__, __LINE__, __FILE__ )
        << "trouble allocating new pyarray: " << e.what() << std::endl;
      throw larbys();
    }


    if ( verbosity==larcv::msg::kDEBUG )
      larcv::logger::get("pyutils::as_union_pixelarray").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
        << "fill array with " << npts << " points" << std::endl;

    size_t blocksize = 2+nimgs;
    for ( size_t ipt=0; ipt<npts; ipt++ ) {
      *((float*)PyArray_GETPTR2( array, ipt, 0 )) = data_v[blocksize*ipt+0];
      *((float*)PyArray_GETPTR2( array, ipt, 1 )) = data_v[blocksize*ipt+1];
      for ( size_t iimg=0; iimg<nimgs; iimg++ )
        *((float*)PyArray_GETPTR2( array, ipt, 2+iimg )) = data_v[blocksize*ipt+2+iimg];
    }

    if ( verbosity==larcv::msg::kDEBUG )
      larcv::logger::get("pyutils::as_union_pixelarray").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
        << "returned array" << std::endl;

    return (PyObject*)array;
  }

  /**
   * wrapper for as_union_pixelarray with vector. for ease of use in python.
   *
   * @param[in] img1 first input image
   * @param[in] img2 second input image
   * @param[in] threshold pixel value for at least one image, must be greater than or equal to threshold to be included
   * @param[in] verbosity level of verbosity for function
   * @return numpy array with shape (N,2+M) where M=images given, N=pixels above threshold in one image
   *
   */
  PyObject* as_union_pixelarray( const larcv::Image2D& img1, const larcv::Image2D& img2,
                                 const float threshold,
                                 larcv::msg::Level_t verbosity ) {
    std::vector< const larcv::Image2D* > img_v;
    img_v.push_back( &img1 );
    img_v.push_back( &img2 );
    return as_union_pixelarray( img_v, threshold, verbosity );
  }

  /**
   * wrapper for as_union_pixelarray with vector. for ease of use in python.
   *
   * @param[in] img1 first input image
   * @param[in] img2 second input image
   * @param[in] img3 third input image
   * @param[in] threshold pixel value for at least one image, must be greater than or equal to threshold to be included
   * @param[in] verbosity level of verbosity for function
   * @return numpy array with shape (N,2+M) where M=images given, N=pixels above threshold in one image
   *
   */
  PyObject* as_union_pixelarray( const larcv::Image2D& img1, const larcv::Image2D& img2,
                                 const larcv::Image2D& img3,
                                 const float threshold, larcv::msg::Level_t verbosity ) {
    std::vector< const larcv::Image2D* > img_v;
    img_v.push_back( &img1 );
    img_v.push_back( &img2 );
    img_v.push_back( &img3 );
    return as_union_pixelarray( img_v, threshold, verbosity );
  }

  /**
   * convert sparse image data into a numpy array
   *
   * @param[in] sparseimg SparseImage object to convert
   * @param[in] verbosity verbose level with 0 the most verbose and 2 quietest
   * @return a numpy array with shape (N,F) where N is number of points
   *         and F is number of features. First two values are (row,col)
   */
  PyObject* as_ndarray( const larcv::SparseImage& sparseimg,
                        larcv::msg::Level_t verbosity ) {

    larcv::SetPyUtil();

    size_t stride = 2+sparseimg.nfeatures();
    size_t npts   = sparseimg.pixellist().size()/stride;
    if ( verbosity==larcv::msg::kDEBUG )
      larcv::logger::get("pyutils::as_ndarray(sparseimg)").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
        << " npts=" << npts << " stride=" << stride << std::endl;

    npy_intp *dim_data = new npy_intp[2];
    dim_data[0] = npts;
    dim_data[1] = stride;
    PyArrayObject* array = nullptr;
    try {
      array = (PyArrayObject*)PyArray_SimpleNew( 2, dim_data, NPY_FLOAT );
    }
    catch (std::exception& e ) {
      larcv::logger::get("pyutils::as_ndarray(sparseimage)").send( larcv::msg::kCRITICAL, __FUNCTION__, __LINE__, __FILE__ )
        << "trouble allocating new pyarray: " << e.what() << std::endl;
      throw larbys();
    }


    if ( verbosity==larcv::msg::kDEBUG )
      larcv::logger::get("pyutils::as_ndarray(sparseimg)").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
        << "fill array with " << npts << " points" << std::endl;

    for ( size_t ipt=0; ipt<npts; ipt++ ) {
      for ( size_t ifeat=0; ifeat<stride; ifeat++ ) {
        *((float*)PyArray_GETPTR2( array, ipt, ifeat )) = sparseimg.pixellist()[stride*ipt+ifeat];
      }
    }

    if ( verbosity==larcv::msg::kDEBUG )
      larcv::logger::get("pyutils::as_ndarray(sparseimg)").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
        << "returned array" << std::endl;

    return (PyObject*)array;

  }

  /**
   * convert numpy array into SparseImage object
   *
   * @param[in] ndarray numpy array with (N,F) shape. F>2 as first two values
   *            are (row,col) coordinates.
   * @param[in] meta_v Vector of larcv::ImageMeta for each feature
   * @param[in] verbosity verbose level with 0 the most verbose and 2 quietest
   * @return a numpy array with shape (N,F) where N is number of points
   *         and F is number of features. First two values are (row,col)
   */
  larcv::SparseImage sparseimg_from_ndarray( PyObject* ndarray,
                        const std::vector<larcv::ImageMeta>& meta_v,
                        larcv::msg::Level_t verbosity )
  {

    float **carray;
    // Create C arrays from numpy objects:
    const int dtype = NPY_FLOAT;
    PyArray_Descr *descr = PyArray_DescrFromType(dtype);
    npy_intp dims[3];
    if (PyArray_AsCArray(&ndarray, (void **)&carray, dims, 2, descr) < 0) {
      logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
                                 "ERROR: cannot convert to 2D C-array");
      throw larbys();
    }
    // checks
    if ( (int)meta_v.size()!=dims[1]-2 ) {
      logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
                                 "ERROR: number of meta != num features");
      throw larbys("larcv::sparseimg_from_ndarray error");
    }

    std::vector<float> data(dims[0] * dims[1], 0.);
    for (int i = 0; i < dims[0]; ++i) {
      for (int j = 0; j < dims[1]; ++j) {
        data[i * dims[1] + j] = (float)(carray[i][j]);
      }
    }
    PyArray_Free(ndarray, (void *)carray);

    int nfeatures = dims[1]-2;
    int npts      = dims[0];
    SparseImage sparseimg(nfeatures, npts, data, meta_v); // empty imae

    return sparseimg;
  }
}

#endif
