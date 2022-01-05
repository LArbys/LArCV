#ifndef __LARCV_PYUTILS_CXX__
#define __LARCV_PYUTILS_CXX__

#include <iostream>
#include "PyUtils.h"
#include "larcv/core/Base/larcv_logger.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifdef USE_PYTHON3
#include "numpy/arrayobject.h"
#include <cassert>
#else
#include <numpy/ndarrayobject.h>
#endif
namespace larcv {

int SetPyUtil() {
  static bool once = false;
  if (!once) {
    logger::get("PyUtils").send(larcv::msg::kNORMAL, __FUNCTION__, __LINE__, "calling import_array1(0)")  << std::endl;
    #ifdef USE_PYTHON3
        import_array1(0);
    #else
        import_array1(0);
    #endif
    once = true;
  }
  return 0;
}

larcv::ClusterMask as_clustermask(PyObject *pyarray_sparse_mask, PyObject *pyarray_box, ImageMeta meta, PyObject *pyarray_prob) {
  /*
  This function takes in a numpy matrix of shape (N,2) representing a binary
  sparse matrix within the box and a pyarray_box
  numpy array of shape (5) [x1,y1,x2,y2,class], and transforms them to the C++
  object ClusterMask, having built a meta, a BBox2D and a vector of Point2Ds
  from them. (Hopefully) it now also takes in a 1 value numpy array with the probability of class
  */
  SetPyUtil();
  float *carray_prob;
  // Create C arrays from numpy objects:
  const int dtype_prob = NPY_FLOAT;
  PyArray_Descr *descr_prob = PyArray_DescrFromType(dtype_prob);
  npy_intp dims_prob[1];
  if (PyArray_AsCArray(&pyarray_prob, (void *)&carray_prob, dims_prob, 1, descr_prob) < 0) {
    logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
                               "ERROR: cannot convert to 1D C-array");
    throw larbys("ERROR: cannot convert to 1D C-array");
  }
  float prob = carray_prob[0];

  float *carray_box;
  // Create C arrays from numpy objects:
  const int dtype_box = NPY_FLOAT;
  PyArray_Descr *descr_box = PyArray_DescrFromType(dtype_box);
  npy_intp dims_box[1];
  if (PyArray_AsCArray(&pyarray_box, (void *)&carray_box, dims_box, 1, descr_box) < 0) {
    logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
                               "ERROR: cannot convert to 1D C-array");
    throw larbys("ERROR: cannot convert to 1D C-array");
  }
  // if (dims_box[0] != 5){
  //   logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
  //                              "ERROR: box array needs to be 5 long, x1,y1,x2,y2,class");
  //   throw larbys("ERROR: box array needs to be 5 long, x1,y1,x2,y2,class");
  // }

  std::vector<float> box_v(dims_box[0],0.);
  for (int i=0;i<dims_box[0]; ++i){
    box_v[i] = (float)(carray_box[i]);
  }
  // if (sizeof(carray_box)/sizeof(carray_box[0]) != 5){
  //   logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
  //                              "ERROR: got this far");
  //   // std::string str1 = std::to_string(sizeof(*carray_box)/sizeof(*carray_box[0]));
  //   std::string str1 = std::to_string(box_v[0])+ " "+ std::to_string(box_v[1])+ " "+ std::to_string(box_v[2])+ " "+ std::to_string(box_v[3])+ " "+ std::to_string(box_v[4]);
  //   std::string str2 = "Error is: ";
  //   std::string str = str2+str1;
  //   throw larbys(str);
  // }
  BBox2D bbox((double)(carray_box[0]), (double)(carray_box[1]), (double)(carray_box[2]), (double)(carray_box[3]));


  InteractionID_t type = carray_box[4];

  // ImageMeta meta((double)(carray_box[2]-carray_box[0]), (double)(carray_box[3]-carray_box[1]), (size_t)(carray_box[3]-carray_box[1]),
  //                (size_t)(carray_box[2]-carray_box[0]), (double)(carray_box[0]), (double)(carray_box[1]), larcv::kINVALID_PLANE);



  float **carray_mask;
  const int dtype_mask = NPY_FLOAT;
  PyArray_Descr *descr_mask = PyArray_DescrFromType(dtype_mask);
  npy_intp dims_mask[2];

  if (PyArray_AsCArray(&pyarray_sparse_mask, (void **)&carray_mask, dims_mask, 2, descr_mask) < 0) {
    logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
                               "ERROR: cannot convert to 2D C-array");
    throw larbys("ERROR: cannot convert to 2D C-array");
  }
  if (dims_mask[1] != 2){
    logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
                               "ERROR: mask array needs to be Npoints by 2");
    throw larbys("ERROR: mask array needs to be Npoints by 2");
  }

  std::vector<Point2D> points_v(dims_mask[0], Point2D() );
  std::vector<float> points_v_ft(dims_mask[0]*2, 0.);
  for (int i=0; i < dims_mask[0]; i++) {
    points_v[i] = Point2D(carray_mask[i][0], carray_mask[i][1]);
    points_v_ft[2*i] = carray_mask[i][0];
    points_v_ft[2*i+1] = carray_mask[i][1];
  }
  BBox2D dummy_box(1600,5000,1600,5000,kINVALID_PROJECTIONID);
  std::vector<Point2D> dummy_v(0,Point2D(0,0));
  ClusterMask cmask(dummy_box, meta, dummy_v, 0);
  cmask.probability_of_class = prob;
  cmask.box = bbox;
  cmask.meta = meta;
  cmask.points_v = points_v;
  cmask.type = type;

  // cmask._box[0] = carray_box[0];
  // cmask._box[1] = carray_box[1];
  // cmask._box[2] = carray_box[2];
  // cmask._box[3] = carray_box[3];
  // cmask._box[4] = carray_box[4];
  // cmask._mask = points_v_ft;

  PyArray_Free(pyarray_box, (void *)carray_box);
  PyArray_Free(pyarray_sparse_mask, (void *)carray_mask);
  PyArray_Free(pyarray_prob, (void *)carray_prob);

  return cmask;
}

  PyObject *as_ndarray_mask(const ClusterMask &mask) {
    SetPyUtil();
    npy_intp dim_data[2];
    dim_data[0] = (mask.box.width()/mask.meta.pixel_width())+1; //Add one for the 0th spot
    dim_data[1] = (mask.box.height()/mask.meta.pixel_height())+1; //Add one for the 0th spot
    std::vector<float> const &vec = mask.as_vector_mask();//= copy_v;

    return PyArray_Transpose(((PyArrayObject*)(PyArray_SimpleNewFromData(2, dim_data, NPY_FLOAT, (char *)&(vec[0])))),NULL);
  }

  PyObject *as_ndarray_mask_pixlist(const ClusterMask &mask, float x_offset, float y_offset ) {
    SetPyUtil();

    int nd = 2;
    npy_intp dim_data[2];
    dim_data[0] = mask.points_v.size();
    dim_data[1] = 2;

    PyArrayObject* arr = (PyArrayObject*)PyArray_ZEROS( nd, dim_data, NPY_FLOAT, 0 );
    float* data = (float*)PyArray_DATA(arr);

    for ( size_t ipt=0; ipt<mask.points_v.size(); ipt++ ) {
      float x = x_offset+mask.points_v.at(ipt).x;
      float y = y_offset+mask.points_v.at(ipt).y;
      *( data + 2*ipt )     = x;
      *( data + 2*ipt + 1 ) = y;
    }

    return (PyObject*)arr;
  }

PyObject *as_ndarray_bbox(const ClusterMask &mask) {
  SetPyUtil();
  npy_intp dim_data[1];
  dim_data[0] = 5;
  std::vector<float> const &vec = mask.as_vector_box();

  return PyArray_Transpose(((PyArrayObject*)(PyArray_SimpleNewFromData(1, dim_data, NPY_FLOAT, (char *)&(vec[0])))),NULL);
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
  npy_intp dim_data[2];
  dim_data[0] = img.meta().cols();
  dim_data[1] = img.meta().rows();
  auto const &vec = img.as_vector();


  return PyArray_SimpleNewFromData( 2, dim_data, NPY_FLOAT, (void*)vec.data() );
  //return PyArray_FromDimsAndData(2, dim_data, NPY_FLOAT, (char *)&(vec[0]));
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
  npy_intp dims[2];
  if (PyArray_AsCArray(&pyarray, (void **)&carray, dims, 2, descr) < 0) {
    logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
                               "ERROR: cannot convert to 2D C-array");
    throw larbys("ERROR: cannot convert to 2D C-array");
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
  npy_intp dims[2];
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
   * convert std::vector<std::uint8_t> to a PyBytes
   *
   * note: used for json handling. performs a copy.
   *
   */
  PyObject* as_pybytes( const std::vector<std::uint8_t>& buf ) {
    SetPyUtil();
    return PyBytes_FromStringAndSize( (const char*)buf.data(), buf.size() );
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
  PyObject* as_sparseimg_ndarray( const larcv::SparseImage& sparseimg,
                                  larcv::msg::Level_t verbosity ) {

    larcv::SetPyUtil();

    size_t stride = 2+sparseimg.nfeatures();
    size_t npts   = sparseimg.pixellist().size()/stride;
    if ( verbosity==larcv::msg::kDEBUG ) {
      larcv::logger::get("pyutils::as_ndarray(sparseimg)").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
        << " npts=" << npts << " stride=" << stride << std::endl;
    }

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


    if ( verbosity==larcv::msg::kDEBUG ) {
      larcv::logger::get("pyutils::as_ndarray(sparseimg)").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
        << "fill array with " << npts << " points" << std::endl;
    }

    for ( size_t ipt=0; ipt<npts; ipt++ ) {
      for ( size_t ifeat=0; ifeat<stride; ifeat++ ) {
        *((float*)PyArray_GETPTR2( array, ipt, ifeat )) = sparseimg.pixellist()[stride*ipt+ifeat];
      }
    }

    if ( verbosity==larcv::msg::kDEBUG ) {
      larcv::logger::get("pyutils::as_ndarray(sparseimg)").send( larcv::msg::kDEBUG, __FUNCTION__, __LINE__, __FILE__ )
        << "returned array" << std::endl;
    }

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
    SetPyUtil();

    int dtype = PyArray_TYPE((PyArrayObject*)ndarray);

    float **carray;
    // Create C arrays from numpy objects:
    PyArray_Descr *descr = PyArray_DescrFromType(dtype);
    npy_intp dims[2];
    if (PyArray_AsCArray(&ndarray, (void **)&carray, dims, 2, descr) < 0) {
      std::stringstream msg;
      msg << "ERROR: cannot convert to 2D C-array." << std::endl;
      logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
                                 msg.str());
      throw larbys();
    }
    // checks
    if ( (int)meta_v.size()!=dims[1]-2 ) {
      std::stringstream msg;
      msg << "ERROR: number of meta (" << meta_v.size() << ") "
          << " != num features (" << dims[1]-2 << ")"
          << std::endl;
      logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
                                 msg.str());
      throw larbys("larcv::sparseimg_from_ndarray error");
    }

    std::vector<float> data(dims[0] * dims[1], 0.);
    for (int i = 0; i < dims[0]; ++i) {
      //std::cout << "N[" << i << "] :";
      for (int j = 0; j < dims[1]; ++j) {
        data[i * dims[1] + j] = (float)(carray[i][j]);
        //std::cout << (float)(carray[i][j]) << " ";
      }
      //std::cout << std::endl;
    }
    PyArray_Free(ndarray, (void *)carray);

    int nfeatures = dims[1]-2;
    int npts      = dims[0];
    SparseImage sparseimg(nfeatures, npts, data, meta_v); // empty imae

    return sparseimg;
  }

}

#endif
