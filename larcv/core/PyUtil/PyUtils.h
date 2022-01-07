#ifndef __LARCV_PYUTILS_H__
#define __LARCV_PYUTILS_H__

struct _object;
typedef _object PyObject;

#ifndef __CLING__
#ifndef __CINT__
#include <Python.h>
#include "bytesobject.h"
#endif
#endif

#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/ROI.h"
#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/DataFormat/SparseImage.h"
#include "larcv/core/DataFormat/ClusterMask.h"
#include "larcv/core/DataFormat/Voxel2D.h"


namespace larcv {

  /// Utility function: call one-time-only numpy module initialization (you don't
  /// have to call)
  int SetPyUtil();
  ///
  PyObject *as_ndarray(const std::vector<float> &data);
  /// larcv::Image2D to numpy array converter
  PyObject *as_ndarray(const Image2D &img);
  /// larcv::Image2D to numpy array converter
  PyObject *as_caffe_ndarray(const Image2D &img);
  /// copy array
  void copy_array(PyObject *arrayin, const std::vector<float> &cvec);
  // void copy_array(PyObject *arrayin);//, const std::vector<float>& vec);

  Image2D as_image2d_meta(PyObject *, ImageMeta meta);

  Image2D as_image2d(PyObject *);

  // allows one to avoid some loops in python
  void fill_img_col(Image2D &img, std::vector<short> &adcs, const int col,
                    const int timedownsampling, const float pedestal = 0.0);

  // ChStatus helpers
  PyObject* as_ndarray( const ChStatus& chstatus );
  PyObject* as_ndarray( const EventChStatus& evstatus );

  ChStatus      as_chstatus( PyObject*, larcv::PlaneID_t planeid );
  EventChStatus as_eventchstatus( PyObject* );

  // Byte vector helper
  PyObject* as_pybytes( const std::vector<std::uint8_t>& buf );

  // as a list of 2D numpy array with (row,col,values) for each numpy row
  PyObject* as_pixelarray( const larcv::Image2D& img, const float threshold,
                           larcv::msg::Level_t verbosity=larcv::msg::kNORMAL );

  PyObject* as_pixelarray_with_selection( const larcv::Image2D& value_img,
                                          const larcv::Image2D& select_img,
                                          const float threshold,
                                          bool selectifabove=true,
                                          larcv::msg::Level_t verbosity=larcv::msg::kNORMAL );

  PyObject* as_union_pixelarray( const std::vector<const larcv::Image2D*> pimg_v,
                                 const float threshold,
                                 larcv::msg::Level_t verbosity );

  PyObject* as_union_pixelarray( const larcv::Image2D& img1,
                                 const larcv::Image2D& img2,
                                 const float threshold,
                                 larcv::msg::Level_t verbosity );

  PyObject* as_union_pixelarray( const larcv::Image2D& img1,
                                 const larcv::Image2D& img2,
                                 const larcv::Image2D& img3,
                                 const float threshold, larcv::msg::Level_t verbosity );

  PyObject* as_sparseimg_ndarray( const larcv::SparseImage&,
                                  larcv::msg::Level_t verbosity=larcv::msg::kNORMAL );


  SparseImage sparseimg_from_ndarray( PyObject*,
                                    const std::vector<larcv::ImageMeta>& meta,
                                    larcv::msg::Level_t verbosity );

  /// larcv::ClusterMask bbox to numpy array converter
  PyObject* as_ndarray_bbox(const larcv::ClusterMask &mask);
  /// larcv::ClusterMask points_v to numpy array converter
  PyObject* as_ndarray_mask(const larcv::ClusterMask &mask);
  /// direct larcv::ClusterMask points_v to numpy array conversion (not embedded)
  PyObject *as_ndarray_mask_pixlist(const larcv::ClusterMask &mask, float x_offset=0., float y_offset=0.);

  ClusterMask as_clustermask(PyObject *, PyObject *, ImageMeta, PyObject *);

  // sparsetensor2d into dense numpy array (np.float)
  PyObject *as_ndarray(const SparseTensor2D& data, bool clear_mem=false);  

}

#endif
