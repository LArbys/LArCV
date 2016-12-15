#ifndef __LARCV_CVUTIL_H__
#define __LARCV_CVUTIL_H__

#ifndef __CINT__
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#endif
#include "DataFormat/Image2D.h"

namespace larcv {
#ifndef __CINT__
  /// Image2D to cv::Mat converter (not supported in ROOT5 CINT)
  cv::Mat as_mat(const Image2D& larcv_img);
  cv::Mat as_gray_mat(const Image2D& larcv_img);
#endif
  /// larcv::Image2D creator from an image file
  Image2D imread(const std::string file_name);
  /// Gray scale larcv::Image2D creator from an image file
  Image2D imread_gray(const std::string file_name);
}

#endif
