#ifndef __LARCV_CVUTIL_H__
#define __LARCV_CVUTIL_H__

#ifndef __CINT__
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#endif
#include "DataFormat/Image2D.h"

namespace larcv {
#ifndef __CINT__
  cv::Mat as_mat(const Image2D& larcv_img);
#endif
  
  Image2D imread(const std::string file_name);
  Image2D imread_gray(const std::string file_name);

}

#endif
