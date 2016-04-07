#ifndef __LARCV_CVUTIL_H__
#define __LARCV_CVUTIL_H__

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "DataFormat/Image2D.h"

namespace larcv {

  cv::Mat as_mat(const Image2D& larcv_img);

  Image2D imread(const std::string file_name);

}

#endif
