#ifndef __IMAGEHOLDER_CXX__
#define __IMAGEHOLDER_CXX__

#include "ImageHolder.h"

namespace larcv {

  ImageHolder::ImageHolder(const std::string name)
    : ProcessBase(name)
  {}

  void ImageHolder::move(std::vector<larcv::Image2D>& dest)
  { dest = std::move(_image_v); }

  void ImageHolder::move(std::vector<larcv::ROI>& dest)
  { dest = std::move(_roi_v); }

}
#endif
