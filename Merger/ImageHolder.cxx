#ifndef __IMAGEHOLDER_CXX__
#define __IMAGEHOLDER_CXX__

#include "ImageHolder.h"

namespace larcv {

  ImageHolder::ImageHolder(const std::string name)
    : ProcessBase(name)
  { _run = _subrun = _event = 0; }

  void ImageHolder::move(std::vector<larcv::Image2D>& dest)
  { dest = std::move(_image_v); }

}
#endif
