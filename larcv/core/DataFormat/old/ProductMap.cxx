#ifndef __LARCV_PRODUCTMAP_CXX__
#define __LARCV_PRODUCTMAP_CXX__

#include "ProductMap.h"

namespace larcv {

  //class ImageMeta;
  template<> std::string ProductName<larcv::ImageMeta>()
  { return "meta2d"; }

  //class Image2D;
  template<> std::string ProductName<larcv::Image2D>()
  { return "img2d"; }

}

#endif
