
#ifndef __LARCV_PRODUCTMAP_H__
#define __LARCV_PRODUCTMAP_H__

#include <string>

namespace larcv {

  template<class T> std::string ProductName();

  class ImageMeta;
  template<> std::string ProductName<larcv::ImageMeta>();

  class Image2D;
  template<> std::string ProductName<larcv::Image2D>();

}
#endif
