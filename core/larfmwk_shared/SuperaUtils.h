#ifndef __SUPERAUTILS_H__
#define __SUPERAUTILS_H__

#include "DataFormat/ImageMeta.h"

namespace larcv {
  namespace supera {

    template <class T>
    larcv::Image2D Extract(const ::larcv::ImageMeta& meta, const std::vector<T>& wires, const int time_offset=0);

  }
}

#include "SuperaUtils.inl"

#endif
