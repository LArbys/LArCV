#ifndef __SUPERAWIREALGO_CXX__
#define __SUPERAWIREALGO_CXX__

#include "SuperaWireAlgo.h"

namespace larlite {
  void SuperaWireAlgo::fillImage( larcv::Image2D& img, const std::vector< larlite::wire >& wires, const int time_offset) {
    _core.fill( img, wires, time_offset );
  }
}

#endif
