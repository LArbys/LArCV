#ifndef __SUPERAWIREALGO_CXX__
#define __SUPERAWIREALGO_CXX__

#include "SuperaWireAlgo.h"

namespace larlite {

  void SuperaWireAlgo::fillImage( larcv::Image2D& img, const std::vector< larlite::wire >& wires, const int time_offset) {
    _core.fill( img, wires, time_offset );
  }

  void SuperaWireAlgo::setVerbosity( unsigned short v ) {
    _core.set_verbosity(v);
  }
}

#endif
