/**
 * \file SuperaWireAlgo.h
 *
 * \ingroup APILArLite
 * 
 * \brief Class def header for a class SuperaWireAlgo
 *
 * A wrapper class that allows one to setup and call the fill
 * function of SuperaCore which converts Wire to Image2D
 *
 * @author kazuhiro
 */

/** \addtogroup APILArLite

    @{*/
#ifndef __SUPERAWIREALGO__
#define __SUPERAWIREALGO__

#include <vector>

#include "DataFormat/Image2D.h"

#include "DataFormat/wire.h"
#include "DataFormat/mctruth.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/simch.h"
#include "DataFormat/opdetwaveform.h"

#include "SuperaCore.h"

namespace larlite {

  class SuperaWireAlgo {
  public: 
    SuperaWireAlgo() {};
    virtual ~SuperaWireAlgo() {};
    
    void fillImage( larcv::Image2D& img, const std::vector< larlite::wire >& wires, const int time_offset);

  protected:
    
    ::larcv::supera::SuperaCore<larlite::opdetwaveform, larlite::wire,
      larlite::mctruth,
      larlite::mctrack, larlite::mcshower,
      larlite::simch> _core;
  };
  
}



#endif
