#ifndef __SUPERA_TYPES_H__
#define __SUPERA_TYPES_H__

#include "DataFormat/ROI.h"
#include <vector>
#include <utility>

namespace larcv {
  namespace supera {

    typedef std::vector<larcv::ROI> ParticleROIArray_t;

    typedef larcv::ROI PrimaryROI_t;

    typedef std::pair<larcv::supera::PrimaryROI_t,larcv::supera::ParticleROIArray_t> InteractionROI_t;
    
  }
}

#endif
