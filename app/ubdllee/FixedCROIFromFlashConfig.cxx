#include "FixedCROIFromFlashConfig.h"

namespace larcv {
  namespace ubdllee {

    void FixedCROIFromFlashConfig::setDefaults() {

      croi_xcenters[0] =  65.0;
      croi_xcenters[1] = 190.0;
      croi_ycenters[0] = -60.0;
      croi_ycenters[1] =  60.0;
      croi_dzcenters[0] = -75.0;
      croi_dzcenters[1] =  75.0;
      split_fixed_ycroi = true;

    }

  }
}


