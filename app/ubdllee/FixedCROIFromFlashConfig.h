#ifndef __FIXED_CROI_FROM_FLASH_CONFIG__
#define __FIXED_CROI_FROM_FLASH_CONFIG__

namespace larcv {
  namespace ubdllee {

    class FixedCROIFromFlashConfig {

    public:

      FixedCROIFromFlashConfig() { setDefaults(); };
      virtual ~FixedCROIFromFlashConfig() {};

      void setDefaults();
      
      // parameters
      float croi_xcenters[2];
      float croi_ycenters[2];
      float croi_dzcenters[2];
      bool  split_fixed_ycroi;
      
    };

  }
}

#endif
