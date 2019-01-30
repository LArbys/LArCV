#ifndef __SUPERA_CROPPER_H__
#define __SUPERA_CROPPER_H__

#include <vector>

// LArSoft
//#include "MCBase/MCTrack.h"
//#include "MCBase/MCShower.h"
//#include "Simulation/SimChannel.h"
//#include "SimulationBase/MCParticle.h"

// LArCV
#include "Base/larcv_base.h"
#include "Base/Range.h"
#include "DataFormat/ROI.h"
#include "DataFormat/DataFormatUtil.h"
#include "FMWKInterface.h"
namespace larcv {

  namespace supera {

    typedef ::larcv::Range<unsigned int> WTRange_t;
    typedef std::vector<larcv::supera::WTRange_t> WTRangeArray_t;

    template <class T, class U, class V>
    class Cropper : public ::larcv::larcv_base {

    public:
      
      Cropper() : _max_time_tick(9600)
		, _time_padding(10)
		, _wire_padding(10)
		, _min_width(10)
		, _min_height(14)
      {}
      
      virtual ~Cropper() {}
      
      void configure(const larcv::supera::Config_t& cfg);
      /**
	 Given single MCTrack, returns length 4 range array (3 planes + time) \n
	 which contains all trajectory points of input MCTrack.
      */
      WTRangeArray_t WireTimeBoundary( const T& mct ) const;
      WTRangeArray_t WireTimeBoundary( const T& mct, const std::vector<V>& sch_v ) const;
      WTRangeArray_t WireTimeBoundary( const U& mcs ) const;
      WTRangeArray_t WireTimeBoundary( const U& mcs, const std::vector<V>& sch_v ) const;

      ::larcv::ROI ParticleROI( const T& mct, const int time_offset                              ) const;
      ::larcv::ROI ParticleROI( const T& mct, const std::vector<V>& sch_v, const int time_offset ) const;
      ::larcv::ROI ParticleROI( const U& mcs, const int time_offset                              ) const;
      ::larcv::ROI ParticleROI( const U& mcs, const std::vector<V>& sch_v, const int time_offset ) const;

    private:
 
      std::vector<larcv::ImageMeta> WTRange2BB(const WTRangeArray_t&) const;

      unsigned int _max_time_tick; ///< Maximum tick number in time
      unsigned int _time_padding;  ///< Padding in time axis (height) for Cropper::Format function
      unsigned int _wire_padding;  ///< Padding in wire axis (width) for Cropper::Format function
      unsigned int _min_width;
      unsigned int _min_height;
    };
  }
}

#include "Cropper.inl"

#endif
