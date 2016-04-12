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
      
      Cropper() : _time_padding(10)
		, _wire_padding(10)
		, _target_width(247)
		, _target_height(247)
		, _compression_factor(0)
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
      WTRangeArray_t WireTimeBoundary( const U& mcs ) const;
      WTRangeArray_t WireTimeBoundary( const U& mcs, const std::vector<V>& sch_v ) const;

      ::larcv::ROI ParticleROI( const T& mct ) const;
      ::larcv::ROI ParticleROI( const U& mcs ) const;
      ::larcv::ROI ParticleROI( const U& mcs, const std::vector<V>& sch_v ) const;

      /**
	 Given a range and corresponding plane ID (plan_id == #plane is considered time), \n
	 perform padding & cropping to a multiple of target width/height.
       */
      //Range_t      Format( const Range_t& range, unsigned short plane_id ) const;
      /**
	 Given a set of ranges (all planes & time), \n
	 perform padding & cropping to a multiple of target width/height.
       */
      //RangeArray_t Format( const RangeArray_t& boundary                  ) const;

    private:
 
      std::vector<larcv::ImageMeta> WTRange2BB(const WTRangeArray_t&) const;
      
      unsigned int _time_padding;  ///< Padding in time axis (height) for Cropper::Format function
      unsigned int _wire_padding;  ///< Padding in wire axis (width) for Cropper::Format function
      unsigned int _target_width;  ///< Unit-size (horizontal, wire, or width) for an output image of Cropper::Format function
      unsigned int _target_height; ///< Unit-size (vertical, time, or height) for an output image of Cropper::Format function
      /**
	 A scale factor used to compress image. If 0, original image is sampled in both height in \n
	 multiple of target size to contain the ROI and automatically compressed (i.e. compression \n
	 factor is computed per image and varies). If set to non-zero value, image is sampled from \n
	 the ROI center for the target size times this compression factor, then compressed (i.e. \n
	 sample image size and hence compression factor stays constant from one image to another). 
       */
      unsigned int _compression_factor;

      unsigned int _min_width;
      unsigned int _min_height;
    };
  }
}

#include "Cropper.inl"

#endif
