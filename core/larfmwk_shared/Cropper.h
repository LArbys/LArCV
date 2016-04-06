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
namespace larcv {

  namespace supera {

    typedef ::larcv::Range<unsigned int> WTRange_t;
    typedef std::vector<larcv::supera::WTRange_t> WTRangeArray_t;

    template <class T, class U>
    class Cropper : public ::larcv::larcv_base {

    public:
      
      Cropper()
      { configure(10,10,247,247,0); }
      
      virtual ~Cropper() {}
      
      void configure(unsigned int time_padding,
		     unsigned int wire_padding,
		     unsigned int target_width,
		     unsigned int target_height,
		     unsigned int compression_factor);
      /**
	 Given single MCTrack, returns length 4 range array (3 planes + time) \n
	 which contains all trajectory points of input MCTrack.
      */
      WTRangeArray_t WireTimeBoundary( const T& mct ) const;
      WTRangeArray_t WireTimeBoundary( const U& mcs ) const;

      ::larcv::ROI ParticleROI( const T& mct ) const;
      ::larcv::ROI ParticleROI( const U& mcs ) const;

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
    };
  }
}

#include "Cropper.inl"

#endif
