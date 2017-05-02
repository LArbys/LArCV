#ifndef __SUPERA_LAR2IMAGE_H__
#define __SUPERA_LAR2IMAGE_H__

#include "FMWKInterface.h"
#include "DataFormat/Image2D.h"
#include "DataFormat/Pixel2DCluster.h"

namespace supera {

  //
  // Hit => Image2D
  //
  
  larcv::Image2D Hit2Image2D(const larcv::ImageMeta& meta,
			     const std::vector<supera::LArHit_t>& hits,
			     const int time_offset=0);
  
  std::vector<larcv::Image2D> Hit2Image2D(const std::vector<larcv::ImageMeta>& meta_v,
					  const std::vector<supera::LArHit_t>& hits,
					  const int time_offset=0);
  
  //
  // Wire => Image2D
  //

  larcv::Image2D Wire2Image2D(const larcv::ImageMeta& meta,
			      const std::vector<supera::LArWire_t>& wires,
			      const int time_offset=0);
  
  std::vector<larcv::Image2D> Wire2Image2D(const std::vector<larcv::ImageMeta>& meta_v,
					   const std::vector<supera::LArWire_t>& wires,
					   const int time_offset=0);

  //
  // OpDigit => Image2D
  //
  larcv::Image2D OpDigit2Image2D(const larcv::ImageMeta& meta,
				 const std::vector<supera::LArOpDigit_t>& opdigit_v,
				 int time_offset=0);

  //
  // SimChannel => Image2D
  //
  std::vector<larcv::Image2D> SimCh2Image2D(const std::vector<larcv::ImageMeta>& meta_v,
					    const std::vector<larcv::ROIType_t>& track2type_v,
					    const std::vector<supera::LArSimCh_t>& sch_v,
					    const int time_offset);


  //
  // SimChannel => Pixel2DCluster
  //
  std::vector<std::vector<larcv::Pixel2DCluster> >
  SimCh2Pixel2DCluster(const std::vector<larcv::ImageMeta>& meta_v,
		       const std::vector<size_t>& row_compression_v,
		       const std::vector<size_t>& col_compression_v,
		       const std::vector<supera::LArSimCh_t>& sch_v,
		       const std::vector<size_t>& trackid2cluster,
		       const int time_offset);
}
#endif
