#ifndef ASTARUTILS_H
#define ASTARUTILS_H

// larcv
#include "DataFormat/Image2D.h"

// ROOT
#include "TVector3.h"

// opencv
#ifndef __CLING__
#ifndef __CINT__
#include  <opencv2/opencv.hpp>
#endif
#endif
#include <opencv2/core/core.hpp>

namespace larcv {

  void
  ProjectTo3D(const ImageMeta& meta,
	      double parent_x,
	      double parent_y,
	      double parent_z,
	      double parent_t,
	      uint plane,
	      double& xpixel, double& ypixel);
  
  std::vector<std::vector<cv::Point_<int> > > TrackToPixels(const std::vector<TVector3>& xyz_v,
							      const std::vector<ImageMeta>& meta_v);
  
}

#endif
/** @} */ // end of doxygen group 

