#ifndef ASTARUTILS_H
#define ASTARUTILS_H

// larcv
#include "larcv/core/DataFormat/Image2D.h"

// ROOT
#include "TVector3.h"

// opencv
#ifdef LARCV_OPENCV
#ifndef __CLING__
#ifndef __CINT__
#include  <opencv2/opencv.hpp>
#endif
#endif
#include <opencv2/core/core.hpp>

#endif //opencv

namespace larcv {

  void
  ProjectTo3D(const ImageMeta& meta,
	      double parent_x,
	      double parent_y,
	      double parent_z,
	      double parent_t,
	      uint plane,
	      double& xpixel, double& ypixel);

#ifdef LARCV_OPENCV
  std::vector<std::vector<cv::Point_<int> > > TrackToPixels(const std::vector<TVector3>& xyz_v,
							      const std::vector<ImageMeta>& meta_v);
#endif

}

#endif
/** @} */ // end of doxygen group
