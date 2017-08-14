#ifndef ASTARUTILS_H
#define ASTARUTILS_H

#include "DataFormat/Image2D.h"

namespace larcv {

  void
  ProjectTo3D(const ImageMeta& meta,
	      double parent_x,
	      double parent_y,
	      double parent_z,
	      double parent_t,
	      uint plane,
	      double& xpixel, double& ypixel);

}

#endif
/** @} */ // end of doxygen group 

