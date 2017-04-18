#ifndef LARBYSUTILS_H
#define LARBYSUTILS_H

#include "Geo2D/Core/Geo2D.h"
#include "Geo2D/Core/Line.h"
#include "Geo2D/Core/Vector.h"
#include "Geo2D/Core/Circle.h"
#include "DataFormat/ImageMeta.h"

namespace larcv {

  cv::Rect
  Get2DRoi(const ImageMeta& meta,
	   const ImageMeta& bb);

  
  void
  Project3D(const ImageMeta& meta,
	    double parent_x,
	    double parent_y,
	    double parent_z,
	    double parent_t,
	    uint plane,
	    double& xpixel, double& ypixel);


  geo2d::Vector<float>
  Intersection(const geo2d::HalfLine<float>& hline,
	       const cv::Rect& rect);

  template <class T>
  T Mean(std::vector<T> v);
  
  template <class T>
  T STD(std::vector<T> v);
  
}

#endif
/** @} */ // end of doxygen group 

