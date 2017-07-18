#ifndef LARBYSUTILS_H
#define LARBYSUTILS_H

#include "Geo2D/Core/Geo2D.h"
#include "Geo2D/Core/Line.h"
#include "Geo2D/Core/Vector.h"
#include "Geo2D/Core/Circle.h"
#include "DataFormat/Image2D.h"

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


  void
  mask_image(Image2D& target, const Image2D& ref);

  std::vector<ImageMeta>
  crop_metas(const std::vector<Image2D>& img_v, const std::vector<ImageMeta>& meta_v);
  
  ImageMeta
  crop_meta(const Image2D& img, const ImageMeta& meta);

  bool InFiducialRegion3D(float X, float Y, float Z,
			  float edge_x=10.0, float edge_y=10.0, float edge_z=10.0);
    
		
}

#endif
/** @} */ // end of doxygen group 

