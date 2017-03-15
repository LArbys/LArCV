#ifndef __IMAGEMODUTILS_CXX__
#define __IMAGEMODUTILS_CXX__

#include "ImageModUtils.h"

namespace larcv {

  Image2D as_image2d(const Pixel2DCluster& pcluster)
  {
    auto meta = pcluster.bounds();
    Image2D res(meta);
    res.paint(0.);
    for(auto const& px : pcluster)
      res.set_pixel( meta.max_y() - px.Y(),
		     px.X() - meta.min_x(),
		     px.Intensity() );
    return res;
  }

  Image2D as_image2d(const Pixel2DCluster& pcluster,
		     size_t target_rows, size_t target_cols)
  {
    // find charge centroid
    double mean_x=0;
    double mean_y=0;
    double nonzero_npt=0;
    for(auto const& px : pcluster) {
      if(px.Intensity() <= 0) continue;
      mean_x += px.X();
      mean_y += px.Y();
      nonzero_npt += 1.;
    }
    mean_x /= nonzero_npt;
    mean_y /= nonzero_npt;

    // compute the origin from centroid position
    double origin_x = mean_x - ((double)target_cols)/2.;
    double origin_y = mean_y + ((double)target_rows)/2.;

    ImageMeta meta(target_cols, target_rows,
		   target_rows, target_cols,
		   origin_x, origin_y,
		   larcv::kINVALID_PLANE);

    Image2D res(meta);
    for(auto const& px : pcluster) {
      int col = (int)((double)(px.X()) - origin_x + 0.5);
      if(col < 0 || col >= (int)(target_cols)) continue;

      int row = (int)(origin_y - (double)(px.Y()) - 0.5);
      if(row < 0 || row >= (int)(target_rows)) continue;

      res.set_pixel(row,col,px.Intensity());
    }
    
    return res;
  }
}

#endif
