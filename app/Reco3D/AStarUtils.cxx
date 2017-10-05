#ifndef ASTARUTILS_CXX
#define ASTARUTILS_CXX

#include "AStarUtils.h"
#include "LArUtil/GeometryHelper.h"
#include "LArUtil/LArProperties.h"
#include <numeric>
#include <assert.h>

namespace larcv {

  void
  ProjectTo3D(const ImageMeta& meta,
	      double parent_x,
	      double parent_y,
	      double parent_z,
	      double parent_t,
	      uint plane,
	      double& xpixel, double& ypixel) 
  {
    
    auto geohelp = larutil::GeometryHelper::GetME();//Geohelper from LArLite
    auto larpro  = larutil::LArProperties::GetME(); //LArProperties from LArLite

    auto vtx_2d = geohelp->Point_3Dto2D(parent_x, parent_y, parent_z, plane );
    
    double x_compression  = (double)meta.width()  / (double)meta.cols();
    double y_compression  = (double)meta.height() / (double)meta.rows();
    xpixel = (vtx_2d.w/geohelp->WireToCm() - meta.tl().x) / x_compression;
    ypixel = (((parent_x/larpro->DriftVelocity() + parent_t/1000.)*2+3200)-meta.br().y)/y_compression;
  }  
  
}

#endif
