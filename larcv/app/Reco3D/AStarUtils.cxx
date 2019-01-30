#ifndef ASTARUTILS_CXX
#define ASTARUTILS_CXX

#include "AStarUtils.h"
#include "LArUtil/GeometryHelper.h"
#include "LArUtil/LArProperties.h"
#include <numeric>
#include <cassert>

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
  
#ifdef LARCV_OPENCV

  std::vector<std::vector<cv::Point_<int> > > TrackToPixels(const std::vector<TVector3>& xyz_v,
							      const std::vector<ImageMeta>& meta_v) {

    std::vector<std::vector<cv::Point_<int> > > res_vv;
    res_vv.resize(3);
    assert(res_vv.size() == meta_v.size());

    for(auto& res_v : res_vv) res_v.resize(xyz_v.size());

    double xpixel = kINVALID_DOUBLE;
    double ypixel = kINVALID_DOUBLE;

    for(size_t tid=0; tid<xyz_v.size(); ++ tid) {
      const auto& trk_pt = xyz_v[tid];
      for(size_t plane=0; plane<3; ++plane) {

	xpixel = kINVALID_DOUBLE;
	ypixel = kINVALID_DOUBLE;

	auto& res_v = res_vv[plane];
	const auto& meta = meta_v[plane];
	ProjectTo3D(meta, trk_pt.X(), trk_pt.Y(), trk_pt.Z(), 0.0, plane, xpixel, ypixel);
	
	int xx = (int)(xpixel+0.5);
	int yy = (int)(ypixel+0.5);
	yy = meta.rows() - yy - 1;
	
	res_v[tid] = cv::Point(yy,xx);
      }
    }    

    return res_vv;
  }

#endif // if LARCV_OPENCV

}

#endif
