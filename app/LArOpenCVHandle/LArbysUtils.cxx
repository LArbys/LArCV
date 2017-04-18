#ifndef LARBYSUTILS_CXX
#define LARBYSUTILS_CXX

#include "LArbysUtils.h"
#include "LArUtil/GeometryHelper.h"
#include "LArUtil/LArProperties.h"
#include <numeric>
namespace larcv {
  
  cv::Rect Get2DRoi(const ImageMeta& meta,
		    const ImageMeta& bb) {

    //bb == ROI on plane META
    
    float x_compression = meta.width()  / (float) meta.cols();
    float y_compression = meta.height() / (float) meta.rows();
    

    int x=(bb.bl().x - meta.bl().x)/x_compression;
    int y=(bb.bl().y - meta.bl().y)/y_compression;
    
    int dx=(bb.tr().x-bb.bl().x)/x_compression;
    int dy=(bb.tr().y-bb.bl().y)/y_compression;

    return cv::Rect(x,y,dx,dy);
  }

  
  void Project3D(const ImageMeta& meta,
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
    
    double x_compression  = meta.width()  / meta.cols();
    double y_compression  = meta.height() / meta.rows();
    xpixel = (vtx_2d.w/geohelp->WireToCm() - meta.tl().x) / x_compression;
    ypixel = (((parent_x/larpro->DriftVelocity() + parent_t/1000.)*2+3200)-meta.br().y)/y_compression;
  }
  

  // Input is track line and 4 linesegment consists the ROI
  // Function intersection finds the intersecion point of track
  // and ROI in particle direction
  
  geo2d::Vector<float> Intersection (const geo2d::HalfLine<float>& hline,
				     const cv::Rect& rect)
  {
    
    geo2d::LineSegment<float> ls1(geo2d::Vector<float>(rect.x           ,rect.y            ),
				  geo2d::Vector<float>(rect.x+rect.width,rect.y            ));

    geo2d::LineSegment<float> ls2(geo2d::Vector<float>(rect.x+rect.width,rect.y            ),
				  geo2d::Vector<float>(rect.x+rect.width,rect.y+rect.height));

    geo2d::LineSegment<float> ls3(geo2d::Vector<float>(rect.x+rect.width,rect.y+rect.height),
				  geo2d::Vector<float>(rect.x           ,rect.y+rect.height));

    geo2d::LineSegment<float> ls4(geo2d::Vector<float>(rect.x           ,rect.y+rect.height),
				  geo2d::Vector<float>(rect.x           ,rect.y            ));
    

    geo2d::Vector<float> pt(-1,-1);
    
    try {
      auto x = hline.x(ls1.pt1.y);
      pt.x=x;
      pt.y=ls1.pt1.y;
      if ( pt.x <= ls1.pt2.x and pt.x >= ls1.pt1.x ) return pt;
    } catch(...){}

    try {
      auto y = hline.y(ls2.pt1.x);
      pt.x=ls2.pt1.x;
      pt.y=y;
      if ( pt.y <= ls2.pt2.y and pt.y >= ls2.pt1.y ) return pt;
    } catch(...){}


    try {
      auto x = hline.x(ls3.pt1.y);
      pt.x=x;
      pt.y=ls3.pt1.y;
      if ( pt.x >= ls3.pt2.x and pt.x <= ls3.pt1.x ) return pt;
    } catch(...){}


    try {
      auto y= hline.y(ls4.pt1.x);
      pt.x=ls4.pt1.x;
      pt.y=y;
      if ( pt.y >= ls4.pt2.y and pt.y <= ls4.pt1.y ) return pt;
    } catch(...){}

    return pt;
    
  }


  template <class T>
  T Mean(std::vector<T> v)
  {
    T sum = std::accumulate(v.begin(), v.end(), 0.0);
    T mean = sum / v.size();
    
    return mean;
  }

  template <class T>
  T STD(std::vector<T> v)
  {
    T sum = std::accumulate(v.begin(), v.end(), 0.0);
    T mean = sum / v.size();
    T sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    T stdev = std::sqrt(sq_sum / v.size() - mean * mean);
    
    return stdev;
  }

  
}
#endif
