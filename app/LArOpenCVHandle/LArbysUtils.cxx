#ifndef LARBYSUTILS_CXX
#define LARBYSUTILS_CXX

#include "LArbysUtils.h"
#include "LArUtil/GeometryHelper.h"
#include "LArUtil/LArProperties.h"
#include <numeric>
#include <assert.h>

namespace larcv {
  
  cv::Rect
  Get2DRoi(const ImageMeta& meta,
	   const ImageMeta& bb) {

    //bb == ROI on plane META
    
    float x_compression = meta.width()  / (float) meta.cols();
    float y_compression = meta.height() / (float) meta.rows();

    int x=(bb.bl().x - meta.bl().x)/x_compression;
    int y=(bb.bl().y - meta.bl().y)/y_compression;
    
    int dx=(bb.tr().x - bb.bl().x)/x_compression;
    int dy=(bb.tr().y - bb.bl().y)/y_compression;

    return cv::Rect(x,y,dx,dy);
  }

  
  void
  Project3D(const ImageMeta& meta,
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
  

  // Input is track line and 4 linesegment consists the ROI
  // Function intersection finds the intersecion point of track
  // and ROI in particle direction
  
  geo2d::Vector<float>
  Intersection(const geo2d::HalfLine<float>& hline,
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
  
  void
  mask_image(Image2D& target, const Image2D& ref)
  {
    if(target.meta() != ref.meta()) 
      throw larbys("Cannot mask images w/ different meta");

    auto meta = target.meta();
    std::vector<float> data = target.move();
    auto const& ref_vec = ref.as_vector();

    for(size_t i=0; i<data.size(); ++i) { if(ref_vec[i]>0) data[i]=0; }	

    target.move(std::move(data));
  }

  std::vector<ImageMeta>
  crop_metas(const std::vector<Image2D>& img_v, const std::vector<ImageMeta>& meta_v) {
    assert(img_v.size() == meta_v.size());
    std::vector<ImageMeta> res_v;
    res_v.resize(img_v.size());

    for(size_t img_id=0; img_id < img_v.size(); ++img_id)
      res_v[img_id] = crop_meta(img_v[img_id], meta_v[img_id]);
    
    return res_v;
  }
  
  ImageMeta
  crop_meta(const Image2D& img, const ImageMeta& meta)
  {
    // Croppin region must be within the image
    if( meta.min_x() < img.meta().min_x() || meta.min_y() < img.meta().min_y() ||
	meta.max_x() > img.meta().max_x() || meta.max_y() > img.meta().max_y() )
      throw larbys("Cropping region contains region outside the image!");
    
    size_t min_col = img.meta().col(meta.min_x() + img.meta().pixel_width()  / 2. );
    size_t max_col = img.meta().col(meta.max_x() - img.meta().pixel_width()  / 2. );
    size_t min_row = img.meta().row(meta.max_y() - img.meta().pixel_height() / 2. );
    size_t max_row = img.meta().row(meta.min_y() + img.meta().pixel_height() / 2. );
    
    return ImageMeta( (max_col - min_col + 1) * img.meta().pixel_width(),
		      (max_row - min_row + 1) * img.meta().pixel_height(),
		      (max_row - min_row + 1),
		      (max_col - min_col + 1),
		      img.meta().min_x() + min_col * img.meta().pixel_width(),
		      img.meta().max_y() - min_row * img.meta().pixel_height(),
		      img.meta().plane());

  }
}

#endif
