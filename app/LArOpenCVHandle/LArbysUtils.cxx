#ifndef LARBYSUTILS_CXX
#define LARBYSUTILS_CXX

#include "LArbysUtils.h"
#include "LArUtil/GeometryHelper.h"
#include "LArUtil/LArProperties.h"
#include <numeric>

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
    
    double x_compression  = meta.width()  / meta.cols();
    double y_compression  = meta.height() / meta.rows();
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

  /*
  Image2D
  embed_image(const Image2D& ref, const Image2D& target) {

    // get the image size
    int orig_rows = target.meta().rows();
    int orig_cols = target.meta().cols();

    int ref_rows = ref.meta().rows();
    int ref_cols = ref.meta().cols();
    
    if ( orig_rows > ref_rows ) {
      LARCV_ERROR() << "Image is taller than Embedding image (" << orig_rows << ">" << fOutputRows << ")" << std::endl;
      throw larbys();
    }
    if ( orig_cols > ref_cols ) {
      LARCV_ERROR() << "Image is wider than Embedding image (" << orig_cols << ">" << fOutputCols <<")" << std::endl;
      throw larbys();
    }
    
    // get the origin
    float topleft_x = target.meta().min_x();
    float topleft_y = target.meta().max_y();
    
    // get width
    float width_per_pixel  = target.meta().pixel_width();
    float height_per_pixel = target.meta().pixel_height();
    
    // get new width, keeping same pixel scales
    float new_width  = float(fOutputCols)*width_per_pixel;
    float new_height = float(fOutputRows)*height_per_pixel;
    
    // new origin
    int offset_row = 0.5*(ref_rows - orig_rows);
    int offset_col = 0.5*(ref_cols - orig_cols);
    float new_topleft_x = topleft_x - offset_row*width_per_pixel;
    float new_topleft_y = topleft_y + offset_col*height_per_pixel;
    
    // define the new meta
    larcv::ImageMeta new_meta( new_width, new_height,
			       ref_rows, ref_cols,
			       new_topleft_x, new_topleft_y, 
			       target.meta().plane() );
    
      // define the new image
      larcv::Image2D new_image( new_meta );
      new_image.paint( 0.0 );
      new_image.overlay( target );

    

  }
  */
}

#endif
