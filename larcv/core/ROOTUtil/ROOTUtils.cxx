#include "ROOTUtils.h"

#include <sstream>

namespace larcv {


  TH2D as_th2d( const larcv::Image2D& img, std::string histname ) {

    const larcv::ImageMeta& meta = img.meta();
    
    // convert to th2d
    std::string hname = histname;
    if ( hname=="" )
      hname = "_temp_image2d_";

    TH2D h( hname.c_str(), "", meta.cols(), meta.min_x(), meta.max_x(), meta.rows(), meta.min_y(), meta.max_y() );
    for (size_t r=0; r<meta.rows(); r++) {
      for (size_t c=0; c<meta.cols(); c++) {
	h.SetBinContent( c+1, r+1, img.pixel( r, c ) );
      }
    }
    
    return h;
  }

  std::vector< TH2D > as_th2d_v( const std::vector<larcv::Image2D>& img_v, std::string histstemname ) {

    std::vector< TH2D > h_v;
    h_v.reserve( img_v.size() );
    
    std::string stem = histstemname;
    if ( stem=="" )
      stem = "_image2d_v";
    
    for (size_t i=0; i<img_v.size(); i++) {
      
      std::stringstream ss;
      ss << stem << "_" << i;

      TH2D h = as_th2d( img_v[i], ss.str().c_str() );
      h_v.push_back( std::move(h) );
    }

    return h_v;
  }

}
