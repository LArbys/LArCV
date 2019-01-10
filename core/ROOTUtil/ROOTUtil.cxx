#include "ROOTUtil.h"

#include <sstream>

#include "TStyle.h"
#include "TColor.h"

namespace larcv {


  TH2D as_th2d( const larcv::Image2D& img, std::string histname ) {

    const larcv::ImageMeta& meta = img.meta();
    
    // convert to th2d
    std::string hname = histname;
    if ( hname=="" )
      hname = "_temp_image2d_";

    TH2D h( hname.c_str(), "",
            meta.cols(), meta.min_x(), meta.max_x(),
            meta.rows(), meta.min_y(), meta.max_y() );
    
    for (size_t r=0; r<meta.rows(); r++) {
      for (size_t c=0; c<meta.cols(); c++) {
	h.SetBinContent( c+1, meta.rows()-r, img.pixel( r, c ) ); // reverse time order for larcv1 ...
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

  /**
   * Use TGraph to represent points of a Pixel2DCluster
   *
   * @param[in] cluster Pixel2DCluster. (x,y) values assumed to be (col,row)
   * @param[in] output_is_wiretick true if (x,y) values for TGraph should be (wire,tick) coordinates
   * @param[in] meta pointer to ImageMeta used to convert coordinates. if nullptr, no conversion.
   * @return TGraph with coordinaes determined by output_is_wiretick
   */
  TGraph as_tgraph( const larcv::Pixel2DCluster& cluster,
                    bool output_is_wiretick,
                    const larcv::ImageMeta* meta ) {
    size_t npts = cluster.size();
    TGraph g( npts );
    
    for ( size_t ipt=0; ipt<npts; ipt++ ) {
      auto const& pix = cluster[ipt];
      if ( meta==nullptr || !output_is_wiretick ) {
        // no conversion
        g.SetPoint( ipt, (float)pix.X(), (float)pix.Y() );
      }
      else if ( meta && output_is_wiretick) {
        // has meta
        if ( pix.X()<meta->cols() && pix.Y()<meta->rows() ) 
          g.SetPoint( ipt, meta->pos_x( pix.X() ), meta->pos_y( pix.Y() ) );
      }
    }
    
    return g;
  }
  
  /**
   * Use TGraph to represent closed contour stored as a Pixel2DCluster
   *
   * @param[in] cluster Pixel2DCluster whose order is the assumed one for the contour.
   * @param[in] output_is_wiretick true if (x,y) values for TGraph should be (wire,tick) coordinates
   * @param[in] meta pointer to ImageMeta used to convert coordinates. if nullptr, no conversion.
   * @return TGraph with coordinaes determined by output_is_wiretick
   */
  TGraph as_contour_tgraph( const larcv::Pixel2DCluster& cluster, 
                            bool output_is_wiretick,
                            const larcv::ImageMeta* meta ) {
    
    TGraph g = as_tgraph( cluster, output_is_wiretick, meta );
    size_t npts = g.GetN();
    g.Set( npts+1 );
    Double_t x,y;    
    g.GetPoint( 0, x, y );
    g.SetPoint( npts, x, y );
    
    return g;
  }
  
  /**
   * Set ROOT to draw next color plots in a blue-to-red palette useful for correlations
   */
  void setBlueRedColorPalette() 
  {
    // A colour palette that goes blue->white->red, useful for
    // correlation matrices
    const int NRGBs = 3;
    const int n_color_contours = 256;
    static bool initialized=false;
    static int* colors=new int[n_color_contours];
    
    if(!initialized){
      gStyle->SetNumberContours(n_color_contours);
      Double_t stops[NRGBs] = { 0.00, 0.50, 1.00};
      Double_t red[NRGBs]   = { 0.00, 1.00, 1.00};
      Double_t green[NRGBs] = { 0.00, 1.00, 0.00};
      Double_t blue[NRGBs]  = { 1.00, 1.00, 0.00};
      int colmin=TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, n_color_contours);
      for(uint i=0; i<n_color_contours; ++i) colors[i]=colmin+i;
      
      initialized=true;
    }
    gStyle->SetNumberContours(n_color_contours);
    gStyle->SetPalette(n_color_contours, colors);
  }

  /**
   * Set ROOT to draw next color plots in a white(zero) to rainbow color
   */  
  void setRainbowColorPalette()
  {
    const int NRGBs = 7;
    static bool initialized=false;
    const int n_color_contours = 999;    
    static int* colors=new int[n_color_contours];
    
    if(!initialized){
      gStyle->SetNumberContours(n_color_contours);
      Double_t stops[NRGBs] = { 0.00, 0.05, 0.23, 0.45, 0.60, 0.85, 1.00 };
      Double_t red[NRGBs]   = { 1.00, 0.00, 0.00, 0.00, 1.00, 1.00, 0.33 };
      Double_t green[NRGBs] = { 1.00, 1.00, 0.30, 0.40, 1.00, 0.00, 0.00 };
      Double_t blue[NRGBs]  = { 1.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00 };
      int colmin=TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, n_color_contours);  
      for(uint i=0; i<n_color_contours; ++i) colors[i]=colmin+i;
      
      initialized=true;
    }
    gStyle->SetNumberContours(n_color_contours);
    gStyle->SetPalette(n_color_contours, colors);
  }

  /**
   * Set ROOT to draw in a heat-map color scheme with red the largest value
   */    
  void setRedHeatColorPalette()
  {
    const int NRGBs = 9;
    static bool initialized=false;
    const  Int_t n_color_contours = 999;
    static int* colors=new int[n_color_contours];
    
    if(!initialized){
      // White -> red
      Double_t stops[NRGBs] = { 0.00, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000};
      Double_t red[NRGBs]   = { 1.00, 1.00, 0.99, 0.99, 0.98, 0.94, 0.80, 0.65, 0.40 };
      Double_t green[NRGBs] = { 0.96, 0.88, 0.73, 0.57, 0.42, 0.23, 0.09, 0.06, 0.00 };
      Double_t blue[NRGBs]  = { 0.94, 0.82, 0.63, 0.45, 0.29, 0.17, 0.11, 0.08, 0.05 };
      int colmin=TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, n_color_contours);
      for(uint i=0; i<n_color_contours; ++i) colors[i]=colmin+i;
      
      initialized=true;
    }
    gStyle->SetNumberContours(n_color_contours);
    gStyle->SetPalette(n_color_contours, colors);    
  }
  
}
