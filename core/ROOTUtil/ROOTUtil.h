#ifndef __ROOTUTILS_h__
#define __ROOTUTILS_h__

/* **************************************   
 * ROOTUtils
 *
 * Utility commands, usually to
 * perform quick viewing or dumping of
 * images
 *
 * authors: twongj01@tufts.edu
 *
 * 5/9/2018: first draft
 * ************************************** */

#include <string>
#include <vector>

#include "TCanvas.h"
#include "TH2D.h"
#include "TGraph.h"

#include "DataFormat/Image2D.h"
#include "DataFormat/Pixel2DCluster.h"
#include "DataFormat/ImageMeta.h"

namespace larcv {

  // conversion into 2D histogram for plotting
  TH2D as_th2d( const larcv::Image2D& img, std::string histname );
  
  std::vector< TH2D > as_th2d_v( const std::vector<larcv::Image2D>& img_v, std::string histstemname );

  // PixelCluster as tgraph
  TGraph as_tgraph( const larcv::Pixel2DCluster& cluster, 
                    bool output_is_wiretick,
                    const larcv::ImageMeta* meta );
  
  // PixelCluster as contour
  TGraph as_contour_tgraph( const larcv::Pixel2DCluster& cluster, 
                            bool output_is_wiretick,
                            const larcv::ImageMeta* meta );
  

  // Color Palettes
  void setBlueRedColorPalette();
  
  void setRainbowPalette();
  
  void setRedHeatColorPalette();

  
  
}

#endif
