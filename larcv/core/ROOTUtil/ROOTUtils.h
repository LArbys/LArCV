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

#include "larcv/core/DataFormat/Image2D.h"

namespace larcv {

  class rootutils {
  public:
    rootutils() {};
    ~rootutils() {};

    
    static TH2D as_th2d( const larcv::Image2D& img, std::string histname );
  
    static std::vector< TH2D > as_th2d_v( const std::vector<larcv::Image2D>& img_v, std::string histstemname );

  };
  
}

#endif
