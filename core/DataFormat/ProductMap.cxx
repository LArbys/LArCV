#ifndef __LARCV_PRODUCTMAP_CXX__
#define __LARCV_PRODUCTMAP_CXX__

#include "ProductMap.h"
#include "Base/larbys.h"
namespace larcv {

  const std::string ProductName(ProductType_t type)
  {
    switch(type) {
    case kProductImage2D: return "image2d";
    case kProductROI:     return "partroi";
    default:
      throw larbys("Unsupported type! Implement DataFormat/ProductMap.cxx!");
    }
  }

  template<> ProductType_t ProductType< larcv::Image2D > () { return kProductImage2D; }
  template<> ProductType_t ProductType< larcv::ROI     > () { return kProductROI;     }

}

#endif
