#ifndef __LARCV_PRODUCTMAP_CXX__
#define __LARCV_PRODUCTMAP_CXX__

#include "ProductMap.h"
#include "larcv/core/Base/larbys.h"
#include <sstream>

namespace larcv {

  const char * ProductName(ProductType_t type)
  {
    switch(type) {
    case kProductImage2D:  return "image2d";
    case kProductROI:      return "partroi";
    case kProductChStatus: return "chstatus";
    case kProductPixel2D:  return "pixel2d";
    case kProductPGraph:   return "pgraph";
    case kProductVoxel3D:  return "voxel3d";
    case kProductSparseImage:  return "sparseimg";
    case kProductClusterMask:  return "clustermask";
      //case kProductGeo2D:    return "geo2d";
    default:
      std::stringstream ss;
      ss << "Unsupported type (" << type << ")! Implement DataFormat/ProductMap.cxx!" << std::endl;
      throw larbys(ss.str());
    }
  }

  template<> ProductType_t ProductType< larcv::Image2D  > () { return kProductImage2D;  }
  template<> ProductType_t ProductType< larcv::ROI      > () { return kProductROI;      }
  template<> ProductType_t ProductType< larcv::ChStatus > () { return kProductChStatus; }
  template<> ProductType_t ProductType< larcv::Pixel2D  > () { return kProductPixel2D;  }
  template<> ProductType_t ProductType< larcv::PGraph   > () { return kProductPGraph;   }
  template<> ProductType_t ProductType< larcv::Voxel3D  > () { return kProductVoxel3D;  }
  template<> ProductType_t ProductType< larcv::SparseImage   > () { return kProductSparseImage;   }
  template<> ProductType_t ProductType< larcv::ClusterMask   > () { return kProductClusterMask;   }

}

#endif
