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
    case kProductParticle: return "particle";
    case kProductClusterPixel2D: return "cluster2d";
    case kProductSparseTensor2D: return "sparse2d";
    case kProductSparseTensor3D: return "sparse3d";            
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
  template<> ProductType_t ProductType< larcv::Particle   > () { return kProductParticle;   }
  template<> ProductType_t ProductType< larcv::SparseTensor2D   > () { return kProductSparseTensor2D;   }
  template<> ProductType_t ProductType< larcv::SparseTensor3D   > () { return kProductSparseTensor3D;   }      

  std::string GetProductTypeName( ProductType_t prodtype ) {
    if (prodtype<0 || prodtype>=larcv::kProductUnknown)
      return ProductTypeNames_v.back();
    else
      return ProductTypeNames_v[prodtype];
  };

  ProductType_t GetProductTypeID( std::string prodname ) {
    for ( size_t itype=0; itype<kProductUnknown; itype++ )
      if (ProductTypeNames_v[itype]==prodname ) return (ProductType_t)itype;
    return kProductUnknown;
  };
  
}

#endif
