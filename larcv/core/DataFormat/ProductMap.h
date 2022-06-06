 
#ifndef __LARCV_PRODUCTMAP_H__
#define __LARCV_PRODUCTMAP_H__

#include <string>
#include "DataFormatTypes.h"
namespace larcv {

  const char *  ProductName(ProductType_t type);

  template<class T> ProductType_t ProductType();

  class Image2D;
  template<> ProductType_t ProductType< larcv::Image2D  > ();
  class ROI;
  template<> ProductType_t ProductType< larcv::ROI      > ();
  class ChStatus;
  template<> ProductType_t ProductType< larcv::ChStatus > ();
  class Pixel2D;
  template<> ProductType_t ProductType< larcv::Pixel2D  > ();
  class PGraph;
  template<> ProductType_t ProductType< larcv::PGraph   > ();
  class Voxel3D;
  template<> ProductType_t ProductType< larcv::Voxel3D  > ();
  class SparseImage;
  template<> ProductType_t ProductType< larcv::SparseImage  > ();
  class ClusterMask;
  template<> ProductType_t ProductType< larcv::ClusterMask  > ();
  class Particle;
  template<> ProductType_t ProductType< larcv::Particle  > ();
  class SparseTensor2D;
  template<> ProductType_t ProductType< larcv::SparseTensor2D  > ();
  class SparseTensor3D;
  template<> ProductType_t ProductType< larcv::SparseTensor3D  > ();
  class ClusterVoxel3D;
  template<> ProductType_t ProductType< larcv::ClusterVoxel3D  > ();
  
  // python-friendly alternative functions  
  std::string   GetProductTypeName( ProductType_t prodtype );
  ProductType_t GetProductTypeID( std::string prodname );
  
}

#endif
