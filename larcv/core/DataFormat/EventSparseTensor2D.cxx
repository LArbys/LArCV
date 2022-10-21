#ifndef __EVENT_SPARSE_TENSOR_3D_CXX__
#define __EVENT_SPARSE_TENSOR_3D_CXX__

#include "EventSparseTensor2D.h"

namespace larcv {

  /// Global larcv::EventSparseTensor3DFactory to register EventSparseTensor3D
  static EventSparseTensor2DFactory __global_EventSparseTensor2DFactory__;
  
  //
  // EventSparseTensor2D
  //
  const larcv::SparseTensor2D&
  EventSparseTensor2D::sparse_tensor_2d(const ProjectionID_t id) const
  {
    for(auto const& tensor : _tensor_v) {
      if(tensor.meta().id() != id) continue;
      return tensor;
    }

    std::cerr << "EventSparseTensor2D does not hold any SparseTensor2D for ProjectionID_t " << id << std::endl;
    throw larbys();
  }

  void EventSparseTensor2D::emplace(larcv::SparseTensor2D&& cluster)
  {
    for(size_t i=0; i<_tensor_v.size(); ++i) {
      if(_tensor_v[i].meta().id() != cluster.meta().id()) continue;
      _tensor_v[i] = cluster;
      return;
    }
    _tensor_v.emplace_back(cluster);
  }

  void EventSparseTensor2D::set(const larcv::SparseTensor2D& cluster) 
  {
    for(size_t i=0; i<_tensor_v.size(); ++i) {
      if(_tensor_v[i].meta().id() != cluster.meta().id()) continue;
      _tensor_v[i] = cluster;
      return;
    }
    _tensor_v.push_back(cluster);
  }
  
  void EventSparseTensor2D::emplace(larcv::VoxelSet&& cluster, larcv::ImageMeta&& meta)
  {
    larcv::SparseTensor2D source(std::move(cluster),std::move(meta));
    emplace(std::move(source));
  }

  void EventSparseTensor2D::set(const larcv::VoxelSet& cluster, const larcv::ImageMeta& meta)
  {
    larcv::SparseTensor2D source(cluster);
    source.meta(meta);
    emplace(std::move(source));
  }

}

#endif
