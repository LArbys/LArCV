#ifndef LARCV_EVENTVOXEL2D_CXX
#define LARCV_EVENTVOXEL2D_CXX

#include "EventVoxel2D.h"

namespace larcv {

  /// Global larcv::EventClusterPixel2DFactory to register EventClusterPixel2D
  static EventClusterPixel2DFactory __global_EventClusterPixel2DFactory__;

  /// Global larcv::EventSparseTensor2DFactory to register EventSparseTensor2D
  static EventSparseTensor2DFactory __global_EventSparseTensor2DFactory__;

  //
  // EventClusterPixel2D
  //
  const larcv::ClusterPixel2D&
  EventClusterPixel2D::cluster_pixel_2d(const ProjectionID_t id) const
  {
    for(auto const& cluster : _cluster_v) {
      if(cluster.meta().id()!=id) continue;
      return cluster;
    }

    std::cerr << "EventClusterPixel2D does not hold any ClusterPixel2D for ProjectionID_t " << id << std::endl;
    throw larbys();
  }

  void EventClusterPixel2D::emplace(larcv::ClusterPixel2D&& clusters)
  {
    for(size_t i=0;i<_cluster_v.size();++i) {
      if(_cluster_v[i].meta().id() != clusters.meta().id()) continue;
      _cluster_v[i] = clusters;
      return;
    }
    _cluster_v.emplace_back(clusters);
  }

  void EventClusterPixel2D::set(const larcv::ClusterPixel2D& clusters) 
  {
    for(size_t i=0;i<_cluster_v.size();++i) {
      if(_cluster_v[i].meta().id() != clusters.meta().id()) continue;
      _cluster_v[i] = clusters;
      return;
    }
    _cluster_v.push_back(clusters);    
  }
  
  void EventClusterPixel2D::emplace(larcv::VoxelSetArray&& clusters, larcv::ImageMeta&& meta)
  {
    larcv::ClusterPixel2D source(std::move(clusters),std::move(meta));
    emplace(std::move(source));
  }

  void EventClusterPixel2D::set(const larcv::VoxelSetArray& clusters, const larcv::ImageMeta& meta)
  {
    larcv::ClusterPixel2D source(clusters);
    source.meta(meta);
    emplace(std::move(source));
  }

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
