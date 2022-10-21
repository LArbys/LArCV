#ifndef LARCV_EVENTVOXEL2D_CXX
#define LARCV_EVENTVOXEL2D_CXX

#include "EventVoxel2D.h"

namespace larcv {

  /// Global larcv::EventClusterPixel2DFactory to register EventClusterPixel2D
  static EventClusterPixel2DFactory __global_EventClusterPixel2DFactory__;

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


  
}

#endif
