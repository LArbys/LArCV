#ifndef EVENTCLUSTERMASK_CXX
#define EVENTCLUSTERMASK_CXX

#include "EventClusterMask.h"
#include "larcv/core/Base/larbys.h"

namespace larcv {

  /// Global larcv::SBClusterFactory to register ClusterAlgoFactory
  static EventClusterMaskFactory __global_EventClusterMaskFactory__;

  void EventClusterMask::clear()
  {
    EventBase::clear();
    _clustermask_vv.clear();
  }

  const std::vector<ClusterMask>& EventClusterMask::at(size_t id) const
  {
    if( id >= _clustermask_vv.size() ) throw larbys("Invalid request (Index out-o-range)!");
    return _clustermask_vv[id];
  }

  void EventClusterMask::append(const std::vector<ClusterMask>& mask_v)
  {
    _clustermask_vv.push_back(mask_v);
    // _clustermask_vv.back().index((_clustermask_vv.size()-1)); //clustermasks don't have index at present
  }

  void EventClusterMask::emplace(std::vector<ClusterMask>&& mask_v)
  {
    _clustermask_vv.emplace_back(std::move(mask_v));
    // _clustermask_vv.back().index((_clustermask_vv.size()-1)); //clustermasks don't have index at present
  }

  void EventClusterMask::emplace(std::vector<std::vector<larcv::ClusterMask>>&& mask_vv)
  {
    _clustermask_vv = std::move(mask_vv);
    // for(int i=0; i<_clustermask_vv.size(); ++i) _clustermask_vv[i].index(i); //clustermasks don't have index at present
  }



  std::vector<ClusterMask>& EventClusterMask::mod_mask_v_at(int id) {
    if( id >= _clustermask_vv.size() ) throw larbys("Invalid request (Index out-o-range)!");
    return _clustermask_vv[id];
  }

  void EventClusterMask::reserve(size_t s) {
    _clustermask_vv.reserve(s);
  }

}

#endif
