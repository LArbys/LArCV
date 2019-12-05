#ifndef EVENTROI_CXX
#define EVENTROI_CXX

#include "EventROI.h"
#include <sstream>

namespace larcv{

  /// Global larcv::SBClusterFactory to register ClusterAlgoFactory
  static EventROIFactory __global_EventROIFactory__;

  void EventROI::clear()
  {
    EventBase::clear();
    _part_v.clear();
  }

  const ROI& EventROI::at(ImageIndex_t id) const
  {
    if( id >= _part_v.size() ) {
      std::stringstream msg;
      msg << "EventROI.cxx" << ".L" << __LINE__ << ":"
	  << "Invalid request (ImageIndex_t[" << id << "] out-o-range [>=" << _part_v.size() << "] )!"
	  << std::endl;
      throw larbys(msg.str());
    }
    return _part_v[id];
  }

  void EventROI::Append(const ROI& part)
  {
    _part_v.push_back(part);
    _part_v.back().Index((ROIIndex_t)(_part_v.size()-1));
  }

  void EventROI::Set(const std::vector<larcv::ROI>& part_v)
  {
    _part_v.clear();
    _part_v.reserve(part_v.size());
    for(auto const& p : part_v) Append(p);
  }

  void EventROI::Emplace(ROI&& part)
  {
    _part_v.emplace_back(part);
    _part_v.back().Index((ROIIndex_t)(_part_v.size()-1));
  }

  void EventROI::Emplace(std::vector<larcv::ROI>&& part_v)
  {
    _part_v = std::move(part_v);
    for(size_t i=0; i<_part_v.size(); ++i) _part_v[i].Index((ROIIndex_t)i);
  }
}

#endif
