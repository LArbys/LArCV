#ifndef EVENTPARTICLEROI_CXX
#define EVENTPARTICLEROI_CXX

#include "EventParticleROI.h"

namespace larcv{

  void EventParticleROI::clear()
  {
    EventBase::clear();
    _part_v.clear();
  }

  const ParticleROI& EventParticleROI::at(ImageIndex_t id) const
  {
    if( id >= _part_v.size() ) throw larbys("Invalid request (ImageIndex_t out-o-range)!");
    return _part_v[id];
  }

  void EventParticleROI::Append(const ParticleROI& part)
  {
    _part_v.push_back(part);
    _part_v.back().index((ParticleROIIndex_t)(_part_v.size()-1));
  }

  void EventParticleROI::Set(const std::vector<larcv::ParticleROI>& part_v)
  {
    _part_v.clear();
    _part_v.reserve(part_v.size());
    for(auto const& p : part_v) Append(p);
  }

  void EventParticleROI::Emplace(ParticleROI&& part)
  {
    _part_v.emplace_back(part);
    _part_v.back().index((ParticleROIIndex_t)(_part_v.size()-1));
  }

  void EventParticleROI::Emplace(std::vector<larcv::ParticleROI>&& part_v)
  {
    _part_v.clear();
    std::swap(_part_v,part_v);
    for(size_t i=0; i<_part_v.size(); ++i) _part_v[i].Index((ParticleROIIndex_t)i);
  }
}

#endif
