#ifndef EVENTCHSTATUS_CXX
#define EVENTCHSTATUS_CXX

#include "EventChStatus.h"
#include "Base/larbys.h"

namespace larcv {

  /// Global larcv::SBClusterFactory to register ClusterAlgoFactory
  static EventChStatusFactory __global_EventChStatusFactory__;

  void EventChStatus::clear()
  {
    EventBase::clear();
    _status_m.clear();
  }

  const ChStatus& EventChStatus::at(PlaneID_t id) const
  {
    auto iter = _status_m.find(id);
    if( iter == _status_m.end() ) throw larbys("Invalid request (PlaneID_t not found)!");
    return (*iter).second;
  }

  void EventChStatus::Insert(PlaneID_t id, const ChStatus& status)
  { _status_m[id] = status; }

  void EventChStatus::Emplace(PlaneID_t id, ChStatus&& img)
  { _status_m.emplace(id,std::move(img)); }

}

#endif
