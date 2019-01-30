#ifndef EVENTPGRAPH_CXX
#define EVENTPGRAPH_CXX

#include "EventPGraph.h"

namespace larcv{

  /// Global larcv::SBClusterFactory to register ClusterAlgoFactory
  static EventPGraphFactory __global_EventPGraphFactory__;

  void EventPGraph::clear()
  {
    EventBase::clear();
    _int_v.clear();
  }

  const PGraph& EventPGraph::at(size_t id) const
  {
    if( id >= _int_v.size() ) throw larbys("Invalid request (ImageIndex_t out-o-range)!");
    return _int_v[id];
  }

  void EventPGraph::Append(const PGraph& part)
  {
    _int_v.push_back(part);
    //_int_v.back().Index(_int_v.size()-1);
  }

  void EventPGraph::Set(const std::vector<larcv::PGraph>& int_v)
  {
    _int_v.clear();
    _int_v.reserve(int_v.size());
    for(auto const& p : int_v) Append(p);
  }

  void EventPGraph::Emplace(PGraph&& part)
  {
    _int_v.emplace_back(part);
    //_int_v.back().Index(_int_v.size()-1);
  }

  void EventPGraph::Emplace(std::vector<larcv::PGraph>&& int_v)
  {
    _int_v = std::move(int_v);
    //for(size_t i=0; i<_int_v.size(); ++i) _int_v[i].Index(i);
  }
}

#endif
