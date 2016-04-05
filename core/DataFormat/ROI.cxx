#ifndef __ROI_CXX__
#define __ROI_CXX__

#include "ROI.h"
#include <set>
namespace larcv {

  const ImageMeta& ROI::BB(PlaneID_t plane) const
  {
    for(auto const& bb : _bb_v) { if(bb.plane() == plane) return bb; }
    throw larbys("Could not locate requested plane's bounding box!");
  }

  void ROI::AppendBB(const larcv::ImageMeta& bb)
  {
    // Make sure this is not duplicate in terms of plane ID
    std::set<larcv::PlaneID_t> ids;
    for(auto const& meta : _bb_v) ids.insert(meta.plane());
    if(ids.find(bb.plane()) != ids.end()) throw larbys("Cannot have ImageMeta of duplicate plane!");
    _bb_v.push_back(bb);
  }
  
  void ROI::SetBB(const std::vector<larcv::ImageMeta>& bb_v)
  {
    // Make sure this is not duplicate in terms of plane ID
    std::set<larcv::PlaneID_t> ids;
    for(auto const& meta : bb_v) ids.insert(meta.plane());
    if(ids.size() != bb_v.size()) throw larbys("Cannot have ImageMeta of duplicate plane!");
    _bb_v = bb_v;
  }

}

#endif
