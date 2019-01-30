#ifndef PGRAPH_CXX
#define PGRAPH_CXX

#include <assert.h>
#include <sstream>
#include "PGraph.h"
#include "larcv/core/Base/larbys.h"

namespace larcv {

  void PGraph::Clear()
  { _part_v.clear(); _pcluster_idx_v.clear(); }

  void PGraph::Add(const larcv::ROI& roi,size_t pcluster_idx)
  { _part_v.push_back(roi); _pcluster_idx_v.push_back(pcluster_idx); }
  
  void PGraph::Emplace(larcv::ROI&& roi,size_t pcluster_idx)
  { _part_v.emplace_back(std::move(roi)); _pcluster_idx_v.push_back(pcluster_idx); }
  
  void PGraph::Set(const std::vector<larcv::ROI>& roi_v, const std::vector<size_t>& pcluster_idx_v)
  {
    assert(roi_v.size() == pcluster_idx_v.size());
    _part_v = roi_v;
    _pcluster_idx_v = pcluster_idx_v;
  }
  
  void PGraph::Emplace(std::vector<larcv::ROI>&& roi_v, std::vector<size_t>&& pcluster_idx_v)
  {
    assert(roi_v.size() == pcluster_idx_v.size());
    _part_v = std::move(roi_v);
    _pcluster_idx_v = std::move(pcluster_idx_v);
  }
}

#endif
