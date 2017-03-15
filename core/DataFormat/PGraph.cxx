#ifndef PGRAPH_CXX
#define PGRAPH_CXX

#include <sstream>
#include "PGraph.h"
#include "Base/larbys.h"

namespace larcv {

  void PGraph::Clear()
  { _part_v.clear(); _pcluster_idx_v.clear(); }
}

#endif
