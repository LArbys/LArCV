#ifndef __DBSCAN_CXX__
#define __DBSCAN_CXX__

#include "DBSCAN.h"

namespace larcv {

  static DBSCANProcessFactory __global_DBSCANProcessFactory__;

  DBSCAN::DBSCAN(const std::string name)
    : ProcessBase(name)
  {}
    
  void DBSCAN::configure(const PSet& cfg)
  {}

  void DBSCAN::initialize()
  {}

  bool DBSCAN::process(IOManager& mgr)
  {}

  void DBSCAN::finalize()
  {}

}
#endif
