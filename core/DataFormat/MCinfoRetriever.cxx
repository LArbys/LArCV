#ifndef __MCINFORETRIEVER_CXX__
#define __MCINFORETRIEVER_CXX__

#include "MCinfoRetriever.h"

namespace larcv {

  static MCinfoRetrieverProcessFactory __global_MCinfoRetrieverProcessFactory__;

  MCinfoRetriever::MCinfoRetriever(const std::string name)
    : ProcessBase(name)
  {}
    
  void MCinfoRetriever::configure(const PSet& cfg)
  {}

  void MCinfoRetriever::initialize()
  {}

  bool MCinfoRetriever::process(IOManager& mgr)
  {}

  void MCinfoRetriever::finalize()
  {}

}
#endif
