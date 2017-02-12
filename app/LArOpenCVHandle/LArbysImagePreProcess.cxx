#ifndef __LARBYSIMAGEPREPROCESS_CXX__
#define __LARBYSIMAGEPREPROCESS_CXX__

#include "LArbysImagePreProcess.h"
#include "Base/ConfigManager.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  static LArbysImagePreProcessProcessFactory __global_LArbysImagePreProcessProcessFactory__;

  LArbysImagePreProcess::LArbysImagePreProcess(const std::string name)
    : ProcessBase(name)
  {}
  void LArbysImagePreProcess::configure(const PSet& cfg)
  {}
  void LArbysImagePreProcess::initialize()
  {}
  bool LArbysImagePreProcess::process(IOManager& mgr)
  {
    return true;
  }
  void LArbysImagePreProcess::finalize()
  {}
}
#endif
